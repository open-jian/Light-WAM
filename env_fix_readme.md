# Environment Fix Readme

This note records the NCCL environment fix for running Light-WAM / FastWAM style
multi-GPU training on this Blackwell workstation.

## Problem

The original environment could crash during multi-GPU NCCL collectives with:

```text
enqueue.cc:1556 NCCL WARN Cuda failure 700 'an illegal memory access was encountered'
ncclUnhandledCudaError
```

The failure was not caused by Light-WAM model code, DeepSpeed, or the LIBERO
dataset. It was reproduced with standalone NCCL/PyTorch communication tests.

Original stack:

```text
GPU: 8x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
Driver: 570.172.08
PyTorch: 2.7.1+cu128
CUDA runtime used by PyTorch: 12.8
Original nvidia-nccl-cu12: 2.26.2
```

Observed behavior before the fix:

```text
native nccl-tests default protocol:
  <= 64KB messages: pass
  >= 128KB messages: fail

PyTorch ProcessGroupNCCL:
  2-GPU broadcast bf16, 589824 elements: fail
  2-GPU all_reduce fp32, 196608 elements: fail
```

Forcing `NCCL_PROTO=LL` avoided the crash, but that was only a workaround. The
cleaner fix is to use a newer NCCL runtime.

## Root Cause

The issue is best explained as a NCCL 2.26.2 runtime bug/pathology on this
Blackwell setup. NCCL's default protocol/algorithm selection could choose a
broken path for larger collectives.

This was confirmed because:

- The crash reproduced outside Light-WAM.
- `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, and `NCCL_NVLS_ENABLE=0` did not
  fix the original failure.
- `NCCL_PROTO=LL` avoided the crash.
- Upgrading runtime NCCL to 2.27.3 fixed the default-protocol tests.

## Fix

The official `fastwam` environment was updated in place:

```bash
/home/jian/.local/share/mamba/envs/fastwam/bin/python \
  -m pip install --no-deps --force-reinstall nvidia-nccl-cu12==2.27.3
```

Current fixed runtime:

```text
nvidia-nccl-cu12: 2.27.3
ncclGetVersion(): 22703
```

Important: `torch.cuda.nccl.version()` still prints `(2, 26, 2)`. That appears
to be the PyTorch compile-time NCCL version, not the runtime library actually
loaded from `site-packages/nvidia/nccl/lib/libnccl.so.2`.

Use `ncclGetVersion()` or `nccl-tests` to verify the runtime library.

## Verification Commands

Check runtime NCCL:

```bash
/home/jian/.local/share/mamba/envs/fastwam/bin/python - <<'PY'
import ctypes
import importlib.metadata as md
import torch

lib = "/home/jian/.local/share/mamba/envs/fastwam/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2"
nccl = ctypes.CDLL(lib)
version = ctypes.c_int()
ret = nccl.ncclGetVersion(ctypes.byref(version))

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("torch_nccl_compile_tuple", torch.cuda.nccl.version())
print("nvidia-nccl-cu12_dist", md.version("nvidia-nccl-cu12"))
print("ncclGetVersion_ret", ret)
print("ncclGetVersion_runtime", version.value)
PY
```

Expected:

```text
nvidia-nccl-cu12_dist 2.27.3
ncclGetVersion_runtime 22703
```

Run native NCCL default-protocol test:

```bash
unset NCCL_PROTO NCCL_ALGO NCCL_P2P_DISABLE NCCL_NVLS_ENABLE NCCL_IB_DISABLE
export LD_LIBRARY_PATH=/home/jian/.local/share/mamba/envs/fastwam/lib/python3.10/site-packages/nvidia/nccl/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

/data2/jian/tools/nccl-tests/build/all_reduce_perf -b 128K -e 1M -f 2 -g 2
```

Expected:

```text
nccl-library=22703
Out of bounds values : 0 OK
```

Run PyTorch NCCL probes:

```bash
unset NCCL_PROTO NCCL_ALGO NCCL_P2P_DISABLE NCCL_NVLS_ENABLE NCCL_IB_DISABLE
export PATH=/home/jian/.local/share/mamba/envs/fastwam/bin:${PATH}
export LD_LIBRARY_PATH=/home/jian/.local/share/mamba/envs/fastwam/lib/python3.10/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
SCRIPT=/data2/jian/project/FastWAM/scripts/debug_nccl_broadcast.py

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29821 \
  "$SCRIPT" --op broadcast --dtype bf16 --numel 589824

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29822 \
  "$SCRIPT" --op all_reduce --dtype fp32 --numel 196608

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29823 \
  "$SCRIPT" --op all_gather_into_tensor --dtype bf16 --numel 589824
```

Expected:

```text
broadcast OK
all_reduce OK
all_gather_into_tensor OK
```

## Light-WAM Script State

`scripts/train_libero_core.sh` now leaves `NCCL_PROTO` unset by default:

```text
NCCL_PROTO_DEFAULT=unset
```

That means normal NCCL protocol auto-selection is used with the fixed 2.27.3
runtime.

If this environment is reverted to NCCL 2.26.2 or another broken NCCL stack, a
temporary fallback is:

```bash
export NCCL_PROTO_DEFAULT=LL
```

If that is still not enough, use the chunked collective fallback:

```bash
export DISTRIBUTED_CHUNKED_COLLECTIVES_ENABLED=true
export DISTRIBUTED_CHUNKED_COLLECTIVES_MAX_BYTES=65536
```

Those fallbacks are not the preferred final fix.

## Validation Result

After upgrading to NCCL 2.27.3:

```text
native nccl-tests default protocol: pass
PyTorch 2-GPU broadcast bf16: pass
PyTorch 2-GPU all_reduce fp32: pass
PyTorch 8-GPU all_gather bf16: pass
Light-WAM 8-GPU 1-step smoke, NCCL_PROTO unset, chunk disabled: pass
```

Light-WAM smoke result:

```text
step=1/1 loss=0.9417 loss_action=0.3732 loss_video=0.5685
```

## Notes

NVIDIA's CUDA DL 25.06 stack uses NCCL 2.27.3. Community reports around
Blackwell / RTX 50xx / RTX PRO 6000 style systems also point to upgrading NCCL
past 2.26.2 rather than relying on `NCCL_PROTO=LL` permanently.

