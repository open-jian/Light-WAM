# Blackwell NCCL Diagnosis

## Host

- GPUs: 8x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
- Topology: GPUs 0-3 share one NUMA node, GPUs 4-7 share another; no NVLink.
- Driver: 570.172.08.
- PyTorch: 2.7.1+cu128.
- Original NCCL from the PyTorch wheel: 2.26.2+cuda12.2.
- Fixed runtime NCCL: nvidia-nccl-cu12 2.27.3, whose `libnccl.so.2`
  reports `ncclGetVersion=22703`.

## Symptom

Large NCCL collectives can fail with:

```text
enqueue.cc:1556 NCCL WARN Cuda failure 700 'an illegal memory access was encountered'
ncclUnhandledCudaError
```

This is reproducible without Light-WAM, DeepSpeed, the dataset, or model code.

## Local Repro

Naked PyTorch ProcessGroupNCCL:

- 2-GPU `broadcast`, bf16, 131072 elements: pass.
- 2-GPU `broadcast`, bf16, 589824 elements: fail.
- 2-GPU `all_reduce`, fp32, 196608 elements: fail.
- The same failing cases pass when `NCCL_PROTO=LL`.

Native `nccl-tests`, built against the NCCL package used by the PyTorch env:

- Small sizes up to 64KB pass.
- 128KB and above fail with the default protocol selection.
- `-c 0` still fails, so this is not only the validation kernel.
- `NCCL_PROTO=LL` passes for broadcast and all-reduce.
- `NCCL_PROTO=Simple` fails.
- `NCCL_PROTO=LL128` fails.
- `NCCL_ALGO=Ring` passed for the 128KB all-reduce probe.
- `NCCL_ALGO=Tree` failed for the same probe.

## Version Fix

The clean fix is to upgrade the runtime NCCL package used by PyTorch:

```bash
python -m pip install --no-deps --force-reinstall nvidia-nccl-cu12==2.27.3
```

PyTorch 2.7.1+cu128 still declares `nvidia-nccl-cu12==2.26.2` in package
metadata, so `pip check` reports a dependency mismatch. Runtime loading is what
matters here: verify with `ctypes` calling `ncclGetVersion()` on
`site-packages/nvidia/nccl/lib/libnccl.so.2`.

After the NCCL 2.27.3 upgrade, the previously failing cases pass without
setting `NCCL_PROTO`.

`scripts/train_libero_core.sh` now leaves `NCCL_PROTO` unset by default. Chunked
collectives and `NCCL_PROTO_DEFAULT=LL` are still available as explicit
fallbacks, but are disabled by default.

DeepSpeed bucket sizes are kept at the repo's normal large defaults. An 8-GPU
Light-WAM smoke test passed with:

- normal ZeRO-1 bucket sizes,
- `DISTRIBUTED_CHUNKED_COLLECTIVES_ENABLED=false`,
- `NCCL_PROTO` unset,
- `MAX_STEPS=1`.

Smoke result:

```text
step=1/1 loss=0.9417 loss_action=0.3732 loss_video=0.5685
```

## Interpretation

The failure is best explained as a NCCL 2.26.2 protocol/algorithm path bug on
this Blackwell setup, not as a Light-WAM training bug. Upgrading the runtime
NCCL library to 2.27.3 avoids the broken path while preserving NCCL's normal
protocol selection.

## Fallback

If the environment is reverted to NCCL 2.26.2 or another broken stack, first try
forcing LL:

```bash
export NCCL_PROTO_DEFAULT=LL
```

If that is not enough, enable the chunk fallback:

```bash
export DISTRIBUTED_CHUNKED_COLLECTIVES_ENABLED=true
export DISTRIBUTED_CHUNKED_COLLECTIVES_MAX_BYTES=65536
```

That path slices large tensors into smaller collectives, but should be treated as a fallback because it adds launch overhead.
