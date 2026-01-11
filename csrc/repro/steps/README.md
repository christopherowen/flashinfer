This folder contains **incremental “step” repros**. Each file is a copy of the previous step with a
single focused change, so we can bisect what flips SM12x behavior from “works” to “fails”.

## Steps

- `step00_fp8_groupwise_working.cu`
  - Baseline: calls FlashInfer’s known-working SM12x grouped GEMM:
    `flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh`

- `step01_mxf8_mxf4_grouped_init_fail.cu`
  - Baseline failing case: CUTLASS `initialize()` for MXF8×MXF4 grouped GEMM returns
    `Error Internal` on SM121 (even for tiny shapes).

- `step02_mxf8_mxf8_grouped_init.cu`
  - Like step01 but uses **MXF8×MXF8** (no FP4 weights) to isolate whether FP4 is the trigger.

- `step03_mxf8_mxf4_grouped_init_blockwise_schedule.cu`
  - Like step01 but swaps the **mainloop kernel schedule** to `KernelScheduleSm120Blockwise`
    to test schedule sensitivity.

## Running in the container

These files live in the `ai/mxfp4` repo; the `vllm-dev` container’s `/workspace/flashinfer` is **not**
a bind mount in this environment, so you need to `docker cp` the chosen step into the container and
compile it there.

