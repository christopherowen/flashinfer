/***************************************************************************************************
 * SM120 Gated GEMM Kernel - PLACEHOLDER
 * 
 * NOTE: This file is kept for reference but is NOT USED in production.
 * 
 * The gated FC1 path uses the STANDARD GemmUniversal kernel with a gated mainloop.
 * The gated mainloop's single-accumulator mma() applies SiLU inline, so no custom
 * kernel is needed.
 * 
 * Key insight from TRT-LLM:
 *   - SwiGLU is applied INLINE in the mainloop's mma(), not in a separate epilogue
 *   - The mainloop's single-accumulator mma() calls dual-accumulator mma() internally
 *   - Standard epilogue stores the already-fused result
 *
 **************************************************************************************************/

#pragma once

// This header is intentionally minimal.
// The production path uses GemmUniversal with the gated mainloop's single-accumulator mma().

