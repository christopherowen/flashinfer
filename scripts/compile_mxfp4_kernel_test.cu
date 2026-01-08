// Minimal test to verify SM120 MXFP4 MMA instruction availability
// Compile with: nvcc -arch=sm_121a -o test_mxfp4 compile_mxfp4_kernel_test.cu

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Check if FP4 intrinsics are available
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200

__global__ void test_mma_f8f6f4_kernel(
    float* d,
    const uint32_t* a,  // FP4 packed (8 elements per 32-bit)
    const uint32_t* b,  // FP4 packed
    const float* c
) {
    // This tests if the PTX assembler accepts the f8f6f4 MMA instruction
    // on SM120/SM121 architecture
    
    float d0, d1, d2, d3;
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
    uint32_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    uint32_t b0 = b[0], b1 = b[1];
    
    // SM120 MMA instruction for E2M1 x E2M1 (NVFP4)
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
    
    d[0] = d0;
    d[1] = d1;
    d[2] = d2;
    d[3] = d3;
}

#endif

int main() {
    printf("SM120 MXFP4 MMA instruction test\n");
    printf("If this compiled successfully, the instruction is supported.\n");
    
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    printf("Compiled for SM %d\n", __CUDA_ARCH__);
#else
    printf("Note: __CUDA_ARCH__ is only defined in device code.\n");
    printf("Compile with: nvcc -arch=sm_121a -o test_mxfp4 %s\n", __FILE__);
#endif
    
    return 0;
}


