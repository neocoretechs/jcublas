# JCublas

JCublas is a lightweight JNI bridge exposing selected [cuBLAS](https://developer.nvidia.com/cublas) functionality to the JVM.  
It provides native methods for GPU-accelerated linear algebra and utility calls (e.g. `cudaMemGetInfo`) so Java applications can directly leverage CUDA.
It also provides CPU-only methods for comparison or when GPU is not available.
---

## ✨ Features

- JNI bindings for core cuBLAS routines (e.g. `cublasSgemm`, `cublasDgemm`)
- Utility bindings for CUDA runtime calls (e.g. `cudaMemGetInfo`)
- Automatic native library loading via `System.loadLibrary`
- Simple Java wrapper classes for safe, idiomatic usage

---

## 📦 Repository Structure
jcublas/src/com/neocoretechs/cublas/Gemm.java # Java wrapper for cuBLAS calls  
README.md
build.xml
---

## 🚀 Getting Started

### Prerequisites

- NVIDIA GPU with CUDA Toolkit installed
- Java 25
- CMake / nvcc for building the native library
- MatrixDotProduct_vs repository
### Build Native Library

```bash
Visual Studio project in MatrixDotProduct_vs
make

package com.neocoretechs.cublas;

public class Example {

    public static void main(String[] args) {
		long context = Gemm.cublasHandle();
        // Query GPU memory
        long[] memInfo = Gemm.cudaMemGetInfo();
        System.out.println("Free: " + memInfo[0] + " bytes, Total: " + memInfo[1] + " bytes");
        // Run a simple SGEMM batched operation
        ArrayList<float[]> A = ...; // m×k
        ArrayList<float[]> B = ...; // k×n
        ArrayList<float[]> C = new ArrayList<float[]>(m*n);
        if(Gemm.matrixDotProductFBatch(context, m, k, A, k, n, B, C, c.size()) != 0)
			throw new RuntimeExcpetion("Gemm failed");
        System.out.println("Result[0] = " + C[0]);
		cublasHandleDestroy(context);
    }
}
🧩 Native Methods

	 public static native int matrixDotProductD(long handle, int rows1, int columns1, double[] m1, int rows2, int columns2, double[] m2, double[] mr);
	 public static native int matrixDotProductDBatch(long handle, int rows1, int columns1, ArrayList<double[]> m1, int rows2, int columns2, ArrayList<double[]> m2, ArrayList<double[]> mr, int batchSize);
	 public static native int matrixDotProductDStream(long handle, int rows1, int columns1, ArrayList<double[]> m1, int rows2, int columns2, ArrayList<double[]> m2, ArrayList<double[]> mr, int batchSize);
	 public static native int matrixDotProductDCPU(int rows1, int columns1, double[] m1, int rows2, int columns2, double[] m2, double[] mr);
	 public static native int matrixDotProductDCPUBatch(int rows1, int columns1, ArrayList<double[]> m1, int rows2, int columns2, ArrayList<double[]> m2, ArrayList<double[]> mr, int batchSize);
	 public static native int matrixDotProductF(long handle, int rows1, int columns1, float[] m1, int rows2, int columns2, float[] m2, float[] mr);
	 public static native int matrixDotProductFBatch(long handle, int rows1, int columns1, ArrayList<float[]> m1, int rows2, int columns2, ArrayList<float[]> m2, ArrayList<float[]> mr, int batchSize);
	 public static native int matrixDotProductFStream(long handle, int rows1, int columns1, ArrayList<float[]> m1, int rows2, int columns2, ArrayList<float[]> m2, ArrayList<float[]> mr, int batchSize);
	 public static native int matrixDotProductFCPU(int rows1, int columns1, float[] m1, int rows2, int columns2, float[] m2, float[] mr);
	 public static native int matrixDotProductFCPUBatch(int rows1, int columns1, ArrayList<float[]> m1, int rows2, int columns2, ArrayList<float[]> m2, ArrayList<float[]> mr, int batchSize);
	 public static native long cublasHandle();
	 public static native int cublasHandleDestroy(long handle);
	 public static native long cudaMemGetInfo();
```

🛡️ Notes
- Always check return codes from native methods; non‑zero indicates a CUDA/cuBLAS error.
- Pre‑allocate and reuse buffers to avoid memory leaks.
- Use nvidia-smi or Gemm.cudaMemGetInfo() to monitor VRAM usage.

📜 License
MIT 

