# JCublas

JCublas is a lightweight JNI bridge exposing selected [cuBLAS](https://developer.nvidia.com/cublas) functionality to the JVM.  
It provides native methods for GPU-accelerated linear algebra and utility calls (e.g. `cudaMemGetInfo`) so Java applications can directly leverage CUDA.

---

## ✨ Features

- JNI bindings for core cuBLAS routines (e.g. `cublasSgemm`, `cublasDgemm`)
- Utility bindings for CUDA runtime calls (e.g. `cudaMemGetInfo`)
- Automatic native library loading via `System.loadLibrary`
- Simple Java wrapper classes for safe, idiomatic usage

---

## 📦 Repository Structure
jcublas/ 
├─ src/com/neocoretechs/cublas/    
├───────────────────────────── Gemm.java        # Java wrapper for cuBLAS calls 
│   
├─ README.md
│
└─ build.xml
---

## 🚀 Getting Started

### Prerequisites

- NVIDIA GPU with CUDA Toolkit installed
- Java 25
- CMake / nvcc for building the native library

### Build Native Library

```bash
cd src/main/native
mkdir build && cd build
cmake ..This produces a shared library ( on Linux,  on Windows,  on macOS).
Java Usage

make

package com.neocoretechs.brain;

public class Example {
    static {
        System.loadLibrary("jcublas"); // loads libjcublas
    }

    public static void main(String[] args) {
        // Query GPU memory
        long[] memInfo = CudaUtils.cudaMemGetInfo();
        System.out.println("Free: " + memInfo[0] + " bytes, Total: " + memInfo[1] + " bytes");

        // Run a simple SGEMM
        float[] A = ...; // m×k
        float[] B = ...; // k×n
        float[] C = new float[m*n];
        JCublas.sgemm(m, n, k, A, B, C);

        System.out.println("Result[0] = " + C[0]);
    }
}
🧩 Native Methods
CudaUtils
public class CudaUtils {
    public static native long[] cudaMemGetInfo();
}


JCublas
public class JCublas {
    public static native int sgemm(int m, int n, int k,
                                   float[] A, float[] B, float[] C);
    // Add dgemm, batched variants, etc.
}

```

🛡️ Notes
- Always check return codes from native methods; non‑zero indicates a CUDA/cuBLAS error.
- Pre‑allocate and reuse buffers to avoid memory leaks.
- Use nvidia-smi or CudaUtils.cudaMemGetInfo() to monitor VRAM usage.

📜 License
MIT 

