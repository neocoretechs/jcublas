package com.neocoretechs.cublas;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

import java.util.concurrent.atomic.AtomicReference;

public class Gemm {

	private enum LibraryState {
		NOT_LOADED,
		LOADING,
		LOADED
	}
	private static final AtomicReference<LibraryState> libraryLoaded = new AtomicReference<>(LibraryState.NOT_LOADED);

	static {
		Gemm.loadLibrary(new File(System.getProperty("java.library.path")).list());
	}
	
	/**
	 * Pinned host buffer (optional for faster copies)
	 * @param bytes number of bytes to directly allocate via ByteBuffer
	 * @return The allocated ByteBuffer
	 */
    public static ByteBuffer allocPinned(int bytes) {
        // Fallback: regular direct buffer. If you add cudaHostAlloc JNI, pin it there.
        return ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
    }

	/**
	 * Tries to load the necessary library files from the given list of
	 * directories.
	 *
	 * @param paths a list of strings where each describes a directory of a library.
	 */
	public static void loadLibrary(final String[] paths) {
		if (libraryLoaded.get() == LibraryState.LOADED) {
			return;
		}
		if(libraryLoaded.compareAndSet(LibraryState.NOT_LOADED,LibraryState.LOADING)) {
			//.out.println("Loading from paths list of length:"+paths.size());
			for (final String path : paths) {
				//System.out.println(path);
				if(path.endsWith(".so") || path.endsWith(".dll")) {
					String fname = new File(path).getName();
					fname = fname.substring(0,fname.indexOf("."));
					System.out.println("Trying load for:"+fname);
					System.loadLibrary(fname);
				}
			}
			libraryLoaded.set(LibraryState.LOADED);
		}
		while (libraryLoaded.get() == LibraryState.LOADING) {
			try {
				System.out.println("Waiting for load, retry..");
				Thread.sleep(10);
			} catch(final InterruptedException e) {}
		}
	}

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
	public static native int matrixDotProductF16(long handle, int rows1, int columns1, float[] m1, int rows2, int columns2, float[] m2, float[] mr);
	public static native int matrixDotProductF16Batch(long handle, int rows1, int columns1, ArrayList<float[]> m1, int rows2, int columns2, ArrayList<float[]> m2, ArrayList<float[]> mr, int batchSize);
	public static native int matrixDotProductF16StridedBatch(long handle, int rows1, int columns1, ArrayList<float[]> m1, int rows2, int columns2, ArrayList<float[]> m2, ArrayList<float[]> mr, int batchSize);
	public static native int matrixDotProductF32StridedBatch(long handle, int rows1, int columns1, ArrayList<float[]> m1, int rows2, int columns2, ArrayList<float[]> m2, ArrayList<float[]> mr, int batchSize);
	public static native int matrixDotProductF16StridedBatchFlat(long handle, int rowsA, int colsA, float[] A,int rowsB, int colsB, float[] B, float[] C, int batchSize);
	public static native int matrixDotProductF16StridedBatchFlat2(long handle, int rowsA, int colsA, float[] A,int rowsB, int colsB, float[] B, float[] C, int batchSize);
	public static native int matrixDotProductF16Stream(long handle, int rows1, int columns1, ArrayList<float[]> m1, int rows2, int columns2, ArrayList<float[]> m2, ArrayList<float[]> mr, int batchSize);
	public static native int sdot(long handle, int n, float[] x, int incx, float[] y, int incy, float[] result);
    public static native float sdotSlice(long handle, ByteBuffer qBuf, int qOffsetFloats, ByteBuffer kBuf, int kOffsetFloats, int headSize);
	public static native long cudaMallocBytes(long bytes);
	public static native void cudaFreePtr(long dptr);
	public static native int cudaMemcpyHtoD(long dptr, ByteBuffer src, long bytes);
	public static native int cudaMemcpyDtoH(ByteBuffer dst, long dptr, long bytes);
	public static native int sdotDevice(long handle, int n, long dX, int incx, long dY, int incy, long dResult);
	public static native long cublasHandle();
	public static native int cublasHandleDestroy(long handle);
	public static native long[] cudaMemGetInfo();
}
