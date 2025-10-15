package com.neocoretechs.cublas;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

public class Gemm {

	 private enum LibraryState {
		    NOT_LOADED,
		    LOADING,
		    LOADED
	 }
	 private static final AtomicReference<LibraryState> libraryLoaded = new AtomicReference<>(LibraryState.NOT_LOADED);
	 
	 static {
		    Gemm.loadLibrary(Arrays.asList(new File("/usr/lib/jni").list()));
	 }

	 /**
	  * Tries to load the necessary library files from the given list of
	  * directories.
	  *
	  * @param paths a list of strings where each describes a directory of a library.
	  */
	 public static void loadLibrary(final List<String> paths) {
		    if (libraryLoaded.get() == LibraryState.LOADED) {
		      return;
		    }
		    if (libraryLoaded.compareAndSet(LibraryState.NOT_LOADED,LibraryState.LOADING)) {
		      boolean success = false;
		      UnsatisfiedLinkError err = null;
		      //.out.println("Loading from paths list of length:"+paths.size());
		      for (final String path : paths) {
		        try {
		          //System.out.println(path);
		          if(path.endsWith(".so") || path.endsWith(".dll")) {
		        	  String fname = new File(path).getName();
		        	  fname = fname.substring(0,fname.indexOf("."));
		        	  System.out.println("Trying load for:"+fname);
		        	  System.loadLibrary(fname);
		        	  success = true;
		          }
		        } catch (final UnsatisfiedLinkError e) {
		          err = e;
		          success = false;
		          break;
		        }
		      }
		      if (!success) {
		        libraryLoaded.set(LibraryState.NOT_LOADED);
		        throw err;
		      }
		      libraryLoaded.set(LibraryState.LOADED);
		      return;
		    }

		    while (libraryLoaded.get() == LibraryState.LOADING) {
		      try {
		    	System.out.println("Waiting for load, retry..");
		        Thread.sleep(10);
		      } catch(final InterruptedException e) {
		        //ignore
		      }
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
	 public static native int matrixDotProductF16Stream(long handle, int rows1, int columns1, ArrayList<float[]> m1, int rows2, int columns2, ArrayList<float[]> m2, ArrayList<float[]> mr, int batchSize);
	 public static native long cublasHandle();
	 public static native int cublasHandleDestroy(long handle);
	 public static native long[] cudaMemGetInfo();
}
