package com.neocoretechs.cublas;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public final class DeviceBuffer implements AutoCloseable {
	private static boolean DEBUG = false;
	public long devicePtr;
	private final ByteBuffer memoryBuffer;
	private final int blockSize, typeSize, headerBytes, format;
	private boolean isUploaded = false;
	DeviceMemoryReclaim cleaner;
	public enum GGUFQ {
		Q4_0,
		Q8_0,
		F16,
		BF16,
		F32;
	}
	
	public DeviceBuffer(ByteBuffer memoryBuffer, int blockSize, int typeSize, int headerBytes, int format) {
		this.memoryBuffer = memoryBuffer;
		this.blockSize = blockSize;
		this.typeSize = typeSize;
		this.headerBytes = headerBytes;
		this.format = format;
		this.cleaner = new DeviceMemoryReclaim(this);	
	}
	
	public int getBufferCapacity() {
		return memoryBuffer.capacity();
	}


	public void download() {

	}

	@Override public void close() {
		Gemm.cudaFreePtr(devicePtr);
		isUploaded = false;
	}
	
	interface Timer extends AutoCloseable {
	    @Override
	    void close(); // no Exception

	    static Timer log(String label) {
	        return log(label, TimeUnit.MILLISECONDS);
	    }

	    static Timer log(String label, TimeUnit timeUnit) {
	        return new Timer() {
	            final long startNanos = System.nanoTime();

	            @Override
	            public void close() {
	                long elapsedNanos = System.nanoTime() - startNanos;
	                System.err.println(label + ": "
	                        + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
	                        + timeUnit.toChronoUnit().name().toLowerCase());
	            }
	        };
	    }
	}
}
