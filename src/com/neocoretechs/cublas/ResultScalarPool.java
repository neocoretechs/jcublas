package com.neocoretechs.cublas;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;

public final class ResultScalarPool {
	private static final ArrayDeque<Scalar> pool = new ArrayDeque<>();
	public static Scalar acquire() {
	     Scalar s = pool.pollFirst();
	     return (s != null) ? s : new Scalar();
	}
	public static void release(Scalar s) {
	     pool.offerFirst(s);
	}
	public static final class Scalar implements AutoCloseable {
		public final long dPtr;
		public final ByteBuffer hostBuf;
		DeviceMemoryReclaim cleaner;
		Scalar() {
			this.hostBuf = ByteBuffer.allocateDirect(Float.BYTES).order(ByteOrder.nativeOrder());
			this.dPtr = Gemm.cudaMallocBytes(Float.BYTES);
			if(!DeviceMemoryLedger.tryReserve(this.hostBuf.capacity()))
				throw new RuntimeException("Scalar pool allocate failed");
			this.cleaner = new DeviceMemoryReclaim(this.dPtr, this.hostBuf.capacity());	
		}
		public void download() { 
			Gemm.cudaMemcpyDtoH(hostBuf, dPtr, Float.BYTES); 
		}
		public float get() { 
			return hostBuf.getFloat(0); 
		}
		@Override public void close() { 
			//Gemm.cudaFreePtr(dPtr);
			cleaner.close();
		}
	}
}
