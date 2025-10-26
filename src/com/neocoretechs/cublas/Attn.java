package com.neocoretechs.cublas;

public final class Attn {
	private final long handle;
	public Attn(int maxB, int maxH, int maxTq, int maxTk, int d) {
		this.handle = init(maxB, maxH, maxTq, maxTk, d);
		if (handle == 0) throw new RuntimeException("Failed to init AttnCtx");
	}
	public long getHandle() { return handle; }
	public long dq() { return getDQ(handle); }
	public long dk() { return getDK(handle); }
	public long dv() { return getDV(handle); }
	public long ds() { return getDS(handle); }
	public long dO() { return getDO(handle); }

	public void upload(float[] buf, long devicePtr, long offset, int count) {
		int rc = Attn.uploadSlice(handle, buf, devicePtr, offset, count);
		if (rc != 0) throw new RuntimeException("Upload failed, code " + rc);
	}

	public void close() {
		destroy(handle);
	}
	public static native long init(int maxB, int maxH, int maxTq, int maxTk, int d);
	public static native int uploadSlice(long ctxHandle, float[] hostBuf, long devicePtr, long offset, int count);
	public static native int downloadSlice(long ctxHandle, float[] hostBuf, long devicePtr, long offset, int count);
	public static native long getDQ(long ctxHandle);
	public static native long getDK(long ctxHandle);
	public static native long getDV(long ctxHandle);
	public static native long getDS(long ctxHandle);
	public static native long getDO(long ctxHandle);
	public static native void destroy(long ctxHandle);
}
