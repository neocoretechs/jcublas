package com.neocoretechs.cublas;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import com.llama4j.FloatTensor;
import com.llama4j.BufferPool;
/**
 * The ctx handle here refers to the JNI Attn handle, not to be confused with cublasHandle
 */
public final class Attn {
	private boolean DEBUG=false;
	private final long ctx;
	final int Tq, Tk, d, H;
	final FloatBuffer Q_all, K_all, V_all, O_all;
	final FloatBuffer msQKT, msSM, msAV;

	public Attn(long cublasHandle, int Tq, int Tk, int d, int H) {
		if(DEBUG)
			System.out.printf("Attn c'tor: Tq:%d, Tk:%d, d:%d H:%d%n",Tq,Tk,d,H);
		this.ctx = initContext(cublasHandle, Tq, Tk, d, H);
		if (ctx == 0) throw new RuntimeException("Failed to init AttnCtx");
		this.Tq = Tq; this.Tk = Tk; this.d = d; this.H = H;
		int ld = H * d;
		this.Q_all = fbuf(Tq * ld);
		this.K_all = fbuf(Tk * ld);
		this.V_all = fbuf(Tk * ld);
		this.O_all = fbuf(Tq * ld);
		this.msQKT = fbuf(1);
		this.msSM  = fbuf(1);
		this.msAV  = fbuf(1);
	}
	public long getContext() { return ctx; }

	public void packHeads(FloatTensor[] Q_heads, FloatTensor[] K_heads, FloatTensor[] V_heads) {
		packAllHeads(Q_heads, Tq, d, H, Q_all);
		packAllHeads(K_heads, Tk, d, H, K_all);
		packAllHeads(V_heads, Tk, d, H, V_all);
	}

	public void unpackHeads(FloatTensor[] O_heads) {
		unpackAllHeads(O_all, Tq, d, H, O_heads);
	}
    // pack/unpack helpers
    static void packAllHeads(FloatTensor[] heads, int rows, int d, int H, FloatBuffer dst) {
        dst.clear();
        float[] scratch = new float[d];
        for (int r = 0; r < rows; r++) {
            int base = r * (H * d);
            for (int h = 0; h < H; h++) {
                heads[h].exportSlice(scratch, 0, r * d, d);
                dst.position(base + h * d);
                dst.put(scratch, 0, d);
            }
        }
        dst.flip();
    }
    static void unpackAllHeads(FloatBuffer src, int rows, int d, int H, FloatTensor[] outHeads) {
        src.rewind();
        for (int r = 0; r < rows; r++) {
            int base = r * (H * d);
            for (int h = 0; h < H; h++) {
                for (int c = 0; c < d; c++) {
                    outHeads[h].setFloat(r * d + c, src.get(base + h * d + c));
                }
            }
        }
    }

	public int attention() {
		if(DEBUG)
        // ... fill Q/K/V from your FloatTensor slices ...
		System.out.printf("Just before JNI attentionFp32:d=%d Tq=%d Tk=%d FloatBuffer Q_all (position,limit,remain):(%d,%d,%d) "
				+ "K_all (position,limit,remain):(%d,%d,%d) V_all (position,limit,remain):(%d,%d,%d)%n",d,Tq,Tk,
				Q_all.position(),Q_all.limit(),Q_all.remaining(),
			K_all.position(),K_all.limit(),K_all.remaining(),V_all.position(),V_all.limit(),V_all.remaining() );
	     O_all.clear();
	     return attentionFp32(ctx,
	            Q_all, H * d, K_all, H * d, V_all, H * d,
	            O_all, H * d,
	            msQKT, msSM, msAV);
	}

	public void close() {
		freeContext(ctx);
	}
	
	// Helper: allocate direct FloatBuffers of given length
	public static FloatBuffer fbuf(int len) {
		return ByteBuffer.allocateDirect(len * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
	}

	private static int rows(FloatTensor t, int d) {
        int size = t.size();           // total elements in the flat backing array
        if (size % d != 0) {
            throw new IllegalArgumentException("Tensor size not divisible by d: size=" + size + " d=" + d);
        }
        return size / d;               // infer number of rows
	}
    
    public static void packRowMajor(FloatTensor Qh, int Tq, int d, FloatBuffer Qbuf) {
        Qbuf.clear();
        for (int r = 0; r < Tq; r++) {
            float[] tmp = new float[d];
            Qh.exportSlice(tmp, 0, r * d, d);
            Qbuf.put(tmp, 0, d);
        }
        Qbuf.flip();
    }
    public static void unpackRowMajor(FloatBuffer Obuf, int Tq, int d, FloatTensor Oh) {
        Obuf.rewind();
        for (int r = 0; r < Tq; r++) {
            for (int c = 0; c < d; c++) {
                Oh.setFloat(r * d + c, Obuf.get());
            }
        }
    }
    
    public static FloatTensor[] sliceHeads(FloatTensor[] heads, int h0, int hcount) {
        FloatTensor[] out = new FloatTensor[hcount];
        System.arraycopy(heads, h0, out, 0, hcount);
        return out;
    }
    static float[][] sliceRowsFlat(FloatTensor[] heads, int h0, int hCount, int row0, int rowCount, int d, BufferPool pool) {
    	float[][] out = new float[hCount][];
    	for (int h = 0; h < hCount; h++) {
    		FloatTensor src = heads[h0 + h];
    		//out[h] = src.exportSlicePooled(pool, row0, rowCount, d);
    	}
    	return out;
    }

    // artifacts from previous tests
	public void upload(float[] buf, long devicePtr, long offset, int count) {
		int rc = Attn.uploadSlice(ctx, buf, devicePtr, offset, count);
		if (rc != 0) throw new RuntimeException("Upload failed, code " + rc);
	}
	public long dq() { return getDQ(ctx); }
	public long dk() { return getDK(ctx); }
	public long dv() { return getDV(ctx); }
	public long ds() { return getDS(ctx); }
	public long dO() { return getDO(ctx); }
	
	public static native long init(long ctxHandle, int maxB, int maxH, int maxTq, int maxTk, int d);
	public static native int uploadSlice(long ctxHandle, float[] hostBuf, long devicePtr, long offset, int count);
	public static native int downloadSlice(long ctxHandle, float[] hostBuf, long devicePtr, long offset, int count);
	public static native long getDQ(long ctxHandle);
	public static native long getDK(long ctxHandle);
	public static native long getDV(long ctxHandle);
	public static native long getDS(long ctxHandle);
	public static native long getDO(long ctxHandle);
	public static native void destroy(long ctxHandle);
	public static native float[] softMax(float[] in, int rows, int cols);
    // Native lifecycle
    public static native long initContext(long cublasHandle, int Tq, int Tk, int d, int H);
    public static native void freeContext(long ctx);
    //Function to convert GGUF quantized types to float using CUDA format: 0 - Q4_0, 1 - Q8_0, 2 - F16, 3 - B16
    public static native long convertBufferToFloat(ByteBuffer byteBuffer, int blockSize, int typeSize, int headerBytes, int format);
    // Execute full attention on existing direct buffers
    public static native int attentionFp32(long ctx,
            FloatBuffer Q, int ldQ, FloatBuffer K, int ldK, FloatBuffer V, int ldV, 
            FloatBuffer O, int ldO,
            FloatBuffer outMsQkt, FloatBuffer outMsSoftmax, FloatBuffer outMsAv);
    
}
