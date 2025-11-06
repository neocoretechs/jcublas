package com.neocoretechs.cublas;

import java.util.Arrays;

public final class AttentionRunner {
	public static boolean DEBUG = false;
    public static final class Config {
        public final int heads;
        public final int d;       // head size
        public final int Tq;      // query length
        public final int Tk;      // key/value length
        public Config(int heads, int d, int Tq, int Tk) {
            this.heads = heads; this.d = d; this.Tq = Tq; this.Tk = Tk;
        }
    }

    // Core execution: returns O packed per head as float[]
    public static float[] run(long cublasHandle, Attn ctx, Config cfg, float[] Q, float[] K, float[] V) {
        final int H = cfg.heads; // batch size across heads
        final int d = cfg.d;
        final int Tq = cfg.Tq;
        final int Tk = cfg.Tk;

        // Buffers: S (scores) and O (output), packed per head
        float[] S = new float[H * Tq * Tk];
        float[] O = new float[H * Tq * d];

        // Upload Q/K/V to device
        ctx.upload(Q, ctx.dq(), 0L, Q.length);
        ctx.upload(K, ctx.dk(), 0L, K.length);
        ctx.upload(V, ctx.dv(), 0L, V.length);

        // Wrap packed buffers in ArrayLists expected by JNI strided batched
        // GEMM 1: S = Q * K^T, strided batched over heads
        //ArrayList<float[]> qList = new ArrayList<>(List.of(Q));
        //ArrayList<float[]> kList = new ArrayList<>(List.of(K)); 
        //ArrayList<float[]> sList = new ArrayList<>(List.of(S));
        //int rc1 = Gemm.matrixDotProductF16StridedBatch(Llama3., Tq, d, qList, Tk, d, kList, sList, H);
        // GEMM 1: S = (1/sqrt(d)) * Q * K^T   (FP16 inputs, FP32 accumulate)
        if(DEBUG)
        	System.out.printf("d=%d Tk=%d Tq=%d Q.length=%d K.length=%d S.length=%d batch=%d\n",
        	       d, Tk, Tq, Q.length, K.length, S.length, H);
        int rc1 = Gemm.matrixDotProductF16StridedBatchFlat(cublasHandle, Tq, d, Q, Tk, d, K, S, H);
        if (rc1 != 0)
        	throw new RuntimeException("Scores GEMM failed: " + rc1);
        // Softmax row-wise on S (CPU), per head and per query row
        softmaxRowsInPlace(S, H, Tq, Tk, (float)(1.0 / Math.sqrt(d)));

        // Re-upload S after softmax
        ctx.upload(S, ctx.ds(), 0, S.length);

        // GEMM 2: O = S * V (FP16 inputs, FP32 accumulate)
        //ArrayList<float[]> vList = new ArrayList<>(List.of(V));
        //ArrayList<float[]> oList = new ArrayList<>(List.of(O));
/*
        int rc2 = Gemm.matrixDotProductF16StridedBatchFlat2(cublasHandle, Tq, Tk, S, Tk, d, V, O, H);
        if (rc2 != 0) 
        	throw new RuntimeException("Output GEMM failed: " + rc2);

        // Optionally: download O from device slice if your native keeps device O
        // Here we assume JNI copied into oList; if not, uncomment:
        //Attn.downloadSlice(ctx.handle, O, ctx.dO(), 0, O.length);
        if(DEBUG)
        	System.out.println("Olist:"+Arrays.toString(O));
 */
        return O;
    }

    // CPU softmax over rows: for each head and query row, normalize S[h][t][*]
    public static void softmaxRowsInPlace(float[] S, int H, int Tq, int Tk, float scale) {
        final int strideS = Tq * Tk;
        for (int h = 0; h < H; h++) {
            int base = h * strideS;
            for (int t = 0; t < Tq; t++) {
                int row = base + t * Tk;
                // scale and max
                float maxv = Float.NEGATIVE_INFINITY;
                for (int u = 0; u < Tk; u++) {
                    float v = S[row + u] * scale;
                    S[row + u] = v;
                    if (v > maxv) maxv = v;
                }
                // exp and sum
                float sum = 0f;
                for (int u = 0; u < Tk; u++) {
                    float e = (float) Math.exp(S[row + u] - maxv);
                    S[row + u] = e;
                    sum += e;
                }
                // normalize
                float inv = 1.0f / sum;
                for (int u = 0; u < Tk; u++) {
                    S[row + u] *= inv;
                }
            }
        }
    }
}
