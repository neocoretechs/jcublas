package com.neocoretechs.cublas;

public final class DeviceMemoryLedger {
    private static long baselineFree;
    private static long total;
    private static long allocated;
    private static final int MIN_INTERVAL = 8;
    private static int releaseCount = 0;
    private static int refreshInterval = MIN_INTERVAL;
    private static long lastRefreshNanos = System.nanoTime();
    private static final int MAX_INTERVAL = 1024;
    private static final long MAX_REFRESH_GAP_NANOS = 5_000_000_000L; // 5s
    private static final int REFRESH_INTERVAL = 32; // every 32 frees
  
    static {
        long[] info = Gemm.cudaMemGetInfo();
        baselineFree = info[0];
        total        = info[1];
        allocated    = 0;
        lastRefreshNanos = System.nanoTime();
    }

    public static synchronized boolean tryReserve(long bytes) {
    	 // Safety margin: 10% of requested size, clamped between 4MB and 256MB
        long margin = Math.max(4 << 20, Math.min(256 << 20, bytes / 10));
        if(allocated + bytes + margin > baselineFree) {
            return false;
        }
        allocated += bytes;
        return true;
    }

    public static synchronized void release(long bytes) {
        allocated -= bytes;
        if (allocated < 0) allocated = 0;
        releaseCount++;
        long now = System.nanoTime();
        boolean timeExpired = (now - lastRefreshNanos) > MAX_REFRESH_GAP_NANOS;
        boolean countExpired = releaseCount >= refreshInterval;
        if (timeExpired || countExpired) {
            refresh();
            // logarithmic wind‑up
            refreshInterval = Math.min(
                MAX_INTERVAL,
                MIN_INTERVAL * (1 + (int)(Math.log(releaseCount + 1) / Math.log(2)))
            );
            releaseCount = 0;
            lastRefreshNanos = now;
        }
    }

    private static void refresh() {
        long[] info = Gemm.cudaMemGetInfo();
        baselineFree = info[0];
        total        = info[1];
        allocated    = 0;
    }

    public static synchronized void onAllocationFailure() {
        refreshInterval = MIN_INTERVAL;
        releaseCount = 0;
        lastRefreshNanos = System.nanoTime();
    }
}
