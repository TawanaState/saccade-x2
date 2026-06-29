use std::sync::atomic::{AtomicU64, Ordering};
use std::cell::Cell;

/// Global atomic telemetry register bank
pub struct GlobalTelemetry {
    pub total_base_bits: AtomicU64,
    pub total_sparse_bits: AtomicU64,
    pub total_tokens_processed: AtomicU64,
    pub total_elapsed_ns: AtomicU64,
    pub total_param_calls: AtomicU64,
}

impl GlobalTelemetry {
    pub const fn new() -> Self {
        Self {
            total_base_bits: AtomicU64::new(0),
            total_sparse_bits: AtomicU64::new(0),
            total_tokens_processed: AtomicU64::new(0),
            total_elapsed_ns: AtomicU64::new(0),
            total_param_calls: AtomicU64::new(0),
        }
    }

    pub fn reset(&self) {
        self.total_base_bits.store(0, Ordering::Relaxed);
        self.total_sparse_bits.store(0, Ordering::Relaxed);
        self.total_tokens_processed.store(0, Ordering::Relaxed);
        self.total_elapsed_ns.store(0, Ordering::Relaxed);
        self.total_param_calls.store(0, Ordering::Relaxed);
    }
}

pub static TELEMETRY: GlobalTelemetry = GlobalTelemetry::new();

thread_local! {
    static LOCAL_BASE_BITS: Cell<u64> = Cell::new(0);
    static LOCAL_SPARSE_BITS: Cell<u64> = Cell::new(0);
    static LOCAL_TOKENS: Cell<u64> = Cell::new(0);
    static LOCAL_PARAM_CALLS: Cell<u64> = Cell::new(0);
}

/// Log a routing decision. Updates thread-local counters and flushes to global atomics periodically.
#[inline(always)]
pub fn log_routing_decision(is_sparse: bool, in_features: usize, out_features: usize, csc_nnz: usize) {
    let base_bits = (in_features * out_features * 4) as u64;
    let sparse_bits = if is_sparse { (csc_nnz * 8) as u64 } else { 0 };
    let params = (in_features * out_features) as u64;

    LOCAL_BASE_BITS.with(|b| {
        LOCAL_SPARSE_BITS.with(|s| {
            LOCAL_TOKENS.with(|t| {
                LOCAL_PARAM_CALLS.with(|p| {
                    let current_b = b.get() + base_bits;
                    let current_s = s.get() + sparse_bits;
                    let current_t = t.get() + 1;
                    let current_p = p.get() + params;

                    if current_t >= 64 {
                        TELEMETRY.total_base_bits.fetch_add(current_b, Ordering::Relaxed);
                        TELEMETRY.total_sparse_bits.fetch_add(current_s, Ordering::Relaxed);
                        TELEMETRY.total_tokens_processed.fetch_add(current_t, Ordering::Relaxed);
                        TELEMETRY.total_param_calls.fetch_add(current_p, Ordering::Relaxed);
                        b.set(0);
                        s.set(0);
                        t.set(0);
                        p.set(0);
                    } else {
                        b.set(current_b);
                        s.set(current_s);
                        t.set(current_t);
                        p.set(current_p);
                    }
                });
            });
        });
    });
}

/// Log a bypass execution decision.
#[inline(always)]
pub fn log_bypass_decision(in_features: usize, out_features: usize) {
    let fp16_bits = (in_features * out_features * 16) as u64;
    let params = (in_features * out_features) as u64;

    LOCAL_BASE_BITS.with(|b| {
        LOCAL_TOKENS.with(|t| {
            LOCAL_PARAM_CALLS.with(|p| {
                let current_b = b.get() + fp16_bits;
                let current_t = t.get() + 1;
                let current_p = p.get() + params;

                if current_t >= 64 {
                    TELEMETRY.total_base_bits.fetch_add(current_b, Ordering::Relaxed);
                    TELEMETRY.total_tokens_processed.fetch_add(current_t, Ordering::Relaxed);
                    TELEMETRY.total_param_calls.fetch_add(current_p, Ordering::Relaxed);
                    b.set(0);
                    t.set(0);
                    p.set(0);
                } else {
                    b.set(current_b);
                    t.set(current_t);
                    p.set(current_p);
                }
            });
        });
    });
}

/// Flush remaining thread-local metrics to global telemetry.
pub fn flush_telemetry() {
    LOCAL_BASE_BITS.with(|b| {
        LOCAL_SPARSE_BITS.with(|s| {
            LOCAL_TOKENS.with(|t| {
                LOCAL_PARAM_CALLS.with(|p| {
                    let current_b = b.get();
                    let current_s = s.get();
                    let current_t = t.get();
                    let current_p = p.get();
                    if current_t > 0 {
                        TELEMETRY.total_base_bits.fetch_add(current_b, Ordering::Relaxed);
                        TELEMETRY.total_sparse_bits.fetch_add(current_s, Ordering::Relaxed);
                        TELEMETRY.total_tokens_processed.fetch_add(current_t, Ordering::Relaxed);
                        TELEMETRY.total_param_calls.fetch_add(current_p, Ordering::Relaxed);
                        b.set(0);
                        s.set(0);
                        t.set(0);
                        p.set(0);
                    }
                });
            });
        });
    });
}
