//! Single-token (b=1, GEMV) kernel audit on DRAM-resident weights.
//!
//! Every format is replicated K times (pool >= ~1 GB of FP32-equivalent) and the calls round-robin
//! through the pool, so no matrix is cache-resident when it is used.
//!
//!   cargo run --release --bin audit_bench -- bw
//!   cargo run --release --bin audit_bench -- gemv [out in]
use candle_core::{DType, Device, Module, Tensor};
use rayon::prelude::*;
use saccade_core::config::SparseDeltaMatrix;
use saccade_core::{compress_tensor_to_saccade, compute_percentile_threshold, variance_heuristic, SaccadeConfig, SaccadeLinearOp};
use std::hint::black_box;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ---------------------------------------------------------------- bandwidth
fn bandwidth() {
    let n: usize = 1 << 28; // 2 GiB of u64
    let buf: Vec<u64> = (0..n as u64).collect();
    for &threads in &[1usize, 2, 4, 8, 10, 12, 16, 20] {
        let chunk = n / threads;
        let mut best = 0.0f64;
        for _ in 0..4 {
            let t0 = Instant::now();
            let total: u64 = std::thread::scope(|s| {
                let hs: Vec<_> = (0..threads).map(|i| {
                    let sl = &buf[i * chunk..(i + 1) * chunk];
                    s.spawn(move || {
                        let mut a = [0u64; 8];
                        for c in sl.chunks_exact(8) { for j in 0..8 { a[j] = a[j].wrapping_add(c[j]); } }
                        a.iter().fold(0u64, |x, y| x.wrapping_add(*y))
                    })
                }).collect();
                hs.into_iter().map(|h| h.join().unwrap()).fold(0u64, |x, y| x.wrapping_add(y))
            });
            black_box(total);
            best = best.max((chunk * threads * 8) as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
        println!("read-bandwidth threads={:2}  {:7.1} GB/s", threads, best);
    }
}

// ---------------------------------------------------------------- own kernels
fn gemv_bf16(w: &[u16], x: &[f32], out: &mut [f32], inn: usize) {
    out.par_chunks_mut(16).enumerate().for_each(|(ci, oc)| {
        for (k, o) in oc.iter_mut().enumerate() {
            let r = ci * 16 + k;
            let row = &w[r * inn..(r + 1) * inn];
            let mut acc = [0f32; 16];
            for (wc, xc) in row.chunks_exact(16).zip(x.chunks_exact(16)) {
                for i in 0..16 { acc[i] += f32::from_bits((wc[i] as u32) << 16) * xc[i]; }
            }
            *o = acc.iter().sum();
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_row_w4a8(wp: *const u8, xq: *const i8, nblk: usize) -> i32 {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0f);
    let eight = vdupq_n_s8(8);
    let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0));
    let mut b = 0;
    while b + 2 <= nblk {
        let p0 = vld1q_u8(wp.add(b * 16));
        let p1 = vld1q_u8(wp.add(b * 16 + 16));
        let lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p0, mask)), eight);
        let hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p0)), eight);
        let lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p1, mask)), eight);
        let hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p1)), eight);
        a0 = vdotq_s32(a0, lo0, vld1q_s8(xq.add(b * 32)));
        a1 = vdotq_s32(a1, hi0, vld1q_s8(xq.add(b * 32 + 16)));
        a2 = vdotq_s32(a2, lo1, vld1q_s8(xq.add(b * 32 + 32)));
        a3 = vdotq_s32(a3, hi1, vld1q_s8(xq.add(b * 32 + 48)));
        b += 2;
    }
    while b < nblk {
        let p0 = vld1q_u8(wp.add(b * 16));
        let lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p0, mask)), eight);
        let hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p0)), eight);
        a0 = vdotq_s32(a0, lo0, vld1q_s8(xq.add(b * 32)));
        a1 = vdotq_s32(a1, hi0, vld1q_s8(xq.add(b * 32 + 16)));
        b += 1;
    }
    vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)))
}

/// W4A8: per-row-scaled 4-bit weights (block-of-32 layout, byte j = w[j] | w[j+16]<<4) x per-token int8 activations.
fn gemv_w4a8(wq: &[u8], scales: &[f32], x: &[f32], out: &mut [f32], inn: usize) {
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let sx = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let xq: Vec<i8> = x.iter().map(|&v| (v / sx).round().clamp(-127.0, 127.0) as i8).collect();
    let row_bytes = inn / 2;
    let nblk = inn / 32;
    out.par_chunks_mut(16).enumerate().for_each(|(ci, oc)| {
        for (k, o) in oc.iter_mut().enumerate() {
            let r = ci * 16 + k;
            let d = unsafe { dot_row_w4a8(wq.as_ptr().add(r * row_bytes), xq.as_ptr(), nblk) };
            *o = d as f32 * sx * scales[r];
        }
    });
}

/// Repack the repo's u32 nibble layout into the block-of-32 W4A8 layout.
fn repack(op: &SaccadeLinearOp) -> Vec<u8> {
    let (o, i) = (op.out_features, op.in_features);
    let pk = &op.cache.packed_weights;
    let nib = |r: usize, e: usize| ((pk[r * (i / 8) + e / 8] >> ((e % 8) * 4)) & 0xF) as u8;
    let mut v = vec![0u8; o * i / 2];
    for r in 0..o {
        for b in 0..i / 32 {
            for j in 0..16 {
                v[r * i / 2 + b * 16 + j] = nib(r, b * 32 + j) | (nib(r, b * 32 + j + 16) << 4);
            }
        }
    }
    v
}

// ---------------------------------------------------------------- gemv bench
fn time_pool<F: FnMut(usize)>(k: usize, mut f: F) -> f64 {
    for i in 0..k { f(i); } // warm-up sweep
    let mut samples = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        for i in 0..k { f(i); }
        samples.push(t0.elapsed().as_secs_f64() / k as f64);
    }
    median(samples) * 1e3 // ms / call
}

fn gemv(out_f: usize, in_f: usize) -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let params = (out_f * in_f) as f64;
    let k = ((1.5e9 / (params * 4.0)).ceil() as usize).clamp(3, 64);
    println!("\n=== GEMV b=1  out={} in={}  ({:.1}M params)  pool K={} ===", out_f, in_f, params / 1e6, k);

    let w = Tensor::randn(0f32, 0.02f32, (out_f, in_f), &dev)?;
    let x32 = Tensor::randn(0f32, 1f32, (1, 1, in_f), &dev)?;
    let x_vec: Vec<f32> = x32.flatten_all()?.to_vec1()?;
    let mut rows: Vec<(String, f64, f64)> = Vec::new(); // name, ms, GB/s on true bytes

    // candle F32 / BF16 / F16
    for (name, dt, bytes) in [("candle F32", DType::F32, 4.0), ("candle BF16", DType::BF16, 2.0), ("candle F16", DType::F16, 2.0)] {
        let wt = w.to_dtype(dt)?;
        let pool: Vec<_> = (0..k).map(|_| candle_nn::Linear::new(wt.copy().unwrap(), None)).collect();
        let x = x32.to_dtype(dt)?;
        match pool[0].forward(&x) {
            Ok(_) => {
                let ms = time_pool(k, |i| { black_box(pool[i].forward(&x).unwrap()); });
                rows.push((name.into(), ms, params * bytes / (ms / 1e3) / 1e9));
            }
            Err(e) => println!("  {name}: unsupported ({e})"),
        }
    }

    // own bf16 (strong 2-byte baseline: same DRAM traffic as FP16)
    {
        let wb: Vec<u16> = w.to_dtype(DType::BF16)?.flatten_all()?.to_vec1::<half::bf16>()?.iter().map(|b| b.to_bits()).collect();
        let pool: Vec<Vec<u16>> = (0..k).map(|_| wb.clone()).collect();
        let mut out = vec![0f32; out_f];
        let ms = time_pool(k, |i| { gemv_bf16(&pool[i], &x_vec, &mut out, in_f); black_box(&out); });
        rows.push(("own NEON BF16 GEMV".into(), ms, params * 2.0 / (ms / 1e3) / 1e9));
    }

    // Saccade op (repo kernel)
    let thr = compute_percentile_threshold(&w, 0.15)?;
    let b = compress_tensor_to_saccade(&w, thr)?;
    let sp = b.get("delta_row_ptrs").map(|_| SparseDeltaMatrix {
        row_ptrs: b["delta_row_ptrs"].clone(), col_indices: b["delta_col_indices"].clone(),
        values: b["delta_values"].clone(), scale: b["delta_scale"].clone(),
    });
    let cfg = SaccadeConfig { t4: f32::MAX, t8: f32::MAX, block_size: 16, heuristic: variance_heuristic };
    let op0 = SaccadeLinearOp::new(b["packed_base"].clone(), b["scale_base"].clone(), sp, cfg, out_f, in_f)?;
    let nnz = op0.cache.csc.as_ref().map(|c| c.values_f32.len()).unwrap_or(0);
    let fill = nnz as f64 / params;
    let mut pool: Vec<SaccadeLinearOp> = (0..k).map(|_| op0.clone()).collect();
    let x16 = x32.to_dtype(DType::F16)?;

    let base_bytes = params * 0.5;
    let sparse_bytes_mem = nnz as f64 * 8.0; // u32 row idx + f32 value as held in memory by KernelCache
    for op in pool.iter_mut() { op.config.t4 = f32::MAX; op.config.t8 = f32::MAX; }
    let ms_base = time_pool(k, |i| { black_box(x16.apply_op1_no_bwd(&pool[i]).unwrap()); });
    rows.push(("repo kernel, 4-bit base only (route OFF)".into(), ms_base, base_bytes / (ms_base / 1e3) / 1e9));
    for op in pool.iter_mut() { op.config.t4 = -1e30; op.config.t8 = -1e30; }
    let ms_sp = time_pool(k, |i| { black_box(x16.apply_op1_no_bwd(&pool[i]).unwrap()); });
    rows.push((format!("repo kernel, base + CSC delta (route ON, fill {:.1}%)", fill * 100.0), ms_sp, (base_bytes + sparse_bytes_mem) / (ms_sp / 1e3) / 1e9));
    drop(pool);

    // own W4A8 sdot kernel on the same packed weights
    #[cfg(target_arch = "aarch64")]
    {
        let wq = repack(&op0);
        let pool: Vec<Vec<u8>> = (0..k).map(|_| wq.clone()).collect();
        let mut out = vec![0f32; out_f];
        let ms = time_pool(k, |i| { gemv_w4a8(&pool[i], &op0.cache.scales_f32, &x_vec, &mut out, in_f); black_box(&out); });
        rows.push(("own NEON W4A8 sdot GEMV (base only)".into(), ms, base_bytes / (ms / 1e3) / 1e9));

        // correctness: compare against repo base-only output
        let mut op_chk = op0.clone(); op_chk.config.t4 = f32::MAX; op_chk.config.t8 = f32::MAX;
        let ref_out: Vec<f32> = x16.apply_op1_no_bwd(&op_chk)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        gemv_w4a8(&wq, &op0.cache.scales_f32, &x_vec, &mut out, in_f);
        let num: f64 = ref_out.iter().zip(&out).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = ref_out.iter().map(|a| (*a as f64).powi(2)).sum();
        println!("  W4A8 vs repo base-only relative L2 error: {:.4}% (int8 activation rounding)", (num / den).sqrt() * 100.0);
    }

    let f16_ms = rows.iter().find(|r| r.0 == "own NEON BF16 GEMV").map(|r| r.1).unwrap();
    let f32_ms = rows.iter().find(|r| r.0 == "candle F32").map(|r| r.1).unwrap();
    println!("{:<62} {:>9} {:>9} {:>9} {:>9}", "kernel", "ms/GEMV", "GB/s", "vs F32", "vs 16bit");
    for (n, ms, gbs) in &rows {
        println!("{:<62} {:>9.3} {:>9.1} {:>8.2}x {:>8.2}x", n, ms, gbs, f32_ms / ms, f16_ms / ms);
    }
    println!("  (sparse delta in-memory cost: {:.2} B/param at fill {:.1}% vs 0.5 B/param base; 16-bit = 2 B/param)", sparse_bytes_mem / params, fill * 100.0);
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("bw") => bandwidth(),
        Some("gemv") => {
            if a.len() >= 4 {
                gemv(a[2].parse().unwrap(), a[3].parse().unwrap())?;
            } else {
                for (o, i) in [(4864usize, 896usize), (11008, 2048), (18944, 3584)] { gemv(o, i)?; }
            }
        }
        _ => println!("usage: audit_bench bw | gemv [out in]"),
    }
    Ok(())
}
