"""Kronos inference in a short-lived child process (keeps torch out of the always-on worker).

  python kronos_infer.py <input.npz> <output.npz>

input.npz : x (n, T, 6) raw open/high/low/close/volume/amount, x_stamp (n, T, 5), y_stamp (n, H, 5)
output.npz: paths (n, KRONOS_SAMPLES, H, 4) predicted open/high/low/close in price units

The worker starts this once per scan for all coins. When it exits, all torch memory is returned to the OS,
so the 512 MB budget only has to hold torch during the few minutes of a scan.
Same maths as KronosPredictor.predict_batch (per-series z-score, clip, sample, de-normalise) but without
pandas, and every sampled path is kept instead of averaged.
"""
import os
import resource
import sys
import time

import numpy as np

import config as C


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(inp, out):
    t0 = time.time()
    import torch
    torch.set_num_threads(max(1, C.KRONOS_THREADS))
    torch.set_grad_enabled(False)
    from kronos_model import Kronos, KronosTokenizer
    from kronos_model.kronos import auto_regressive_inference

    tokenizer = KronosTokenizer.from_pretrained(C.KRONOS_TOKENIZER).eval()
    model = Kronos.from_pretrained(C.KRONOS_MODEL).eval()
    print(f"[KRONOS] loaded {C.KRONOS_MODEL} in {time.time() - t0:.1f}s (rss peak {peak_rss_mb():.0f} MB)", flush=True)
    if C.KRONOS_SEED:
        torch.manual_seed(C.KRONOS_SEED)

    d = np.load(inp)
    x_all, xs_all, ys_all = d['x'], d['x_stamp'], d['y_stamp']
    n, _, _ = x_all.shape
    h = ys_all.shape[1]
    s_total, batch = C.KRONOS_SAMPLES, max(1, min(C.KRONOS_BATCH, C.KRONOS_SAMPLES))
    clip = 5.0
    paths = np.empty((n, s_total, h, 4), dtype=np.float32)

    for i in range(n):
        t1 = time.time()
        x = x_all[i].astype(np.float32)
        mean, std = x.mean(axis=0), x.std(axis=0)
        xn = np.clip((x - mean) / (std + 1e-5), -clip, clip)
        done = 0
        while done < s_total:
            b = min(batch, s_total - done)
            with torch.inference_mode():
                pred = auto_regressive_inference(
                    tokenizer, model,
                    torch.from_numpy(np.repeat(xn[None], b, axis=0)),
                    torch.from_numpy(np.repeat(xs_all[i][None].astype(np.float32), b, axis=0)),
                    torch.from_numpy(np.repeat(ys_all[i][None].astype(np.float32), b, axis=0)),
                    C.KRONOS_MAX_CONTEXT, h, clip, C.KRONOS_T, 0, C.KRONOS_TOP_P, 1, False,
                )
            pred = pred[:, -h:, :] * (std + 1e-5) + mean          # (b, h, 6) back to prices
            paths[i, done:done + b] = pred[:, :, :4]
            done += b
        print(f"[KRONOS] series {i + 1}/{n}: {s_total} paths in {time.time() - t1:.1f}s "
              f"(rss peak {peak_rss_mb():.0f} MB)", flush=True)

    tmp = out + '.tmp.npz'
    np.savez(tmp, paths=paths)
    os.replace(tmp, out)
    print(f"[KRONOS] done in {time.time() - t0:.1f}s, peak memory {peak_rss_mb():.0f} MB", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
