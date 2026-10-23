"""Model efficiency report: parameters, analytical MACs, and latency benchmarks.

Loads a trained keras model and reports:
  - total parameter count
  - per-layer breakdown (name, output shape, params)
  - MACs estimated analytically (Conv2D / DepthwiseConv2D / Dense layers)
  - latency: mean / p50 / p95 ms per utterance via model.predict on random input

Writes:
  results/tables/efficiency.tex   (booktabs table: params, GMACs, latency @ batch 1)

Requires TensorFlow (run on Kaggle, where TF works). Locally it prints a clear
message and exits non-zero once it actually needs to load a model.

Usage:  python scripts/efficiency.py [--model models/dvd_full_seed42/best.keras]
        [--batch_sizes 1 8 32] [--repeats 20]
"""

import os
import sys
import time
import argparse

import numpy as np

# Known model input (log-mel spectrogram patch): (n_mels, frames, 1)
INPUT_SHAPE = (128, 128, 1)


def estimate_macs(model, input_shape=INPUT_SHAPE):
    """Analytical MAC estimate by walking the built model's layers.

    Conv2D:      kh * kw * C_in * C_out * H_out * W_out
    Depthwise:   kh * kw * C_out * H_out * W_out
    Dense:       in_features * out_features
    Other layers count as 0 (pools/activations/norm are negligible).

    Returns (total_macs, per_layer) where per_layer is a list of
    (name, layer_type, output_shape_str, n_params, macs) tuples.
    """
    per_layer = []
    total = 0
    for layer in model.layers:
        ltype = type(layer).__name__
        macs = 0
        try:
            in_shape = layer.input_shape
            out_shape = layer.output_shape
        except Exception:
            in_shape = out_shape = None
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        if ltype in ("Conv2D", "DepthwiseConv2D", "SeparableConv2D") and out_shape:
            h, w, c_out = out_shape[-3:]
            c_in = in_shape[-1]
            kh, kw = layer.kernel_size
            if ltype == "Conv2D":
                macs = kh * kw * c_in * c_out * h * w
            else:  # depthwise/separable: per-channel kernels
                macs = kh * kw * c_out * h * w
        elif ltype == "Dense" and in_shape:
            macs = int(in_shape[-1]) * int(layer.units)
        total += int(macs)
        params = 0
        try:
            params = int(layer.count_params())
        except Exception:
            pass
        per_layer.append((layer.name, ltype,
                          str(tuple(None if s is None else int(s) for s in out_shape))
                          if out_shape else "-",
                          params, int(macs)))
    return int(total), per_layer


def benchmark_latency(model, batch_sizes, repeats):
    """model.predict latency on random input; returns {bs: (mean, p50, p95) ms/utt}."""
    results = {}
    for bs in batch_sizes:
        x = np.random.randn(bs, *INPUT_SHAPE).astype("float32")
        for _ in range(3):  # warmup
            model.predict(x, verbose=0)
        times_ms = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.predict(x, verbose=0)
            times_ms.append((time.perf_counter() - t0) * 1000.0 / bs)
        results[bs] = (float(np.mean(times_ms)), float(np.percentile(times_ms, 50)),
                       float(np.percentile(times_ms, 95)))
    return results


def write_tex(path, model_name, n_params, gmacs, lat_b1):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Runtime efficiency of the spoofing countermeasure "
        f"({model_name}) on a single GPU. Latency is measured with "
        r"\texttt{model.predict} on random input (20 repeats, 3 warmup).}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Params (M) & GMACs & Latency (ms/utt, batch 1) \\",
        r"\midrule",
        f"{model_name} & {n_params / 1e6:.2f} & {gmacs:.2f} & "
        f"{lat_b1[0]:.1f} (p50 {lat_b1[1]:.1f}, p95 {lat_b1[2]:.1f}) \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/dvd_full_seed42/best.keras")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")

    try:
        from tensorflow import keras  # noqa: deferred import — TF may be unavailable
    except Exception as e:
        print(f"TensorFlow is required to load '{args.model}' and is not usable here ({e}).")
        print("Run this script on Kaggle, where TF works: see kaggle/ and src/train_kaggle.py.")
        sys.exit(1)

    model = keras.models.load_model(args.model, compile=False)
    print(f"loaded {args.model}")
    model.build((None, *INPUT_SHAPE))

    n_params = int(model.count_params())
    macs, per_layer = estimate_macs(model, INPUT_SHAPE)
    print(f"\ntotal params: {n_params:,}")
    print(f"total MACs:   {macs:,}  ({macs / 1e9:.2f} GMACs)\n")
    print(f"{'layer':<32} {'type':<20} {'output shape':<24} {'params':>10} {'MACs':>14}")
    for name, ltype, oshape, params, m in per_layer:
        print(f"{name:<32} {ltype:<20} {oshape:<24} {params:>10,} {m:>14,}")

    print(f"\nlatency benchmark: batch sizes {args.batch_sizes}, {args.repeats} repeats (+3 warmup)")
    lat = benchmark_latency(model, args.batch_sizes, args.repeats)
    for bs, (mean, p50, p95) in lat.items():
        print(f"  batch {bs:>3}: mean {mean:7.2f} ms/utt   p50 {p50:7.2f}   p95 {p95:7.2f}")

    os.makedirs(args.out, exist_ok=True)
    model_name = os.path.basename(os.path.dirname(args.model)) or "model"
    write_tex(os.path.join(args.out, "efficiency.tex"), model_name, n_params,
              macs / 1e9, lat[args.batch_sizes[0]])
    print("\nsummary:")
    print(f"  params: {n_params / 1e6:.2f} M")
    print(f"  compute: {macs / 1e9:.2f} GMACs per utterance")
    print(f"  latency @ batch {args.batch_sizes[0]}: {lat[args.batch_sizes[0]][0]:.1f} ms/utt (mean)")


if __name__ == "__main__":
    main()
