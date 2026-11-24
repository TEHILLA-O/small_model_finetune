"""Plot results/inference/benchmark.csv."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

df = pd.read_csv("results/inference/benchmark.csv")
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for v, g in df.groupby("variant", sort=False):
    axes[0].plot(g.batch_size, g.throughput_sent_per_s, "o-", label=v)
    axes[1].plot(g.batch_size, g.p95_ms, "o-", label=v)
axes[0].set_xscale("log", base=2)
axes[1].set_xscale("log", base=2)
axes[0].set_xlabel("batch size")
axes[1].set_xlabel("batch size")
axes[0].set_ylabel("sentences per second")
axes[1].set_ylabel("p95 latency per batch (ms)")
axes[0].legend(fontsize=7)
one = df[df.batch_size == 1]
axes[2].scatter(one.size_mb, one.accuracy)
for _, r in one.iterrows():
    axes[2].annotate(r.variant, (r.size_mb, r.accuracy), fontsize=7)
axes[2].set_xlabel("model file size (MB)")
axes[2].set_ylabel("SST-2 validation accuracy")
fig.tight_layout()
fig.savefig("results/inference/benchmark.png", dpi=120)
