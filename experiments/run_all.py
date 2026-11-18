"""All experiments: python experiments/run_all.py  (about 10 minutes on a laptop).

Writes figures/*.png, results/*.csv, results/summary.json and assets/ci_shrinking.gif.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from scipy import stats                                              # noqa: E402
from sklearn.neural_network import MLPRegressor                      # noqa: E402
from sklearn.ensemble import RandomForestClassifier                  # noqa: E402
from sklearn.datasets import make_classification                     # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split  # noqa: E402

from rigor import ber, compare, leakage, seeds                       # noqa: E402
import ofdm_task as T                                                # noqa: E402

FIG, RES, ASSETS = (os.path.join(ROOT, d) for d in ("figures", "results", "assets"))
for d in (FIG, RES, ASSETS):
    os.makedirs(d, exist_ok=True)
S = {}

# ----------------------------------------------------------------------------- style
BG, INK, MUTE = "#0f1117", "#e8e6e3", "#8b8f98"
RED, GREEN, AMBER, BLUE, VIOLET = "#ff4d4d", "#2ee59d", "#ffb020", "#4da3ff", "#b18cff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "#151822", "savefig.facecolor": BG,
    "axes.edgecolor": "#3a3f4b", "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "text.color": INK, "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#262a35", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 170, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def mlp(seed, hidden=(128, 128), iters=400):
    return MLPRegressor(hidden_layer_sizes=hidden, max_iter=iters, learning_rate_init=1e-3,
                        early_stopping=False, random_state=seed, tol=1e-6, n_iter_no_change=50)


# ----------------------------------------------------------------- 1. BER intervals
def exp_ber():
    p = 1e-3
    ks = np.unique(np.round(np.geomspace(1, 1000, 18)).astype(int))
    rows = []
    for k in ks:
        n = int(round(k / p))
        rows.append(dict(expected_errors=k, wald=ber.coverage(ber.wald, p, n, 6000),
                         wilson=ber.coverage(ber.wilson, p, n, 6000),
                         clopper=ber.coverage(ber.clopper_pearson, p, n, 6000)))
    cov = pd.DataFrame(rows)
    cov.to_csv(os.path.join(RES, "ci_coverage.csv"), index=False)
    # BPSK in AWGN: fixed 1e5 bits per point versus "until 100 errors"
    rng = np.random.default_rng(1)
    snr = np.arange(0, 11)
    th = 0.5 * stats.norm.sf(np.sqrt(2 * 10 ** (snr / 10))) * 2
    fixed, adapt = [], []
    for s, pt in zip(snr, th):
        n = 100_000
        k = rng.binomial(n, pt)
        fixed.append((k, n))
        n2 = 0
        k2 = 0
        while k2 < 100 and n2 < 2e9:
            m = int(min(max(2_000_000, 100 / pt / 4), 2e8))
            k2 += rng.binomial(m, pt)
            n2 += m
        adapt.append((k2, n2))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.4))
    a1.axhline(95, color=MUTE, ls="--", lw=1)
    a1.semilogx(cov.expected_errors, cov.wald * 100, "-o", color=RED, lw=2.4, label="Wald  p ± 1.96·√(p(1−p)/n)")
    a1.semilogx(cov.expected_errors, cov.wilson * 100, "-s", color=GREEN, lw=2.4, label="Wilson score")
    a1.semilogx(cov.expected_errors, cov.clopper * 100, "-D", color=BLUE, lw=2.4, label="Clopper–Pearson (exact)")
    a1.set_xlabel("expected number of errors in the run  (n·BER)")
    a1.set_ylabel("how often the 95 % interval contains the true BER (%)")
    a1.set_title("A '95 %' interval is not always 95 %")
    a1.set_ylim(55, 101)
    a1.legend(loc="lower right")
    x = np.linspace(-0.2, 10.4, 300)
    a2.semilogy(x, 0.5 * 2 * stats.norm.sf(np.sqrt(2 * 10 ** (x / 10))), color=MUTE, lw=1.5, label="theory  Q(√(2Eb/N0))")
    for (k, n), s in zip(fixed, snr):
        if k > 0:
            lo, hi = ber.clopper_pearson(k, n)
            a2.errorbar(s - 0.12, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", color=RED, ms=7, capsize=4, lw=2)
        else:
            a2.plot([s - 0.12, s - 0.12], [1e-12, ber.rule_of_three(n)], color=RED, lw=2)
            a2.plot(s - 0.12, ber.rule_of_three(n), "v", color=RED, ms=9)
    for (k, n), s in zip(adapt, snr):
        lo, hi = ber.clopper_pearson(k, n)
        a2.errorbar(s + 0.12, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="s", color=GREEN, ms=7, capsize=4, lw=2)
    a2.plot([], [], "o", color=RED, label="10⁵ bits per point (0 errors: ▼ = rule-of-three bound)")
    a2.plot([], [], "s", color=GREEN, label="simulate until 100 errors")
    a2.set_ylim(1e-7, 0.3)
    a2.set_xlabel("Eb/N0 [dB]   (BPSK, AWGN)")
    a2.set_ylabel("bit error rate with 95 % Clopper–Pearson interval")
    a2.set_title("Fixed bits per point vs. a fixed number of errors")
    a2.legend(loc="lower left", fontsize=10.5)
    save(fig, "fig1_ber_confidence.png")
    S["ber"] = dict(coverage_at_3_errors={m: float(cov.iloc[(cov.expected_errors - 3).abs().argmin()][m]) for m in ["wald", "wilson", "clopper"]},
                    bits_for_1e5_pm20pct=ber.bits_needed(1e-5, 0.2), bits_for_1e6_pm20pct=ber.bits_needed(1e-6, 0.2),
                    fixed_zero_error_points=[int(s) for (k, n), s in zip(fixed, snr) if k == 0],
                    adaptive_bits_at_10dB=int(adapt[-1][1]))
    return adapt


# ------------------------------------------------- 2. leakage and 3. honest baselines
def exp_leakage():
    snr = 10
    Yp, H, g = T.dataset(groups=300, per_group=40, snr_db=snr, seed=3)
    X = T.to_real(Yp)
    Y = T.to_real(H)
    res = {}
    for name, (tr, te) in {"random split": leakage.random_split(len(X), 0.25, seed=0),
                           "split by channel": leakage.group_split(g, 0.25, seed=0)}.items():
        m = mlp(0).fit(X[tr], Y[tr])
        Hhat = T.to_complex(m.predict(X[te]))
        res[name] = dict(nmse=T.nmse_db(Hhat, H[te]), overlap=leakage.overlap(g, tr, te),
                         lmmse=T.nmse_db(T.lmmse(Yp[te], snr), H[te]), ls=T.nmse_db(T.ls_interp(Yp[te]), H[te]))
    # truly new channels: a fresh draw
    Yn, Hn, _ = T.dataset(groups=2000, per_group=1, snr_db=snr, seed=99)
    tr_r, _ = leakage.random_split(len(X), 0.25, seed=0)
    m_r = mlp(0).fit(X[tr_r], Y[tr_r])
    res["random split, then fresh channels"] = dict(nmse=T.nmse_db(T.to_complex(m_r.predict(T.to_real(Yn))), Hn),
                                                    overlap=0.0, lmmse=T.nmse_db(T.lmmse(Yn, snr), Hn),
                                                    ls=T.nmse_db(T.ls_interp(Yn), Hn))
    S["leakage"] = {k: {a: round(b, 2) for a, b in v.items()} for k, v in res.items()}
    pd.DataFrame(res).T.to_csv(os.path.join(RES, "leakage.csv"))
    fig, ax = plt.subplots(figsize=(12.5, 6))
    names = list(res)
    xs = np.arange(len(names))
    ax.bar(xs - 0.27, [res[n]["ls"] for n in names], 0.25, color=MUTE, label="LS + linear interpolation")
    ax.bar(xs, [res[n]["lmmse"] for n in names], 0.25, color=BLUE, label="LMMSE (optimal for this channel)")
    ax.bar(xs + 0.27, [res[n]["nmse"] for n in names], 0.25, color=[RED, GREEN, GREEN], label="neural network")
    for x, n in zip(xs, names):
        v = res[n]["nmse"]
        ax.text(x + 0.27, v - 0.6, f"{v:.1f} dB", ha="center", va="top", fontsize=11, fontweight="bold")
        ax.text(x, 1.2, f"{res[n]['overlap'] * 100:.0f} % of test samples\nshare a channel with training", ha="center", fontsize=9.5, color=MUTE)
    ax.set_xticks(xs)
    ax.set_xticklabels(["random split\n(leaky)", "split by channel\nrealization", "random-split model\non fresh channels"])
    ax.set_ylabel("NMSE of the channel estimate [dB]  (lower is better)")
    ax.set_title("A network that 'beats' the optimal estimator is a warning sign\n10 dB SNR, 300 channel realizations × 40 noisy observations each")
    ax.set_ylim(min(res[n]["nmse"] for n in names) - 4, 4)
    ax.axhline(0, color="#3a3f4b", lw=1)
    ax.legend(loc="lower right")
    save(fig, "fig2_leakage.png")
    return res


def exp_baselines():
    snrs = np.arange(0, 31, 5)
    rows = []
    Ytr, Htr, _ = T.dataset(4000, 1, 15, seed=5)
    m15 = mlp(0).fit(T.to_real(Ytr), T.to_real(Htr))
    wrong = np.exp(-0.15 * np.arange(T.L))     # assumes a longer delay spread than the truth
    wrong /= wrong.sum()
    Rw = (T.F * wrong) @ T.F.conj().T
    for s in snrs:
        Yp, H, _ = T.dataset(3000, 1, s, seed=int(100 + s))
        rows.append(dict(snr=s, ls=T.nmse_db(T.ls_interp(Yp), H), lmmse=T.nmse_db(T.lmmse(Yp, s), H),
                         lmmse_mismatch=T.nmse_db(T.lmmse(Yp, s, Rw), H),
                         dnn_15dB=T.nmse_db(T.to_complex(m15.predict(T.to_real(Yp))), H)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "baselines.csv"), index=False)
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.plot(df.snr, df.ls, "-o", color=MUTE, lw=2.4, label="LS + interpolation (the usual baseline)")
    ax.plot(df.snr, df.lmmse_mismatch, "-^", color=AMBER, lw=2.4, label="LMMSE with a wrong delay-spread assumption")
    ax.plot(df.snr, df.lmmse, "-s", color=BLUE, lw=2.8, label="LMMSE with the true statistics (the bound)")
    ax.plot(df.snr, df.dnn_15dB, "-D", color=RED, lw=2.8, label="neural network trained at 15 dB only")
    ax.axvline(15, color=RED, ls=":", lw=1)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("NMSE [dB]")
    ax.set_title("Which baseline you show decides the story")
    ax.legend(loc="lower left")
    save(fig, "fig3_baselines.png")
    S["baselines"] = df.round(2).to_dict("records")
    return df


# ----------------------------------------------------------------------- 4. seeds
def exp_seeds(nseed=20):
    Yp, H, _ = T.dataset(400, 1, 10, seed=7)
    Yt, Ht, _ = T.dataset(3000, 1, 10, seed=8)
    vals = []
    for sd in range(nseed):
        m = mlp(sd, hidden=(256, 256), iters=150).fit(T.to_real(Yp), T.to_real(H))
        vals.append(T.nmse_db(T.to_complex(m.predict(T.to_real(Yt))), Ht))
    vals = np.array(vals)
    pd.DataFrame(dict(seed=np.arange(nseed), nmse_db=vals)).to_csv(os.path.join(RES, "seeds.csv"), index=False)
    sm = seeds.summary(vals)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    order = np.argsort(vals)
    cc = [GREEN if i == 0 else (RED if i == nseed - 1 else VIOLET) for i in range(nseed)]
    a1.axhspan(sm["ci_low"], sm["ci_high"], color=AMBER, alpha=0.18, lw=0)
    a1.axhline(sm["mean"], color=AMBER, lw=2, label=f"mean {sm['mean']:.2f} dB, 95 % CI of the mean [{sm['ci_low']:.2f}, {sm['ci_high']:.2f}]")
    a1.vlines(np.arange(nseed), sm["mean"], vals[order], color=cc, lw=2)
    a1.scatter(np.arange(nseed), vals[order], color=cc, s=70, zorder=3, edgecolor="#000")
    pad = (vals.max() - vals.min()) * 0.6
    a1.set_ylim(vals.min() - pad, vals.max() + pad)
    a1.set_xticks([])
    a1.set_xlabel("20 training runs, sorted (only the random seed differs)")
    a1.set_ylabel("test NMSE [dB]")
    a1.set_title(f"Same data, same network: best {vals.min():.2f} dB, worst {vals.max():.2f} dB")
    a1.legend(loc="lower right", fontsize=10.5)
    ks = [1, 3, 5, 10]
    data = [seeds.best_of(vals, k) for k in ks]
    parts = a2.violinplot(data, positions=range(len(ks)), showmedians=True)
    for b in parts["bodies"]:
        b.set_facecolor(VIOLET)
        b.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(INK)
    a2.set_xticks(range(len(ks)))
    a2.set_xticklabels([f"best of {k}" for k in ks])
    a2.set_ylabel("reported NMSE [dB]")
    a2.set_title("Reporting the best of k seeds looks better for free")
    save(fig, "fig4_seeds.png")
    S["seeds"] = dict(mean=round(sm["mean"], 3), std=round(sm["std"], 3), ci=[round(sm["ci_low"], 3), round(sm["ci_high"], 3)],
                      best=round(float(vals.min()), 3), worst=round(float(vals.max()), 3),
                      best_of_5_median=round(float(np.median(seeds.best_of(vals, 5))), 3))
    return vals


# ------------------------------------------------------------ 5. significance tests
def exp_significance(reps=4000):
    """How big must the test set be before a small gain is real? Power of the exact McNemar
    test when model B is right on a fraction delta more test samples than model A, and the
    two models disagree on 4 % of the samples apart from that (paired, correlated errors)."""
    rng = np.random.default_rng(11)
    sizes = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    deltas = [0.005, 0.01, 0.02, 0.05]
    base = 0.02                                    # P(A right, B wrong)
    rows = []
    for dlt in deltas:
        q01, q10 = base, base + dlt                # A-only correct, B-only correct
        for n in sizes:
            c = rng.multinomial(n, [q01, q10, 1 - q01 - q10], size=reps)
            n01, n10 = c[:, 0], c[:, 1]
            m = n01 + n10
            p = np.where(m > 0, np.minimum(1, 2 * stats.binom.cdf(np.minimum(n01, n10), np.maximum(m, 1), 0.5)), 1.0)
            rows.append(dict(delta=dlt, n=int(n), power=float(np.mean((p < 0.05) & (n10 > n01)))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "mcnemar_power.csv"), index=False)
    # multiple comparisons: m independent tests, no real effect
    m = np.arange(1, 41)
    fwer = 1 - 0.95 ** m
    sim = rng.uniform(size=(20000, 20))
    any_raw = (sim < 0.05).any(1).mean()
    any_holm = np.mean([compare.holm(row).any() for row in sim[:5000]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    cols = [RED, AMBER, BLUE, GREEN]
    need = {}
    for dlt, c in zip(deltas, cols):
        d = df[df.delta == dlt]
        a1.semilogx(d.n, d.power * 100, "-o", color=c, lw=2.6, ms=7, label=f"B right on {dlt * 100:g} % more samples")
        above = d[d.power >= 0.8]
        need[f"{dlt * 100:g}%"] = int(above.n.min()) if len(above) else None
    a1.axhline(80, color=MUTE, ls="--", lw=1.2)
    a1.text(105, 75, "80 % power", color=MUTE)
    a1.set_xlabel("test-set size (samples)")
    a1.set_ylabel("chance that McNemar's test (p < 0.05) detects the gain (%)")
    a1.set_title("A real +1 % gain needs thousands of test samples to show")
    a1.set_ylim(0, 102)
    a1.legend(loc="lower right", fontsize=10.5)
    a2.plot(m, fwer * 100, color=RED, lw=2.8, label="no correction: 1 − 0.95ᵐ")
    a2.axhline(5, color=GREEN, lw=2.8, label="Holm–Bonferroni keeps it at 5 %")
    a2.scatter([20], [any_raw * 100], color=RED, s=90, zorder=3, edgecolor="#000")
    a2.scatter([20], [any_holm * 100], color=GREEN, s=90, zorder=3, edgecolor="#000")
    a2.annotate(f"20 comparisons: {any_raw * 100:.0f} %\nchance of ≥ 1 false 'win'", (20, any_raw * 100), xytext=(23, 40), arrowprops=dict(arrowstyle="->", color=INK))
    a2.set_xlabel("number of comparisons (metrics × datasets × SNRs)")
    a2.set_ylabel("probability of at least one false 'significant' result (%)")
    a2.set_title("Many comparisons, no real effect")
    a2.legend(loc="center right")
    save(fig, "fig5_significance.png")
    p1000 = df[(df.delta == 0.01) & (df.n == 1000)].power.iloc[0]
    S["significance"] = dict(n_for_80pct_power=need, power_1pct_at_1000=round(float(p1000), 3),
                             any_of_20_raw=round(float(any_raw), 3), any_of_20_holm=round(float(any_holm), 3))
    return df


# ---------------------------------------------------------------------- animation
def animation():
    from PIL import Image
    rng = np.random.default_rng(4)
    p = 1e-4
    n_tot = 4_000_000
    errs = rng.random(n_tot) < p
    cks = np.unique(np.round(np.geomspace(5_000, n_tot, 40)).astype(int))
    tmp = os.path.join(FIG, "_frames")
    os.makedirs(tmp, exist_ok=True)
    frames = []
    hist = []
    for i, n in enumerate(cks):
        k = int(errs[:n].sum())
        lo, hi = ber.clopper_pearson(k, n)
        hist.append((n, k / n if k else np.nan, lo, hi))
        hh = np.array(hist, float)
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.fill_between(hh[:, 0], np.maximum(hh[:, 2], 1e-7), hh[:, 3], color=VIOLET, alpha=0.35, lw=0)
        ax.plot(hh[:, 0], hh[:, 1], color=INK, lw=2)
        ax.axhline(p, color=GREEN, ls="--", lw=1.6)
        ax.text(6000, p * 1.25, "true BER 10⁻⁴", color=GREEN, fontsize=11)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(5_000, n_tot); ax.set_ylim(1e-6, 2e-3)
        ax.set_xlabel("bits simulated"); ax.set_ylabel("BER estimate, 95 % interval")
        rel = (hi - lo) / 2 / (k / n) * 100 if k else np.inf
        ax.set_title(f"{n:,} bits · {k} errors · interval ±{rel:.0f} %" if k else f"{n:,} bits · 0 errors · BER < {hi:.1e}", fontsize=13)
        fig.tight_layout()
        pth = os.path.join(tmp, f"f{i:03d}.png")
        fig.savefig(pth, dpi=80)
        plt.close(fig)
        frames.append(Image.open(pth).convert("RGB"))
    frames += [frames[-1]] * 8
    pal = frames[-1].quantize(colors=64)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(os.path.join(ASSETS, "ci_shrinking.gif"), save_all=True, append_images=q[1:], duration=160, loop=0, optimize=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)


def main(names=None):
    t0 = time.time()
    path = os.path.join(RES, "summary.json")
    if names and os.path.exists(path):
        S.update(json.load(open(path)))
    exps = [("ber", exp_ber), ("leakage", exp_leakage), ("baselines", exp_baselines),
            ("seeds", exp_seeds), ("significance", exp_significance), ("animation", animation)]
    for name, f in exps:
        if names and name not in names:
            continue
        f()
        print(f"done: {name} ({time.time() - t0:.0f} s)", flush=True)
    if not names:
        S["run_time_s"] = round(time.time() - t0)
    with open(path, "w") as fh:
        json.dump(S, fh, indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
