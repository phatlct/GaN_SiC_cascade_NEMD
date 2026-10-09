#!/usr/bin/env python3
# Hinh cho bai bao (ban sua theo review). Moi hinh mot lenh con:
#   fig2 <legacy.csv> [new.csv]                 trieu chung o dt=1 fs (+ moc tham chieu 0.25 fs neu co new.csv)
#   fig3 <recompute_G.py> "nhan::log::profile::tstart" ...   profile nhiet do (tstart='-' = nua sau production)
#   fig4 <nve_dir> <cutoff_vs_S.csv>            co che: (a) troi NVE + sai so, (b) S theo so cap Ga-N
#   si3  <legacy.csv>                           G_corr theo E (Supplementary)
#   figE --a <log 1 fs ...> --b <log 0.25 fs ...>   nang luong thermostat tich luy theo thoi gian
#   fig7 <new.csv> <recompute_G.py>            G tai mat phang theo EX (bootstrap) vs G_reg
#   fig5 <legacy.csv> <new.csv>                 cung cau truc (G_old->G_corr->G_bal->G_reg) + G_reg n=3
# Output: PNG 300 dpi trong thu muc hien tai.
import sys, os, re, glob, csv, collections as C
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
COL = {"Ga": "#1f5fa8", "N": "#c4501b", "P": "0.35"}
ES = (1, 3, 5, 7, 10)
f = lambda r, k: float(r[k])
q = lambda v: (np.median(v), np.percentile(v, 25), np.percentile(v, 75)) if len(v) else (np.nan,) * 3

def errbar(ax, x, stats, **kw):
    m = np.array([s[0] for s in stats]); lo = np.array([s[1] for s in stats]); hi = np.array([s[2] for s in stats])
    ax.errorbar(x, m, yerr=[m - lo, hi - m], capsize=2, ms=4, **kw)

def spearman(x, y, n=20000, rng=np.random.default_rng(0)):
    rk = lambda a: np.argsort(np.argsort(a)).astype(float); rx, ry = rk(x), rk(y); r = np.corrcoef(rx, ry)[0, 1]
    return r, np.mean([abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(r) for _ in range(n)])

# ---------------- Fig 2: trieu chung ----------------
def fig2(legacy, new=None):
    R = list(csv.DictReader(open(legacy)))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for k, sp in enumerate(("Ga", "N")):
        x = np.array(ES) + (k - 0.5) * 0.25
        errbar(ax[0], x, [q([f(r, "G_old") for r in R if r["species"] == sp and f(r, "E") == E]) for E in ES],
               fmt="o-", color=COL[sp], label=f"$G_{{old}}$, {sp}-PKA")
        errbar(ax[0], x, [q([f(r, "G_corr") for r in R if r["species"] == sp and f(r, "E") == E and r["QC"] == "1"]) for E in ES],
               fmt="s--", color=COL[sp], mfc="white", label=f"$J_c$-based, {sp}-PKA")
        errbar(ax[1], x, [q([f(r, "S") for r in R if r["species"] == sp and f(r, "E") == E]) for E in ES],
               fmt="o-", color=COL[sp], label=f"{sp}-PKA, $\\Delta t$ = 1 fs")
    if new:                                                     # moc tham chieu dt = 0.25 fs (G_bal)
        N = list(csv.DictReader(open(new)))
        P = [r for r in N if r["species"] == "P"]; D = [r for r in N if r["species"] == "N" and f(r, "E") == 10]
        if P:
            g = np.array([f(r, "G_bal") for r in P]); m, s = g.mean(), (g.std(ddof=1) if len(g) > 1 else f(P[0], "sG_bal"))
            ax[0].axhspan(m - s, m + s, color="0.85", zorder=0); ax[0].axhline(m, color="0.4", lw=1, zorder=0)
            ax[0].text(1.0, m * 0.82, f"pristine, $\\Delta t$ = 0.25 fs (n = {len(g)})", fontsize=7, color="0.3")
            sp_ = np.array([f(r, "S_win") for r in P]); ax[1].errorbar(0, sp_.mean(), yerr=sp_.std() if len(sp_) > 1 else 0,
                                                                       fmt="D", mfc="white", color="0.3", ms=5, capsize=2, label="$\\Delta t$ = 0.25 fs")
        if D:
            g = np.array([f(r, "G_bal") for r in D])
            ax[0].errorbar(10.45, g.mean(), yerr=g.std(ddof=1) if len(g) > 1 else f(D[0], "sG_bal"),
                           fmt="*", color="k", ms=9, capsize=2, label=f"$G_{{bal}}$, 10 keV N, 0.25 fs (n = {len(g)})")
            sd = np.array([f(r, "S_win") for r in D]); ax[1].errorbar(10.45, sd.mean(), yerr=sd.std() if len(sd) > 1 else 0,
                                                                     fmt="D", mfc="white", color="0.3", ms=5, capsize=2)
    ax[0].set(xlabel="PKA energy (keV)", ylabel="G (MW m$^{-2}$ K$^{-1}$)", yscale="log")
    ax[0].set_title("(a)", loc="left"); ax[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set(xlabel="PKA energy (keV)", ylabel="$S = J_h + J_c$ (eV ps$^{-1}$)"); ax[1].set_title("(b)", loc="left")
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig("Fig2_symptoms.png", dpi=300, bbox_inches="tight"); print("Fig2_symptoms.png")

# ---------------- Fig 3: profile nhiet do ----------------
def fig3(rgpy, specs):
    ns = {"__name__": "lib"}; exec(open(rgpy).read().split("TSTART = None")[0], ns)
    fig, ax = plt.subplots(figsize=(7.0, 2.9)); sty = ["-", "-", "--", ":"]; cols = ["#c4501b", "0.3", "#1f5fa8", "#6a3d9a"]; lim = [282.0, 318.0]
    for i, s in enumerate(specs):
        lab, log, prof, ts = s.split("::")
        ns["TSTART"] = None if ts == "-" else float(ts)
        o = ns["parse_log"](log); Lz, zlo, dt = o[2], o[6], o[8]
        bl = ns["parse_profile"](prof, Lz, zlo, None if ts == "-" else round((float(ts) + 10) / dt))[0]
        z, T = bl[0][:, 0], np.mean([b[:, 1] for b in bl], axis=0); m = (z > zlo + 4) & (z < zlo + Lz - 6) & (T > 250)  # bo bin lop co dinh / bien
        ax.plot(z[m], T[m], sty[i % 4], color=cols[i % 4], lw=1.4, marker="o", ms=2.5, label=lab)
        lim = [min(lim[0], T[m].min() - 1.5), max(lim[1], T[m].max() + 2.5)]
    T0 = 300.0
    ax.axvspan(5.2, 25.2, color="#f4d0c0", alpha=0.5, lw=0); ax.axvspan(89.8, 109.8, color="#c9d9ee", alpha=0.6, lw=0)
    xt = ax.get_xaxis_transform()
    ax.text(15.2, 0.03, "hot reservoir", ha="center", fontsize=7, transform=xt); ax.text(99.8, 0.03, "cold reservoir", ha="center", fontsize=7, transform=xt)
    for Ts in (T0 + 10, T0 - 10): ax.axhline(Ts, color="0.6", lw=0.7, ls="--")
    ax.axvline(38.126, color="k", lw=0.8); ax.text(38.9, 0.10, "$z_{int}$", fontsize=8, transform=xt)
    ax.text(31.6, 0.03, "free GaN", ha="center", fontsize=7, transform=xt)
    ax.set(xlabel="z (Å)", ylabel="T (K)", ylim=lim, xlim=(0, 115)); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout(); fig.savefig("Fig3_profiles.png", dpi=300, bbox_inches="tight"); print("Fig3_profiles.png")

# ---------------- Fig 4: co che ----------------
def nve_slopes(path):
    t, e, blk = [], [], False
    for line in open(path, errors="ignore"):
        if re.match(r"^\s+Step\s", line): blk = True; continue
        if line.startswith("Loop"): blk = False
        if blk and re.match(r"^\s+\d+\s", line): p = line.split(); t.append(float(p[1])); e.append(float(p[5]))
    t, e = np.array(t), np.array(e); h = len(t) // 2
    s = np.polyfit(t, e, 1)[0]; s1 = np.polyfit(t[:h + 1], e[:h + 1], 1)[0]; s2 = np.polyfit(t[h:], e[h:], 1)[0]
    return s, abs(s1 - s2) / 2                                  # sai so: nua chenh lech do doc 2 nua run

def fig4(nved, cutcsv):
    LAB = {"equil_300K": ("Pristine", "0.35"), "postanneal_300K_Ga_1.0keV": ("1 keV Ga-PKA", COL["Ga"]),
           "postanneal_300K_N_10.0keV": ("10 keV N-PKA", COL["N"])}
    d = C.defaultdict(list)
    for p in glob.glob(os.path.join(nved, "log.nve_data.*_dt*")):
        m = re.search(r"log\.nve_data\.(.+)_dt([\d.]+)$", p)
        if m and m.group(1) in LAB: d[m.group(1)].append((float(m.group(2)) * 1e3, *nve_slopes(p)))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1, 1.15, 0.85]})
    for k, (lab, c) in LAB.items():
        if k not in d: continue
        a = np.array(sorted(d[k])); ax[0].errorbar(a[:, 0], np.abs(a[:, 1]), yerr=a[:, 2], fmt="o-", color=c, ms=4, capsize=2, label=lab)
    ref = sorted(d.get("postanneal_300K_N_10.0keV", [(1.0, 1.0, 0)]))[-1]; xs = np.array([0.25, 1.0])
    for pw, ls in ((2, ":"), (4, "--")):
        ax[0].plot(xs, abs(ref[1]) * (xs / ref[0]) ** pw, ls, color="0.6", lw=0.8)
        ax[0].text(0.27, abs(ref[1]) * (0.27 / ref[0]) ** pw, f"$\\propto\\Delta t^{pw}$", fontsize=7, color="0.4", va="bottom")
    ax[0].set(xscale="log", yscale="log", xlabel="Timestep (fs)", ylabel="|d$E_{tot}$/dt| (eV ps$^{-1}$)")
    ax[0].set_xticks([0.25, 0.5, 1]); ax[0].set_xticklabels(["0.25", "0.5", "1"]); ax[0].xaxis.set_minor_formatter(mt.NullFormatter())
    ax[0].set_title("(a) NVE, 300 K", loc="left"); ax[0].legend(fontsize=7, loc="upper left")
    R = list(csv.DictReader(open(cutcsv))); S = np.array([f(r, "S") for r in R]); X = np.array([f(r, "nGaN") for r in R])
    g = C.defaultdict(list)
    for i, r in enumerate(R): g[(r["E"], r["T"], r["species"])].append(i)
    cells = [v for v in g.values() if len(v) >= 3]
    cen = lambda v, c: np.concatenate([np.argsort(np.argsort(v[ix])) - (len(ix) - 1) / 2 for ix in c])
    rs, rx = cen(S, cells), cen(X, cells); rho = np.corrcoef(rs, rx)[0, 1]; rng = np.random.default_rng(0); null = []
    for _ in range(5000):
        null.append(abs(np.corrcoef(rx, np.concatenate([rng.permutation(np.argsort(np.argsort(S[ix])) - (len(ix) - 1) / 2) for ix in cells]))[0, 1]))
    p = np.mean(np.array(null) >= abs(rho))
    for E, c in zip(ES, plt.cm.viridis(np.linspace(0, 0.9, 5))):
        m = np.array([f(r, "E") == E for r in R]); ax[1].scatter(X[m], S[m], s=9, color=c, alpha=0.8, label=f"{E} keV")
    ax[1].text(0.03, 0.97, f"within (E, T, PKA) cells:\n$\\rho$ = {rho:+.2f}, p = {p:.3f}, n = {len(rs)}", transform=ax[1].transAxes, va="top", fontsize=7)
    ax[1].set(xlabel="Ga–N pairs in cutoff transition zone", ylabel="$S$ at $\\Delta t$ = 1 fs (eV ps$^{-1}$)")
    ax[1].set_title("(b)", loc="left"); ax[1].legend(fontsize=6.5, loc="lower right", ncol=2)

    # (c) so cap trong vung chuyen tiep cutoff theo muc hu hai (gia tri do bang cutoff_pairs.py; xem Table 2)
    PC = {"pristine": [(481, 1402)], "1 keV": [(891, 1402), (879, 1361), (896, 1378)], "10 keV": [(3600, 1941), (2005, 1637), (3782, 2015)]}
    for j, (key, mk_, col, off) in enumerate((("Ga–Ga", "s", "0.25", -0.12), ("Ga–N", "o", "#0B6E69", 0.12))):
        for i, (lab, vals) in enumerate(PC.items()):
            v = np.array([x[j] for x in vals]); xs_ = np.full(len(v), i + off) + (np.arange(len(v)) - (len(v) - 1) / 2) * 0.06
            ax[2].plot(xs_, v, mk_, color=col, mfc="white", ms=4.5, label=key if i == 0 else None)
            ax[2].plot([i + off - 0.1, i + off + 0.1], [v.mean()] * 2, "-", color=col, lw=1.6)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(list(PC.keys()), fontsize=7.5); ax[2].set_xlim(-0.5, 2.5)
    ax[2].set(ylabel="pairs in transition zone", yscale="log"); ax[2].set_title("(c)", loc="left")
    ax[2].set_yticks([500, 1000, 2000, 4000]); ax[2].set_yticklabels(["500", "1000", "2000", "4000"])
    ax[2].yaxis.set_minor_formatter(mt.NullFormatter())
    ax[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig("Fig4_mechanism.png", dpi=300); print("Fig4_mechanism.png")

# ---------------- SI: G_corr theo E ----------------
def si3(legacy):
    Q = [r for r in csv.DictReader(open(legacy)) if r["QC"] == "1"]; rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for k, sp in enumerate(("Ga", "N")):
        x = np.array([f(r, "E") for r in Q if r["species"] == sp]); y = np.array([f(r, "G_corr") for r in Q if r["species"] == sp])
        ax.scatter(x + (k - 0.5) * 0.3 + rng.uniform(-0.08, 0.08, len(x)), y, s=10, color=COL[sp], alpha=0.75, label=f"{sp}-PKA")
    E = np.array([f(r, "E") for r in Q]); G = np.array([f(r, "G_corr") for r in Q]); txt = []
    for Em in (5, 10):
        rho, p = spearman(E[E <= Em], G[E <= Em]); txt.append(f"E $\\leq$ {Em} keV: $\\rho$ = {rho:+.2f}, p = {p:.3f}, n = {np.sum(E <= Em)}")
    ax.set(xlabel="PKA energy (keV)", ylabel="$J_c$-based G (MW m$^{-2}$ K$^{-1}$)", yscale="log")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.text(0.5, -0.30, "\n".join(txt), transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.tight_layout(); fig.savefig("FigS3_Jc_based_G.png", dpi=300, bbox_inches="tight"); print("FigS3_Jc_based_G.png")


# ---------------- Fig 5: cung cau truc + G_reg n=3 ----------------
def _tcdf_tail(t, nu, n=20000):                 # P(|T| >= |t|), Student t, khong can scipy
    import math
    c = math.gamma((nu + 1) / 2) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2))
    s = np.linspace(1e-6, 1, n); x = abs(t) / s
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return 2 * trap(c * (1 + x ** 2 / nu) ** (-(nu + 1) / 2) * abs(t) / s ** 2, s)

def fig5(legacy, new):
    import itertools
    L = list(csv.DictReader(open(legacy))); N = list(csv.DictReader(open(new)))
    sel = lambda R, s, sp, E: [r for r in R if int(r["seed"]) == s and r["species"] == sp and abs(f(r, "E") - E) < 1e-6 and r["T"] == "300"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    est = [("$G_{old}$\n1 fs", "G_old", L), ("$J_c$-based\n1 fs", "G_corr", L), ("$G_{bal}$ plane\n0.25 fs", "G_bal", N), ("$G_{reg}$\n0.25 fs", "G_reg", N)]
    mk = {1: ("o", "#c4501b"), 2: ("s", "#1f5fa8"), 3: ("D", "#2a9d5c")}
    for s in (1, 2, 3):
        for i, (lab, key, R) in enumerate(est):
            rr = sel(R, s, "N", 10.0)
            if not rr: continue
            v = f(rr[0], key); x = i + (s - 2) * 0.12
            if v > 0: ax[0].plot(x, v, mk[s][0], color=mk[s][1], ms=5, label=f"Seed {s}" if i == 0 or (i == 2 and s == 3) else None)
            else: ax[0].plot(x, 30, "x", color=mk[s][1], ms=6); ax[0].annotate("ΔT<0", (x, 30), xytext=(3, 3), textcoords="offset points", fontsize=6, color=mk[s][1])
    P = np.array([f(r, "G_reg") for r in N if r["species"] == "P"])
    if len(P): ax[0].axhspan(P.mean() - P.std(ddof=1), P.mean() + P.std(ddof=1), color="0.88", zorder=0); ax[0].text(-0.4, P.mean() * 0.8, "pristine $G_{reg}$", fontsize=6.5, color="0.35")
    h, l = ax[0].get_legend_handles_labels(); u = dict(zip(l, h))
    ax[0].legend(u.values(), u.keys(), fontsize=7, loc="upper right")
    ax[0].set(yscale="log", ylabel="G (MW m$^{-2}$ K$^{-1}$)", xlim=(-0.5, 3.5)); ax[0].set_xticks(range(4)); ax[0].set_xticklabels([e[0] for e in est], fontsize=7.5)
    ax[0].set_title("(a) same 10 keV N-PKA structures", loc="left", fontsize=9)
    D = np.array([f(r, "G_reg") for r in N if r["species"] == "N" and abs(f(r, "E") - 10) < 1e-6])
    for k, (arr, lab, c) in enumerate(((P, "pristine", "0.35"), (D, "10 keV N-PKA", "#c4501b"))):
        ax[1].plot(np.full(len(arr), k) + np.linspace(-0.08, 0.08, len(arr)), arr, "o", color=c, ms=5, mfc="white")
        ax[1].errorbar(k + 0.22, arr.mean(), yerr=arr.std(ddof=1), fmt="_", color=c, ms=14, capsize=4, lw=1.5)
    if len(P) > 1 and len(D) > 1:
        a, b = D.var(ddof=1) / len(D), P.var(ddof=1) / len(P); t = (D.mean() - P.mean()) / np.sqrt(a + b)
        nu = (a + b) ** 2 / (a ** 2 / (len(D) - 1) + b ** 2 / (len(P) - 1)); pw = _tcdf_tail(t, nu)
        allv = np.r_[P, D]; obs = abs(D.mean() - P.mean()); cnt = tot = 0
        for idx in itertools.combinations(range(len(allv)), len(D)):
            m = np.zeros(len(allv), bool); m[list(idx)] = True; tot += 1; cnt += abs(allv[m].mean() - allv[~m].mean()) >= obs - 1e-9
        ax[1].text(0.3, 0.84, f"ratio = {D.mean() / P.mean():.2f}\nWelch p = {pw:.3f}\npermutation p = {cnt / tot:.2f}",
                   transform=ax[1].transAxes, ha="center", va="center", fontsize=7, bbox=dict(fc="white", ec="0.8", lw=0.5))
        print(f"ratio={D.mean()/P.mean():.3f} t={t:.2f} nu={nu:.2f} p_welch={pw:.4f} p_perm={cnt/tot:.3f}")
    ax[1].set(xlim=(-0.5, 1.6), ylabel="$G_{reg}$ (MW m$^{-2}$ K$^{-1}$), $\\Delta t$ = 0.25 fs"); ax[1].set_xticks([0.1, 1.1]); ax[1].set_xticklabels([f"pristine\n(n = {len(P)})", f"10 keV N-PKA\n(n = {len(D)})"])
    ax[1].set_title("(b)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig("Fig5_corrected_G.png", dpi=300); print("Fig5_corrected_G.png")

# ---------------- Hinh nang luong thermostat theo thoi gian ----------------
def read_tally(path):
    dt, blk, rows = 0.001, 0, []
    for line in open(path, errors="ignore"):
        m = re.match(r"^\s*timestep\s+([0-9.eE+-]+)\s*$", line)
        if m: dt = float(m.group(1))
        if re.match(r"^\s+Step\s", line) and "f_fix_hot" in line: blk += 1; cur = []; continue
        if blk == 2 and re.match(r"^\s+\d+\s+[\d.]", line):
            p = line.split(); rows.append((int(p[0]) * dt, float(p[4]), float(p[5])))
    a = np.array(rows); return a[:, 0], a[:, 1], a[:, 2]           # t (ps), Q_h, Q_c (+ = rut khoi nguyen tu)

def figE(groups):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    sty = ["-", "--", ":"]
    for k, (title, logs) in enumerate(groups):
        Ss = []
        for i, lg in enumerate(logs):
            t, qh, qc = read_tally(lg); ls = sty[i % 3]; lab = f"seed {i + 1}"
            ax[k].plot(t, -qh, ls, color="#B23A2E", lw=0.9, alpha=0.85, label=f"hot: energy supplied, {lab}")
            ax[k].plot(t, qc, ls, color="#1F5A7A", lw=0.9, alpha=0.85, label=f"cold: energy removed, {lab}")
            sl_h, sl_c = np.polyfit(t, -qh, 1)[0], np.polyfit(t, qc, 1)[0]; Ss.append(sl_c - sl_h)
            print(f"{title} | {lab}: dE_in/dt (hot) = {sl_h:+.3f}, dE_out/dt (cold) = {sl_c:+.3f}, S = {-(sl_h) + sl_c:+.3f} eV/ps")
        ax[k].axhline(0, color="0.6", lw=0.7)
        ax[k].text(0.03 if k == 0 else 0.97, 0.04, "$S$ = " + ", ".join(f"{v:+.2f}" for v in Ss) + " eV ps$^{-1}$", transform=ax[k].transAxes,
                   ha="left" if k == 0 else "right", va="bottom", fontsize=7, bbox=dict(fc="white", ec="0.8", lw=0.5))
        ax[k].set(xlabel="production time (ps)", ylabel="cumulative energy (eV)"); ax[k].set_title(title, loc="left", fontsize=9)
    h, l = ax[0].get_legend_handles_labels()
    keep = [(hh, ll.split(",")[0]) for hh, ll in zip(h, l) if "seed 1" in ll]
    ax[0].legend([x[0] for x in keep], [x[1] for x in keep], fontsize=7, loc="upper left")
    if any(len(g[1]) > 1 for g in groups):
        from matplotlib.lines import Line2D
        ax[1].legend([Line2D([], [], color="0.3", ls=s_) for s_ in sty[:max(len(g[1]) for g in groups)]],
                     [f"seed {i + 1}" for i in range(max(len(g[1]) for g in groups))], fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig("FigE_reservoir_energy.png", dpi=300); print("FigE_reservoir_energy.png")

# ---------------- Fig 7: G tai mat phang (theo EX) vs G_reg ----------------
def fig7(new_csv, rgpy, B=500):
    ns = {"__name__": "lib"}; exec(open(rgpy).read().split("TSTART = None")[0], ns); ns["TSTART"] = 50.0
    CONV, ZI, EXS = 1.602176634e-7, 38.126, np.array([2, 3, 4, 5, 6], float); rng = np.random.default_rng(0)
    def met(z, T, J, A, zlo):
        g = []
        for e in EXS:
            ns["EX"], ns["FH"] = e, 12.0; g.append(J * CONV / (A * 1e-20) / ns["deltaT"](z, T) / 1e6)
        T1 = np.interp(zlo + 27.0, z, T); m = (z > ZI + 3) & (z < ZI + 25); T2 = np.polyval(np.polyfit(z[m], T[m], 1), ZI + 15)
        return np.array(g), J * CONV / (A * 1e-20) / (T1 - T2) / 1e6
    runs = []
    for r in csv.DictReader(open(new_csv)):
        if "dt025" not in r["path"]: continue
        o = ns["parse_log"](r["path"]); Lz, A, zlo, dt = o[2], o[3], o[6], o[8]
        bl = ns["parse_profile"](r["profile"], Lz, zlo, round(60 / dt))[0]
        z = bl[0][:, 0]; Tb = np.array([b[:, 1] for b in bl]); J = (float(r["Jc2"]) - float(r["Jh2"])) / 2
        g0, r0 = met(z, Tb.mean(0), J, A, zlo)
        bs = [met(z, Tb[rng.integers(0, len(bl), len(bl))].mean(0), J, A, zlo) for _ in range(B)]
        G = np.array([b[0] for b in bs]); R = np.array([b[1] for b in bs])
        runs.append(dict(grp="P" if r["species"] == "P" else "D", seed=int(r["seed"]), g=g0, glo=np.percentile(G, 2.5, 0), ghi=np.percentile(G, 97.5, 0),
                         reg=r0, rlo=np.percentile(R, 2.5), rhi=np.percentile(R, 97.5)))
    runs.sort(key=lambda d: (d["grp"], d["seed"]))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    C = {"P": "0.30", "D": "#B23A2E"}; MK = {1: "o", 2: "s", 3: "D"}; NAME = {"P": "pristine", "D": "10 keV N-PKA"}
    for d in runs:
        x = EXS + (d["seed"] - 2) * 0.09 + (0.03 if d["grp"] == "D" else -0.03)
        ax[0].errorbar(x, d["g"], yerr=[d["g"] - d["glo"], d["ghi"] - d["g"]], fmt=MK[d["seed"]] + "-", color=C[d["grp"]], ms=4,
                       mfc="white" if d["grp"] == "P" else C["D"], capsize=1.5, lw=0.9, elinewidth=0.6, label=f"{NAME[d['grp']]}, seed {d['seed']}")
    top = min(max(d["ghi"].max() for d in runs), 1.8 * max(d["g"].max() for d in runs)); ax[0].set_ylim(top=top)
    ax[0].set(xlabel="exclusion half-width EX (Å)", ylabel="plane-based G (MW m$^{-2}$ K$^{-1}$)", xticks=EXS)
    ax[0].set_title("(a)", loc="left", fontsize=9); ax[0].legend(fontsize=6.3, ncol=2, loc="upper left", frameon=False)
    xs = {("P", "plane"): 0, ("P", "reg"): 1, ("D", "plane"): 2.4, ("D", "reg"): 3.4}
    for grp in ("P", "D"):
        for kind in ("plane", "reg"):
            v = []
            for d in [q for q in runs if q["grp"] == grp]:
                x = xs[(grp, kind)] + (d["seed"] - 2) * 0.12
                if kind == "plane": y, lo, hi = d["g"][1], d["glo"][1], d["ghi"][1]
                else: y, lo, hi = d["reg"], d["rlo"], d["rhi"]
                ax[1].errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=MK[d["seed"]], color=C[grp], ms=4, capsize=1.5, elinewidth=0.7,
                               mfc="white" if kind == "plane" else C[grp]); v.append(y)
            v = np.array(v); cv = v.std(ddof=1) / v.mean() * 100 if len(v) > 1 else np.nan
            ax[1].text(xs[(grp, kind)], -0.20, f"CV {cv:.0f}%", transform=ax[1].get_xaxis_transform(), ha="center", va="top", fontsize=7, color=C[grp])
    ax[1].set_xticks([0, 1, 2.4, 3.4]); ax[1].set_xticklabels(["plane\nEX = 3 Å", "region\n$G_{reg}$", "plane\nEX = 3 Å", "region\n$G_{reg}$"], fontsize=7)
    for xm, grp in ((0.5, "P"), (2.9, "D")):
        ax[1].text(xm, -0.31, NAME[grp], transform=ax[1].get_xaxis_transform(), ha="center", va="top", fontsize=8, color=C[grp])
    ax[1].set(ylabel="G (MW m$^{-2}$ K$^{-1}$)", xlim=(-0.6, 4.0)); ax[1].set_title("(b)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig("Fig7_plane_vs_region.png", dpi=300, bbox_inches="tight"); print("Fig7_plane_vs_region.png")
    for d in runs: print(f"{NAME[d['grp']]:14s} seed{d['seed']}: G(EX2..6) = {np.round(d['g'], 1)}  G_reg = {d['reg']:.1f} [{d['rlo']:.1f}, {d['rhi']:.1f}]")

cmd = sys.argv[1]
if cmd == "fig2": fig2(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
elif cmd == "fig3": fig3(sys.argv[2], sys.argv[3:])
elif cmd == "fig4": fig4(sys.argv[2], sys.argv[3])
elif cmd == "si3": si3(sys.argv[2])
elif cmd == "fig5": fig5(sys.argv[2], sys.argv[3])
elif cmd == "fig7": fig7(sys.argv[2], sys.argv[3])
elif cmd == "figE":                                   # figE --a log1 [log2 ..] --b log1 [log2 ..]
    i, j = sys.argv.index("--a"), sys.argv.index("--b")
    figE([("(a) $\\Delta t$ = 1 fs, 10 keV N-PKA", sys.argv[i + 1:j]), ("(b) $\\Delta t$ = 0.25 fs, same structures", sys.argv[j + 1:])])
