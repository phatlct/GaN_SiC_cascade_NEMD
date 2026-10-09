#!/usr/bin/env python3
# Fig 1 (ban style rieng): (a) quy trinh ; (b) hinh hoc NEMD theo z. Khong can du lieu.
# Output: Fig1_schematic.png (600 dpi) + Fig1_schematic.pdf (vector, font nhung)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

# ---- bang mau chung cho ca bai (mau co nghia vat ly) ----
INK, MUTE = "#1F2A36", "#5B6573"
HOT, HOT_T = "#B23A2E", "#F3D6CF"          # nong: do gach
COLD, COLD_T = "#1F5A7A", "#D2E1EA"        # lanh: xanh thep
GAN_T, SIC_T = "#ECE3CF", "#E6E9EC"        # vat lieu: GaN cat nhat, SiC xam lanh
BAD, GOOD, GOOD_T = "#8C1C13", "#0B6E69", "#D5EBE8"
PLANE = "#C08400"                          # cua so fit tai mat phang: vang dat

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "font.size": 8.5, "pdf.fonttype": 42, "hatch.color": "white", "hatch.linewidth": 0.5,
    "axes.edgecolor": INK, "xtick.color": INK, "axes.labelcolor": INK, "text.color": INK})

fig = plt.figure(figsize=(7.2, 5.3))
axA = fig.add_axes([0.0, 0.50, 1.0, 0.49]); axA.set_xlim(0, 100); axA.set_ylim(0, 50); axA.axis("off")
axB = fig.add_axes([0.055, 0.095, 0.93, 0.37])

def head(x, y, w, h, n, title, accent):
    axA.add_patch(Rectangle((x, y + h - 6.2), w, 6.2, fc=accent, ec=INK, lw=0.8))
    axA.add_patch(Circle((x + 3.0, y + h - 3.1), 2.0, fc="white", ec="none"))
    axA.text(x + 3.0, y + h - 3.1, str(n), ha="center", va="center", fontsize=7.5, weight="bold", color=accent)
    axA.text(x + w / 2 + 1.6, y + h - 3.1, title, ha="center", va="center", fontsize=8.6, weight="bold", color="white")

def card(x, y, w, h, n, title, lines, accent, fs=7.8):
    axA.add_patch(Rectangle((x, y), w, h, fc="white", ec=INK, lw=0.8)); head(x, y, w, h, n, title, accent)
    axA.text(x + w / 2, y + h - 9.0, "\n".join(lines), ha="center", va="top", fontsize=fs, linespacing=1.45)

def arr(x0, x1, y=30):
    axA.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10, lw=1.1, color=INK))

axA.text(0.6, 49.5, "a", fontsize=11, weight="bold", va="top")
card(1.5, 17, 17, 28, 1, "Model", ["33,660 atoms", "61.7 × 53.4 Å in xy", "$L_z$ = 116.9 Å", "periodic x, y; fixed z", "Tersoff + LJ (UFF)"], INK)
card(21.5, 17, 17, 28, 2, "Cascade", ["Ga or N PKA", "1, 3, 5, 7, 10 keV", "300, 350, 400 K", "adaptive Δt"], HOT)
card(41.5, 17, 17, 28, 3, "Anneal", ["Nosé–Hoover", "T → 600 K → T", "100 ps", "Δt = 1 fs"], HOT)
arr(18.9, 21.1); arr(38.9, 41.1); arr(58.9, 61.1)
axA.add_patch(Rectangle((61.5, 2), 17.5, 43, fc="white", ec=INK, lw=0.8)); head(61.5, 2, 17.5, 43, 4, "NEMD", COLD)
axA.text(70.25, 37.3, "Langevin ±10 K\nτ = 0.1 ps", ha="center", va="top", fontsize=7.2, color=MUTE, linespacing=1.3)
for y0, lab, col, fill, lines in ((17.4, "original", BAD, "#F6E1DE", ["Δt = 1 fs", "200 + 400 ps", "165 runs"]),
                                   (2.6, "corrected", GOOD, GOOD_T, ["Δt = 0.25 fs", "50 + 200 ps", "3 + 3 runs"])):
    axA.add_patch(Rectangle((63.0, y0), 14.5, 13.8, fc=fill, ec=col, lw=1.1))
    axA.text(70.25, y0 + 12.4, lab, ha="center", va="top", fontsize=8.0, weight="bold", color=col)
    axA.text(70.25, y0 + 8.4, "\n".join(lines), ha="center", va="top", fontsize=7.4, linespacing=1.35)
arr(79.4, 81.6)
card(82.0, 2, 16.5, 43, 5, "Analysis", ["reservoir rates $J_h$, $J_c$", "imbalance $S = J_h + J_c$", "",
                                        "plane: Δ$T$ at $z_{\\rm int}$", "region: Δ$T_{\\rm reg}$", "",
                                        "NVE drift vs Δt", "cutoff-zone pairs"], INK, fs=7.4)

# ---------------- (b) hinh hoc NEMD ----------------
ax = axB; Z1, ZI = 114.95, 38.126
ax.set_xlim(-2, 117); ax.set_ylim(-0.6, 10.2); ax.set_yticks([])
for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
ax.set_xlabel("z (Å)", fontsize=9); ax.tick_params(labelsize=8.5, width=0.8)
ax.text(-2, 10.2, "b", fontsize=11, weight="bold", va="top")
Y0, H = 2.3, 3.4
ax.add_patch(Rectangle((0, Y0), ZI, H, fc=GAN_T, ec="none"))
ax.add_patch(Rectangle((ZI, Y0), Z1 - ZI, H, fc=SIC_T, ec="none"))
ax.add_patch(Rectangle((5.0, Y0), 20.0, H, fc=HOT_T, ec=HOT, lw=1.0, hatch="////"))
ax.add_patch(Rectangle((Z1 - 25.0, Y0), 20.0, H, fc=COLD_T, ec=COLD, lw=1.0, hatch="\\\\\\\\"))
for z0 in (0.0, Z1 - 3.0): ax.add_patch(Rectangle((z0, Y0), 3.0, H, fc=MUTE, ec="none"))
ax.add_patch(Rectangle((0, Y0), Z1, H, fc="none", ec=INK, lw=0.9))
ax.text(15.0, Y0 + H / 2, "hot\nT + 10 K", ha="center", va="center", fontsize=7.8, weight="bold", color=HOT, bbox=dict(fc=HOT_T, ec="none", pad=0.6))
ax.text(Z1 - 15.0, Y0 + H / 2, "cold\nT − 10 K", ha="center", va="center", fontsize=7.8, weight="bold", color=COLD, bbox=dict(fc=COLD_T, ec="none", pad=0.6))
ax.text(31.6, Y0 + H - 0.55, "GaN", ha="center", va="top", fontsize=8, weight="bold")
ax.text(64.0, Y0 + H / 2, "SiC", ha="center", va="center", fontsize=8, weight="bold")
for zc in (1.5, Z1 - 1.5): ax.text(zc, Y0 - 0.35, "fixed", ha="center", va="top", fontsize=6.8, color=MUTE)
ax.plot([ZI, ZI], [Y0 - 0.6, Y0 + H + 0.9], color=INK, lw=1.4)
ax.text(ZI + 0.8, Y0 - 0.45, "$z_{\\rm int}$", ha="left", va="top", fontsize=8.2)
rng = np.random.default_rng(7); zz = np.clip(29.2 + rng.normal(0, 2.3, 34), 25.8, 37.3)
ax.scatter(zz, 3.25 + rng.normal(0, 0.32, 34), s=9, facecolors="white", edgecolors=INK, lw=0.6, zorder=4)
ax.annotate("defect cluster", (29.2, 3.9), xytext=(16.5, 9.4), fontsize=7.6, ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkB=2))
yb = 6.75
for a, b in ((ZI - 15, ZI - 3), (ZI + 3, ZI + 15)): ax.plot([a, b], [yb, yb], color=PLANE, lw=2.6, solid_capstyle="butt")
ax.text(ZI + 16.2, yb, "plane fits → Δ$T$ at $z_{\\rm int}$", va="center", fontsize=7.8, color=PLANE, weight="bold")
yr, z1, z2 = 8.15, 27.16, ZI + 15
ax.annotate("", (z1, yr), (z2, yr), arrowprops=dict(arrowstyle="<|-|>", lw=1.1, color=GOOD, mutation_scale=8))
for z, lab in ((z1, "$z_1$"), (z2, "$z_2$")):
    ax.plot([z, z], [yr - 0.45, yr + 0.45], color=GOOD, lw=1.1); ax.text(z, yr + 0.6, lab, ha="center", va="bottom", fontsize=7.6, color=GOOD)
ax.text(z2 + 2.2, yr, "region → Δ$T_{\\rm reg}$", va="center", fontsize=7.8, color=GOOD, weight="bold")
ax.annotate("", (86, 0.85), (44, 0.85), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK, mutation_scale=10))
ax.text(65, 0.45, "heat flux $J$;  ideal: $J_h = -J_c$,  $S = 0$", ha="center", va="top", fontsize=7.6, color=MUTE)
fig.savefig("Fig1_schematic.png", dpi=600); fig.savefig("Fig1_schematic.pdf"); print("Fig1_schematic.png / .pdf")
