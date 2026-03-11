import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── Leitura e limpeza ────────────────────────────────────────────────────────
arquivo = 'C:\\repo\\ADM0416-admfin2\\anbimaRanking.xls'
raw = pd.read_excel(arquivo, sheet_name="PLcategoria", header=None)

# A linha 4 (índice 4) contém os cabeçalhos reais
header_row = raw.iloc[4].tolist()
header_row[0] = "Ordem"
header_row[1] = "Gestor"
cols = ["Ordem", "Gestor", "Renda Fixa", "Ações", "Multimercados",
        "Cambial", "Previdência", "ETF", "FIDC", "FIP", "FIAGRO", "FII",
        "Off-Shore", "Total"]

df = raw.iloc[5:].copy()
df.columns = raw.iloc[4].tolist()[:14]
df.columns = cols

# Manter apenas linhas com Ordem numérica (gestores de fato)
df = df[pd.to_numeric(df["Ordem"], errors="coerce").notna()].copy()
df["Ordem"] = df["Ordem"].astype(int)

categorias = ["Renda Fixa", "Ações", "Multimercados", "Cambial",
              "Previdência", "ETF", "FIDC", "FIP", "FIAGRO", "FII",
              "Off-Shore"]

for col in categorias + ["Total"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values("Total", ascending=False).reset_index(drop=True)
df["Gestor"] = df["Gestor"].str.strip()

top20 = df.head(20).copy()
top10 = df.head(10).copy()

# Paleta e helpers
CORES_CAT = {
    "Renda Fixa":    "#1f77b4",
    "Previdência":   "#ff7f0e",
    "Multimercados": "#2ca02c",
    "FIDC":          "#d62728",
    "FIAGRO":        "#9467bd",
    "FII":           "#8c564b",
    "ETF":           "#e377c2",
    "Ações":         "#7f7f7f",
    "FIP":           "#bcbd22",
    "Cambial":       "#17becf",
    "Off-Shore":     "#aec7e8",
}

def bilhoes(x, _):
    return f"R$ {x/1e6:.1f} tri" if x >= 1e6 else f"R$ {x/1e3:.0f} bi"

# ─── FIG 1: Top 20 PL Total — barras horizontais ─────────────────────────────
fig1, ax1 = plt.subplots(figsize=(13, 9))
fig1.patch.set_facecolor("#f5f5f5")
ax1.set_facecolor("#f5f5f5")

bars = ax1.barh(
    top20["Gestor"][::-1],
    top20["Total"][::-1] / 1e3,
    color="#1f77b4", edgecolor="white", linewidth=0.5
)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R$ {x:.0f} bi"))
ax1.set_xlabel("Patrimônio Líquido (R$ bilhões)", fontsize=11)
ax1.set_title("Top 20 Gestores — PL Total (Jan/2026)", fontsize=14, fontweight="bold", pad=12)
ax1.axvline(top20["Total"].mean() / 1e3, color="tomato", linestyle="--", linewidth=1.2,
            label=f"Média Top 20: R$ {top20['Total'].mean()/1e3:.0f} bi")
ax1.legend(fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)
for bar in bars:
    w = bar.get_width()
    ax1.text(w + 5, bar.get_y() + bar.get_height() / 2,
             f"{w/1e3:.0f} bi" if w < 1e6 else f"{w/1e6:.2f} tri",
             va="center", fontsize=7.5, color="#333")
plt.tight_layout()

# ─── FIG 2: Composição por categoria — stacked bar Top 10 ────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 7))
fig2.patch.set_facecolor("#f9f9f9")
ax2.set_facecolor("#f9f9f9")

bottom = np.zeros(len(top10))
gestores_abrev = top10["Gestor"].str[:20].tolist()

for cat in categorias:
    vals = top10[cat].fillna(0).values / 1e3
    ax2.bar(gestores_abrev, vals, bottom=bottom,
            label=cat, color=CORES_CAT[cat], edgecolor="white", linewidth=0.4)
    bottom += vals

ax2.set_ylabel("Patrimônio Líquido (R$ bilhões)", fontsize=11)
ax2.set_title("Composição do PL por Categoria — Top 10 Gestores (Jan/2026)",
              fontsize=13, fontweight="bold", pad=12)
ax2.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.7)
ax2.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.tight_layout()

# ─── FIG 3: Participação de mercado — pie com destaque ───────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 9))
fig3.patch.set_facecolor("#f5f5f5")
ax3.set_facecolor("#f5f5f5")

outros_total = df.iloc[10:]["Total"].sum()
labels_pie = top10["Gestor"].str[:18].tolist() + ["Demais Gestores"]
sizes_pie = top10["Total"].tolist() + [outros_total]
explode = [0.04] * 10 + [0.0]
cores_pie = plt.cm.tab20.colors[:11]

wedges, texts, autotexts = ax3.pie(
    sizes_pie, labels=None, autopct="%1.1f%%",
    startangle=140, explode=explode, colors=cores_pie,
    pctdistance=0.82, wedgeprops=dict(edgecolor="white", linewidth=0.8)
)
for at in autotexts:
    at.set_fontsize(8)

ax3.legend(wedges, labels_pie, loc="lower left", bbox_to_anchor=(-0.15, -0.05),
           fontsize=8.5, framealpha=0.8)
ax3.set_title("Participação no PL Total — Top 10 vs Demais (Jan/2026)",
              fontsize=13, fontweight="bold")
plt.tight_layout()

# ─── FIG 4: Heatmap de concentração por categoria (normalizado) ───────────────
fig4, ax4 = plt.subplots(figsize=(14, 8))
fig4.patch.set_facecolor("#fafafa")

heat_data = top20[categorias].fillna(0).copy()
# Normaliza coluna a coluna (% dentro de cada categoria)
heat_norm = heat_data.div(heat_data.sum(axis=0), axis=1) * 100

im = ax4.imshow(heat_norm.values, aspect="auto", cmap="YlOrRd")
ax4.set_xticks(range(len(categorias)))
ax4.set_xticklabels(categorias, rotation=40, ha="right", fontsize=9)
ax4.set_yticks(range(len(top20)))
ax4.set_yticklabels(top20["Gestor"].str[:22], fontsize=8)
ax4.set_title("Concentração por Categoria — % do PL de cada Categoria no Top 20 (Jan/2026)",
              fontsize=12, fontweight="bold", pad=12)
plt.colorbar(im, ax=ax4, label="% do PL da categoria")

for i in range(len(top20)):
    for j in range(len(categorias)):
        val = heat_norm.values[i, j]
        if val > 1:
            ax4.text(j, i, f"{val:.1f}%", ha="center", va="center",
                     fontsize=6.5, color="black" if val < 50 else "white")
plt.tight_layout()

# ─── FIG 5: Distribuição do PL Total — histograma + KDE ──────────────────────
fig5, ax5 = plt.subplots(figsize=(11, 6))
fig5.patch.set_facecolor("#f5f5f5")
ax5.set_facecolor("#f5f5f5")

pl_log = np.log10(df["Total"].dropna())
ax5.hist(pl_log, bins=40, color="#1f77b4", edgecolor="white", alpha=0.75,
         density=True, label="Histograma (log10)")

from scipy.stats import gaussian_kde
kde = gaussian_kde(pl_log)
xs = np.linspace(pl_log.min(), pl_log.max(), 300)
ax5.plot(xs, kde(xs), color="tomato", lw=2.2, label="KDE")

ax5.axvline(np.log10(df["Total"].median()), color="green", linestyle="--", lw=1.5,
            label=f"Mediana: R$ {df['Total'].median()/1e3:.1f} bi")
ax5.axvline(np.log10(df["Total"].mean()), color="orange", linestyle="--", lw=1.5,
            label=f"Média: R$ {df['Total'].mean()/1e3:.1f} bi")

ax5.set_xlabel("log₁₀(PL Total em R$ milhões)", fontsize=11)
ax5.set_ylabel("Densidade", fontsize=11)
ax5.set_title(f"Distribuição do PL Total — {len(df)} Gestores (Jan/2026)",
              fontsize=13, fontweight="bold")
ax5.legend(fontsize=9)
ax5.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

# ─── FIG 6: Scatter — Renda Fixa vs Multimercados, tamanho = Total ───────────
fig6, ax6 = plt.subplots(figsize=(11, 7))
fig6.patch.set_facecolor("#f5f5f5")
ax6.set_facecolor("#f5f5f5")

scatter_df = df[df["Renda Fixa"].notna() & df["Multimercados"].notna()].head(60)
sizes = (scatter_df["Total"] / scatter_df["Total"].max()) * 1200 + 20

sc = ax6.scatter(
    scatter_df["Renda Fixa"] / 1e3,
    scatter_df["Multimercados"] / 1e3,
    s=sizes, alpha=0.65,
    c=np.log10(scatter_df["Total"]), cmap="plasma",
    edgecolors="white", linewidths=0.5
)
plt.colorbar(sc, ax=ax6, label="log₁₀(PL Total)")

for _, row in scatter_df.head(10).iterrows():
    ax6.annotate(row["Gestor"][:14],
                 xy=(row["Renda Fixa"] / 1e3, row["Multimercados"] / 1e3),
                 fontsize=7, alpha=0.85,
                 xytext=(4, 4), textcoords="offset points")

ax6.set_xlabel("Renda Fixa (R$ bilhões)", fontsize=11)
ax6.set_ylabel("Multimercados (R$ bilhões)", fontsize=11)
ax6.set_title("Renda Fixa vs Multimercados — tamanho proporcional ao PL Total (Jan/2026)",
              fontsize=12, fontweight="bold")
ax6.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

plt.show()