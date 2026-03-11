"""
Divida Liquida do Setor Publico — Governo Federal e Banco Central
Fonte: BCB/SGS via python-bcb

Series:
  2053 = Total        (u.m.c. milhoes)
  2063 = Interna      (u.m.c. milhoes)
  2073 = Externa      (u.m.c. milhoes)

u.m.c. = unidade monetaria corrente (equivalente a R$ para toda a serie)

Instalacao:  pip install python-bcb pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings
from bcb import sgs

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 1. COLETA
# ──────────────────────────────────────────────────────────────────────────────

print("Baixando series SGS...")
df = sgs.get(
    {"total": 2053, "interna": 2063, "externa": 2073},
    start="1995-01-01"
)

total   = df["total"]   / 1_000   # R$ bilhoes
interna = df["interna"] / 1_000
externa = df["externa"] / 1_000

print(f"Series prontas: {df.index[0].date()} a {df.index[-1].date()} — {len(df)} obs.")

residuo = (interna + externa - total).abs().max()
print(f"Residuo max (interna + externa - total): R$ {residuo:.2f} bi  [deve ser ~0]")


# ──────────────────────────────────────────────────────────────────────────────
# 2. PRESIDENCIAS
# ──────────────────────────────────────────────────────────────────────────────

presidencias_raw = [
    ("FHC I+II",     "1995-01-01", "2003-01-01", "#2980b9", "PSDB"),
    ("Lula I+II",    "2003-01-01", "2011-01-01", "#e74c3c", "PT"),
    ("Dilma I+II",   "2011-01-01", "2016-08-31", "#c0392b", "PT"),
    ("Temer",        "2016-08-31", "2019-01-01", "#27ae60", "PMDB"),
    ("Bolsonaro",    "2019-01-01", "2023-01-01", "#16a085", "PL"),
    ("Lula III",     "2023-01-01", "2027-01-01", "#e74c3c", "PT"),
]
presidencias = [
    {"nome": n, "inicio": pd.Timestamp(i), "fim": pd.Timestamp(f),
     "cor": c, "partido": p}
    for n, i, f, c, p in presidencias_raw
]


# ──────────────────────────────────────────────────────────────────────────────
# 3. TEMA
# ──────────────────────────────────────────────────────────────────────────────

BG     = "#0D1117"
GRID_C = "#2a2a3a"
TEXT_C = "#DDDDDD"
BLUE   = "#3A9EEA"
RED    = "#E85C5C"
PURPLE = "#9b59b6"
GOLD   = "#F5C518"


# ──────────────────────────────────────────────────────────────────────────────
# 4. HELPERS VISUAIS
# ──────────────────────────────────────────────────────────────────────────────

def aplicar_presidencias(axes_list, data_min, data_max, rotulos_em=None):
    for p in presidencias:
        ini = max(p["inicio"], data_min)
        fim = min(p["fim"], data_max)
        if ini >= fim:
            continue
        for ax in axes_list:
            ax.axvspan(ini, fim, alpha=0.09, color=p["cor"], zorder=0)
            ax.axvline(p["inicio"], color=p["cor"], linewidth=0.7,
                       linestyle="--", alpha=0.45, zorder=1)
        if rotulos_em is not None:
            centro = ini + (fim - ini) / 2
            rotulos_em.text(centro, 1.015, p["nome"],
                            transform=rotulos_em.get_xaxis_transform(),
                            ha="center", va="bottom",
                            fontsize=8, color=p["cor"], fontweight="bold")


def legenda_presidencias(fig, data_min, data_max):
    visiveis = [p for p in presidencias
                if p["inicio"] <= data_max and p["fim"] >= data_min]
    patches = [mpatches.Patch(color=p["cor"], alpha=0.75,
                              label=f"{p['nome']} ({p['partido']})")
               for p in visiveis]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT_C,
               fontsize=8, title="Presidencias", title_fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))


def estilo_eixos(axes_list):
    for ax in axes_list:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C, labelsize=9)
        ax.yaxis.label.set_color(TEXT_C)
        ax.xaxis.label.set_color(TEXT_C)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)


def anotar(ax, serie, ts_str, txt, deslocamento_y, data_min, data_max):
    ts = pd.Timestamp(ts_str)
    if not (data_min <= ts <= data_max):
        return
    pos = serie.index.get_indexer([ts], method="nearest")[0]
    y = float(serie.iloc[pos])
    ax.annotate(txt, xy=(ts, y), xytext=(ts, y + deslocamento_y),
                color=TEXT_C, fontsize=7.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEXT_C, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.25", fc="#1a1a2e",
                          ec=GRID_C, alpha=0.85))


# ──────────────────────────────────────────────────────────────────────────────
# 5. FIGURA
# ──────────────────────────────────────────────────────────────────────────────

data_min = total.index.min()
data_max = total.index.max()
d = total.max() * 0.07

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(18, 11), sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06}
)
fig.patch.set_facecolor(BG)
estilo_eixos([ax_top, ax_bot])
aplicar_presidencias([ax_top, ax_bot], data_min, data_max, rotulos_em=ax_top)

# painel superior: interna + externa
ax_top.fill_between(interna.index, interna.values, 0,
                    alpha=0.20, color=BLUE, zorder=2)
ax_top.fill_between(externa.index, externa.values, 0,
                    where=(externa.values >= 0),
                    alpha=0.20, color=RED, zorder=2)
ax_top.fill_between(externa.index, externa.values, 0,
                    where=(externa.values < 0),
                    alpha=0.20, color=PURPLE, zorder=2)

ax_top.plot(interna.index, interna.values,
            color=BLUE, linewidth=2.0, label="Interna", zorder=3)
ax_top.plot(externa.index, externa.values,
            color=RED, linewidth=2.0,
            label="Externa  (negativo = Brasil credor liquido)", zorder=3)

ax_top.axhline(0, color=TEXT_C, linewidth=0.8, linestyle="-", alpha=0.3)
ax_top.set_ylabel("R$ bilhoes correntes", fontsize=10)
ax_top.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f} bi"))
ax_top.grid(axis="y", color=GRID_C, linewidth=0.5)
ax_top.legend(facecolor="#1a1a2e", edgecolor=GRID_C,
              labelcolor=TEXT_C, fontsize=9, loc="upper left")
ax_top.set_title(
    "Divida Liquida do Setor Publico — Governo Federal e Banco Central (R$ bilhoes correntes)",
    color="white", fontsize=13, fontweight="bold", pad=26, loc="left"
)

# painel inferior: total
ax_bot.fill_between(total.index, total.values, 0,
                    alpha=0.22, color=GOLD, zorder=2)
ax_bot.plot(total.index, total.values,
            color=GOLD, linewidth=2.0, label="Total", zorder=3)
ax_bot.axhline(0, color=TEXT_C, linewidth=0.8, linestyle="-", alpha=0.3)
ax_bot.set_ylabel("R$ bilhoes correntes", fontsize=10)
ax_bot.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f} bi"))
ax_bot.grid(axis="y", color=GRID_C, linewidth=0.5)
ax_bot.legend(facecolor="#1a1a2e", edgecolor=GRID_C,
              labelcolor=TEXT_C, fontsize=9, loc="upper left")
ax_bot.set_xlabel("Ano", fontsize=10)

# anotacoes
anotar(ax_top, interna, "1999-01-01", "Crise cambial", d, data_min, data_max)
anotar(ax_top, interna, "2002-10-01", "Crise Lula",    d, data_min, data_max)
anotar(ax_top, total,   "2008-09-01", "Crise Global",  d, data_min, data_max)
anotar(ax_top, total,   "2015-01-01", "Recessao",      d, data_min, data_max)
anotar(ax_bot, total,   "2020-03-01", "COVID-19",      d, data_min, data_max)

legenda_presidencias(fig, data_min, data_max)
fig.text(0.99, 0.005,
         "Fonte: BCB — SGS 2053, 2063, 2073  |  Elaboracao propria",
         ha="right", va="bottom", color="#666666", fontsize=7)

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()