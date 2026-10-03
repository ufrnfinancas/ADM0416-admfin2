# -*- coding: utf-8 -*-
# Estudo de caso: BB Top Acoes Small Caps
# Antes da gestao (2005-2007), gestao (2007-2010) e pos-gestao (2010-2015)
# Script linear: rode secao por secao (celulas #%% no VS Code/Spyder)

#%% 0. PARAMETROS
import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

PROJETO = r"C:\repo\ADM0416-admfin2"
CNPJ_FUNDO = "05.100.234/0001-24"            # BB Top Acoes Small Caps (master)
CNPJ_FIC = "05.100.221/0001-55"              # BB Acoes Small Caps FIC (varejo)
INICIO_PRE = pd.Timestamp("2005-01-03")      # inicio da analise
DATA_ASSUNCAO = pd.Timestamp("2008-08-01")   # entrada na gestao: corte brusco de acoes (41 em jul/08 -> 24 em ago/08)
DATA_SAIDA_GESTAO = pd.Timestamp("2010-07-30")  # saida da gestao (jul/2010)
FIM_ANALISE = pd.Timestamp("2015-12-31")     # fim da analise
ANOS = range(2005, 2016)
ORDEM = ["antes", "gestao", "depois"]
PERIODOS = [("antes", INICIO_PRE, DATA_ASSUNCAO),
            ("gestao", DATA_ASSUNCAO, DATA_SAIDA_GESTAO),
            ("depois", DATA_SAIDA_GESTAO, FIM_ANALISE + pd.Timedelta(days=1))]
ARQ_BENCH = os.path.join(PROJETO, "smll.csv")          # opcional: data;fechamento (dd/mm/aaaa)
ARQ_PRECOS = os.path.join(PROJETO, "precos_acoes.csv") # opcional: data;TICKER1;TICKER2;...
ARQ_CDI = os.path.join(PROJETO, "cdi.csv")             # CDI limpo (gerado na secao 3)
ARQ_CDI_SGS = os.path.join(PROJETO, "STP-20261003082454790.csv")  # exportacao do SGS, serie 12
PESO_MAX = 0.10                              # teto por acao na otimizacao
DU = 252
B = 5000                                     # replicacoes do bootstrap
BLOCO = 10                                   # tamanho do bloco (dias) no bootstrap
JANELA_OOS = 126                             # dias fora da amostra para testar a otimizacao
PASTA = os.path.join(PROJETO, "saidas")
os.makedirs(PASTA, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})
print("Saidas em:", PASTA)

#%% 1. CADASTRO CVM: grupo de pares (fundos de acoes small/mid caps, inclusive cancelados)
cad = pd.read_csv("https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv",
                  sep=";", encoding="latin-1", dtype=str)
print(cad.loc[cad["CNPJ_FUNDO"].isin([CNPJ_FUNDO, CNPJ_FIC]),
              ["CNPJ_FUNDO", "DENOM_SOCIAL", "DT_REG", "SIT"]].to_string())

nome = cad["DENOM_SOCIAL"].str.upper().fillna("")
classe = cad["CLASSE"].fillna("")
anbima = cad["CLASSE_ANBIMA"].fillna("").str.upper()
crit_nome = nome.str.contains("SMALL|SMLL|SMID|MID CAP|MIDCAP|MID-CAP")
crit_anbima = anbima.str.contains("SMALL")
crit_classe = classe.str.contains("Ações|Acoes") | (classe == "")
crit_exclusivo = cad["FUNDO_EXCLUSIVO"].fillna("N") == "S"
crit_fic = cad["FUNDO_COTAS"].fillna("N") == "S"
sel = (crit_nome | crit_anbima) & crit_classe & ~crit_exclusivo & ~crit_fic
print("por nome:", crit_nome.sum(), "| por classe ANBIMA:", crit_anbima.sum(),
      "| selecionados (acoes, nao exclusivos, nao FIC):", sel.sum())
print(anbima[crit_anbima].value_counts().to_string())

pares = sorted(set(cad.loc[sel, "CNPJ_FUNDO"]) | {CNPJ_FUNDO, CNPJ_FIC})
print(f"Fundos no grupo de comparacao (incluindo o FIC do BB, que sai na secao 6): {len(pares)}")

#%% 2. COTAS DIARIAS (informe diario historico da CVM), com cache local
arq_cotas = os.path.join(PASTA, f"cotas_pares_v2_{min(ANOS)}_{max(ANOS)}.csv")
if os.path.exists(arq_cotas):
    cotas = pd.read_csv(arq_cotas, index_col=0, parse_dates=True)
else:
    partes = []
    for ano in ANOS:
        url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/HIST/inf_diario_fi_{ano}.zip"
        print("baixando", url)
        r = requests.get(url, timeout=600)
        if r.status_code != 200:
            print("  falhou:", r.status_code)
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for arq in z.namelist():
            df = pd.read_csv(z.open(arq), sep=";", encoding="latin-1",
                             usecols=["CNPJ_FUNDO", "DT_COMPTC", "VL_QUOTA"])
            partes.append(df[df["CNPJ_FUNDO"].isin(pares)])
    inf = pd.concat(partes)
    inf["DT_COMPTC"] = pd.to_datetime(inf["DT_COMPTC"])
    cotas = inf.pivot_table(index="DT_COMPTC", columns="CNPJ_FUNDO",
                            values="VL_QUOTA").sort_index()
    cotas.to_csv(arq_cotas)

ret = cotas.pct_change(fill_method=None)
ret = ret.where(ret.abs() < 0.5)             # remove erros grosseiros de cota

print(cotas[[CNPJ_FUNDO, CNPJ_FIC]].dropna(how="all").head())
print(cotas[[CNPJ_FUNDO, CNPJ_FIC]].dropna(how="all").tail())
print(cotas[[CNPJ_FUNDO, CNPJ_FIC]].notna().sum())

#%% 3. CDI (arquivo do SGS/BCB, serie 12) E BENCHMARK
if os.path.exists(ARQ_CDI):
    cdi = pd.read_csv(ARQ_CDI, sep=";", parse_dates=["data"], dayfirst=True,
                      index_col="data")["valor"].sort_index()
    print("CDI lido de", ARQ_CDI)
elif os.path.exists(ARQ_CDI_SGS):
    cdi = pd.read_csv(ARQ_CDI_SGS, sep=";", decimal=",", encoding="latin-1",
                      skipfooter=1, engine="python")
    cdi.columns = ["data", "valor"]
    cdi["data"] = pd.to_datetime(cdi["data"], dayfirst=True)
    cdi = cdi.set_index("data")["valor"].astype(float).sort_index() / 100
    cdi.rename("valor").to_csv(ARQ_CDI, sep=";", date_format="%d/%m/%Y", index_label="data")
    print("CDI lido do arquivo do SGS e salvo em", ARQ_CDI)
else:
    raise FileNotFoundError("CDI nao encontrado: exporte a serie 12 pelo SGS e ajuste ARQ_CDI_SGS")
print(f"CDI: {len(cdi)} dias, de {cdi.index.min().date()} a {cdi.index.max().date()}")
print(((1 + cdi).groupby(cdi.index.year).prod() - 1).round(4).to_string())

if os.path.exists(ARQ_BENCH):
    bench = pd.read_csv(ARQ_BENCH, sep=";", parse_dates=["data"], dayfirst=True,
                        index_col="data")["fechamento"].sort_index()
    NOME_BENCH = "SMLL"
else:
    import yfinance as yf
    bench = yf.download("^BVSP", start=INICIO_PRE, end=FIM_ANALISE + pd.Timedelta(days=5),
                        auto_adjust=True, progress=False)["Close"].squeeze()
    NOME_BENCH = "Ibovespa"
    print("AVISO: usando Ibovespa como benchmark; para small caps prefira o SMLL.")

base = pd.concat([ret[CNPJ_FUNDO].rename("fundo"),
                  ret[CNPJ_FIC].rename("fic"),
                  bench.pct_change(fill_method=None).rename("bench"),
                  cdi.rename("cdi")], axis=1)
base = base.dropna(subset=["fundo", "bench", "cdi"])
base = base[(base.index >= INICIO_PRE) & (base.index <= FIM_ANALISE)]
base["ativo"] = base["fundo"] - base["bench"]
base["periodo"] = np.select([base.index < DATA_ASSUNCAO, base.index < DATA_SAIDA_GESTAO],
                            ["antes", "gestao"], "depois")
print("Benchmark:", NOME_BENCH)
print(base.groupby("periodo").size().reindex(ORDEM))

#%% 3B. DATACAO DA ENTRADA: mudanca mensal no risco relativo do fundo (2007-2009)
jan = base[(base.index >= "2007-01-01") & (base.index < "2010-01-01")]
mes = jan.index.to_period("M")
tab_mes = pd.DataFrame({
    "vol_fundo": jan["fundo"].groupby(mes).std() * np.sqrt(DU),
    "vol_bench": jan["bench"].groupby(mes).std() * np.sqrt(DU),
    "tracking_error": jan["ativo"].groupby(mes).std() * np.sqrt(DU),
    "dias": jan["fundo"].groupby(mes).size(),
})
tab_mes["vol_relativa"] = tab_mes["vol_fundo"] / tab_mes["vol_bench"]
tab_mes["correlacao"] = [jan.loc[mes == m, "fundo"].corr(jan.loc[mes == m, "bench"])
                         for m in tab_mes.index]
print(tab_mes.round(3).to_string())
tab_mes.to_csv(os.path.join(PASTA, "tab_datacao_mensal.csv"))

movel = jan[["fundo", "bench", "ativo"]].rolling(21).std() * np.sqrt(DU)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(movel.index, movel["fundo"] / movel["bench"], color="#1f3b73", label="Vol. relativa 21d")
ax.plot(movel.index, movel["ativo"], color="firebrick", lw=0.9, label="Tracking error 21d (a.a.)")
ax.axvline(DATA_ASSUNCAO, color="gray", ls="--")
ax.set_title("Mudanca no risco relativo do fundo, 2007-2009")
ax.legend(frameon=False)
fig.savefig(os.path.join(PASTA, "fig_datacao.pdf"))
plt.close(fig)
print("Grafico salvo em", os.path.join(PASTA, "fig_datacao.pdf"))

#%% 4. METRICAS POR PERIODO
linhas = []
for per, g in base.groupby("periodo"):
    exc = g["fundo"] - g["cdi"]
    exc_b = g["bench"] - g["cdi"]
    exc_f = (g["fic"] - g["cdi"]).dropna()
    ativo = g["fundo"] - g["bench"]
    acum = (1 + g["fundo"]).cumprod()
    linhas.append({
        "periodo": per,
        "inicio": g.index.min().date(), "fim": g.index.max().date(), "dias": len(g),
        "retorno_anual": (1 + g["fundo"]).prod() ** (DU / len(g)) - 1,
        "retorno_anual_bench": (1 + g["bench"]).prod() ** (DU / len(g)) - 1,
        "retorno_anual_cdi": (1 + g["cdi"]).prod() ** (DU / len(g)) - 1,
        "vol_anual": g["fundo"].std() * np.sqrt(DU),
        "vol_anual_bench": g["bench"].std() * np.sqrt(DU),
        "vol_relativa": g["fundo"].std() / g["bench"].std(),
        "sharpe": exc.mean() / exc.std() * np.sqrt(DU),
        "sharpe_bench": exc_b.mean() / exc_b.std() * np.sqrt(DU),
        "sharpe_fic": exc_f.mean() / exc_f.std() * np.sqrt(DU) if len(exc_f) > 20 else np.nan,
        "beta": np.cov(g["fundo"], g["bench"])[0, 1] / g["bench"].var(),
        "tracking_error": ativo.std() * np.sqrt(DU),
        "information_ratio": ativo.mean() / ativo.std() * np.sqrt(DU),
        "max_drawdown": (acum / acum.cummax() - 1).min(),
    })
tab = pd.DataFrame(linhas).set_index("periodo").reindex(ORDEM).T
print(tab.to_string())
tab.to_csv(os.path.join(PASTA, "tab_periodos.csv"))

#%% 5. BOOTSTRAP EM BLOCOS: gestao x antes e gestao x depois (Sharpe e dif-em-dif vs benchmark)
rng = np.random.default_rng(42)
dados = {}
for per, g in base.groupby("periodo"):
    dados[per] = np.column_stack([g["fundo"] - g["cdi"], g["bench"] - g["cdi"]])

comparacoes = [("gestao", "antes"), ("gestao", "depois")]
boot_dif = {c: np.empty(B) for c in comparacoes}
boot_did = {c: np.empty(B) for c in comparacoes}
for it in range(B):
    sh = {}
    for per, x in dados.items():
        n = len(x)
        ini = rng.integers(0, n - BLOCO, size=n // BLOCO + 1)
        idx = (ini[:, None] + np.arange(BLOCO)).ravel()[:n]
        s = x[idx]
        sh[per] = s.mean(axis=0) / s.std(axis=0, ddof=1) * np.sqrt(DU)
    for a, c in comparacoes:
        boot_dif[(a, c)][it] = sh[a][0] - sh[c][0]
        boot_did[(a, c)][it] = (sh[a][0] - sh[c][0]) - (sh[a][1] - sh[c][1])

linhas = []
for a, c in comparacoes:
    for rotulo, arr in (("Sharpe fundo", boot_dif[(a, c)]),
                        ("Sharpe fundo menos Sharpe bench", boot_did[(a, c)])):
        linhas.append({"comparacao": f"{a} - {c}", "estatistica": rotulo,
                       "media": arr.mean(),
                       "ic95_inf": np.percentile(arr, 2.5),
                       "ic95_sup": np.percentile(arr, 97.5),
                       "p_unilateral": (arr <= 0).mean()})
boot = pd.DataFrame(linhas)
print(boot.to_string(index=False))
boot.to_csv(os.path.join(PASTA, "tab_bootstrap.csv"), index=False)

#%% 5B. REDUCAO DE RISCO: decomposicao (beta x risco especifico) e bootstrap
estat = ["vol_relativa", "vol_especifica", "especifica_relativa", "correlacao"]
pontual = {}
for per, x in dados.items():
    f, m = x[:, 0], x[:, 1]
    beta = np.cov(f, m)[0, 1] / m.var(ddof=1)
    resid = f - beta * m
    pontual[per] = [f.std(ddof=1) / m.std(ddof=1),
                    resid.std(ddof=1) * np.sqrt(DU),
                    resid.std(ddof=1) / m.std(ddof=1),
                    np.corrcoef(f, m)[0, 1]]
tab_risco = pd.DataFrame(pontual, index=estat)[ORDEM]
print(tab_risco.round(3).to_string())
tab_risco.to_csv(os.path.join(PASTA, "tab_risco_periodos.csv"))

rng = np.random.default_rng(7)
boot_risco = {c: np.empty((B, len(estat))) for c in comparacoes}
for it in range(B):
    r = {}
    for per, x in dados.items():
        n = len(x)
        ini = rng.integers(0, n - BLOCO, size=n // BLOCO + 1)
        idx = (ini[:, None] + np.arange(BLOCO)).ravel()[:n]
        f, m = x[idx, 0], x[idx, 1]
        beta = np.cov(f, m)[0, 1] / m.var(ddof=1)
        resid = f - beta * m
        r[per] = np.array([f.std(ddof=1) / m.std(ddof=1),
                           resid.std(ddof=1) * np.sqrt(DU),
                           resid.std(ddof=1) / m.std(ddof=1),
                           np.corrcoef(f, m)[0, 1]])
    for a, c in comparacoes:
        boot_risco[(a, c)][it] = r[a] - r[c]

linhas = []
for a, c in comparacoes:
    for k, nome_e in enumerate(estat):
        arr = boot_risco[(a, c)][:, k]
        esperado = "aumento" if nome_e == "correlacao" else "reducao"
        linhas.append({"comparacao": f"{a} - {c}", "estatistica": nome_e,
                       "diferenca": tab_risco.loc[nome_e, a] - tab_risco.loc[nome_e, c],
                       "ic95_inf": np.percentile(arr, 2.5),
                       "ic95_sup": np.percentile(arr, 97.5),
                       "hipotese": esperado,
                       "p_unilateral": (arr <= 0).mean() if esperado == "aumento" else (arr >= 0).mean()})
boot_r = pd.DataFrame(linhas)
print(boot_r.round(3).to_string(index=False))
boot_r.to_csv(os.path.join(PASTA, "tab_bootstrap_risco.csv"), index=False)

#%% 5C. ROBUSTEZ: benchmark de small caps (ETF SMAL11, disponivel a partir de 28/11/2008)
# retornos diarios e semanais: o semanal reduz o efeito de precos defasados de um ETF pouco liquido
ARQ_SMAL11 = os.path.join(PROJETO, "small11.csv")
smal = pd.read_csv(ARQ_SMAL11, sep=";", parse_dates=["data"], dayfirst=True,
                   index_col="data")["fechamento"].sort_index()
rob_d = pd.concat([base[["fundo", "bench", "cdi"]],
                   smal.pct_change(fill_method=None).rename("smal")], axis=1)
rob_d = rob_d.dropna(subset=["fundo", "bench", "cdi", "smal"])
rob_s = (1 + rob_d).resample("W-FRI").prod() - 1
rob_s = rob_s[(1 + rob_d).resample("W-FRI").size() > 0]
linhas = []
for freq, dfq, fator in (("diaria", rob_d, DU), ("semanal", rob_s, 52)):
    dfq = dfq.copy()
    dfq["periodo"] = np.select([dfq.index < DATA_ASSUNCAO, dfq.index < DATA_SAIDA_GESTAO],
                               ["antes", "gestao"], "depois")
    for per in ["gestao", "depois"]:
        g = dfq[dfq["periodo"] == per]
        f = (g["fundo"] - g["cdi"]).values
        s_ = (g["smal"] - g["cdi"]).values
        m = (g["bench"] - g["cdi"]).values
        beta_s = np.cov(f, s_)[0, 1] / s_.var(ddof=1)
        resid_s = f - beta_s * s_
        ativo_s = (g["fundo"] - g["smal"]).values
        linhas.append({
            "frequencia": freq, "periodo": per,
            "inicio": g.index.min().date(), "fim": g.index.max().date(), "obs": len(g),
            "retorno_anual_fundo": (1 + g["fundo"]).prod() ** (fator / len(g)) - 1,
            "retorno_anual_smal11": (1 + g["smal"]).prod() ** (fator / len(g)) - 1,
            "sharpe_fundo": f.mean() / f.std(ddof=1) * np.sqrt(fator),
            "sharpe_smal11": s_.mean() / s_.std(ddof=1) * np.sqrt(fator),
            "beta_smal11": beta_s,
            "vol_relativa_smal11": f.std(ddof=1) / s_.std(ddof=1),
            "especifica_relativa_smal11": resid_s.std(ddof=1) / s_.std(ddof=1),
            "corr_fundo_smal11": np.corrcoef(f, s_)[0, 1],
            "corr_fundo_ibov": np.corrcoef(f, m)[0, 1],
            "corr_smal11_ibov": np.corrcoef(s_, m)[0, 1],
            "information_ratio_smal11": ativo_s.mean() / ativo_s.std(ddof=1) * np.sqrt(fator),
        })
tab_rob = pd.DataFrame(linhas).set_index(["frequencia", "periodo"]).T
print(tab_rob.round(3).to_string())
tab_rob.to_csv(os.path.join(PASTA, "tab_robustez_smal11.csv"))

#%% 6. COMPARACAO COM PARES: posicao do fundo em cada periodo
COBERTURA_MIN = 0.8                          # fracao minima de dias com cota no periodo
ret_p = ret.drop(columns=[CNPJ_FIC], errors="ignore").join(cdi.rename("cdi"), how="inner")
res = []
for per, ini, fim in PERIODOS:
    jan = ret_p[(ret_p.index >= ini) & (ret_p.index < fim)]
    exc = jan.drop(columns="cdi").sub(jan["cdi"], axis=0)
    validos = exc.columns[exc.notna().mean() >= COBERTURA_MIN]
    res.append(pd.DataFrame({
        "periodo": per,
        "sharpe": exc[validos].mean() / exc[validos].std() * np.sqrt(DU),
        "vol": jan[validos].std() * np.sqrt(DU),
        "retorno_anual": (1 + jan[validos]).prod() ** (DU / jan[validos].notna().sum()) - 1,
    }))
pares_df = pd.concat(res)
pares_df["n_fundos"] = pares_df.groupby("periodo")["sharpe"].transform("size")
pares_df["posicao_sharpe"] = pares_df.groupby("periodo")["sharpe"].rank(ascending=False)
pares_df["pct_sharpe"] = pares_df.groupby("periodo")["sharpe"].rank(pct=True)
pares_df["pct_vol"] = pares_df.groupby("periodo")["vol"].rank(pct=True)
print(pares_df.loc[pares_df.index == CNPJ_FUNDO].round(3).to_string())
print(pares_df.groupby("periodo")[["sharpe", "vol", "retorno_anual"]].median().reindex(ORDEM).round(3))
print(pares_df.groupby("periodo").size().reindex(ORDEM).rename("n_fundos"))
pares_df.to_csv(os.path.join(PASTA, "tab_pares.csv"))

#%% 7. COMPOSICAO DA CARTEIRA (CDA/CVM, bloco 4 = acoes), com cache local
arq_cda = os.path.join(PASTA, f"cda_fundo_{min(ANOS)}_{max(ANOS)}.csv")
pesos = None
if os.path.exists(arq_cda):
    cda = pd.read_csv(arq_cda, dtype=str)
    print("CDA lido do cache", arq_cda)
else:
    partes = []
    for ano in ANOS:
        r = requests.get(f"https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/HIST/cda_fi_{ano}.zip", timeout=600)
        conteudos = [r.content] if r.status_code == 200 else []
        if not conteudos:
            print(f"{ano}: sem arquivo HIST, tentando arquivos mensais")
            for mes in range(1, 13):
                r = requests.get(f"https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_{ano}{mes:02d}.zip",
                                 timeout=600)
                if r.status_code == 200:
                    conteudos.append(r.content)
        for conteudo in conteudos:
            z = zipfile.ZipFile(io.BytesIO(conteudo))
            for arq in z.namelist():
                if "BLC_4" in arq:
                    df = pd.read_csv(z.open(arq), sep=";", encoding="latin-1", dtype=str)
                    partes.append(df[df["CNPJ_FUNDO"] == CNPJ_FUNDO])
        print(ano, "ok" if conteudos else "sem dados")
    cda = pd.concat(partes) if partes else pd.DataFrame()
    cda.to_csv(arq_cda, index=False)

if len(cda) > 0:
    cda["VL_MERC_POS_FINAL"] = pd.to_numeric(cda["VL_MERC_POS_FINAL"], errors="coerce")
    cda["DT_COMPTC"] = pd.to_datetime(cda["DT_COMPTC"])
    cda = cda[cda["VL_MERC_POS_FINAL"] > 0]
    pesos = cda.groupby(["DT_COMPTC", "CD_ATIVO"])["VL_MERC_POS_FINAL"].sum()
    pesos = pesos / pesos.groupby(level=0).transform("sum")
    comp = pd.DataFrame({
        "n_acoes": pesos.groupby(level=0).size(),
        "n_efetivo": 1 / (pesos ** 2).groupby(level=0).sum(),
        "peso_top10": pesos.groupby(level=0).apply(lambda s: s.nlargest(10).sum()),
    })
    comp["periodo"] = np.select([comp.index < DATA_ASSUNCAO, comp.index < DATA_SAIDA_GESTAO],
                                ["antes", "gestao"], "depois")
    print(comp.loc["2008-01-01":"2008-12-31"].round(3).to_string())
    print(comp.groupby("periodo")[["n_acoes", "n_efetivo", "peso_top10"]].mean().reindex(ORDEM).round(2))
    comp.to_csv(os.path.join(PASTA, "tab_composicao.csv"))
else:
    print("CDA sem registros para o fundo. Alternativa: Economatica ou relatorios internos.")

#%% 8. GRAFICOS E DRAWDOWN
# drawdown maximo por periodo: fundo x benchmark
linhas = []
for per, g in base.groupby("periodo"):
    acum_f = (1 + g["fundo"]).cumprod()
    acum_b = (1 + g["bench"]).cumprod()
    dd_f = acum_f / acum_f.cummax() - 1
    dd_b = acum_b / acum_b.cummax() - 1
    linhas.append({"periodo": per,
                   "dd_max_fundo": dd_f.min(), "data_fundo": dd_f.idxmin().date(),
                   "dd_max_bench": dd_b.min(), "data_bench": dd_b.idxmin().date()})
tab_dd = pd.DataFrame(linhas).set_index("periodo").reindex(ORDEM)
print(tab_dd.round(3).to_string())
tab_dd.to_csv(os.path.join(PASTA, "tab_drawdown.csv"))

acum = (1 + base[["fundo", "fic", "bench"]].fillna(0)).cumprod()
fig, ax = plt.subplots(figsize=(9, 4))
ax.axvspan(DATA_ASSUNCAO, DATA_SAIDA_GESTAO, color="gray", alpha=0.12, label="Periodo da gestao")
ax.plot(acum.index, acum["fundo"], label="BB Top Acoes Small Caps", color="#1f3b73")
ax.plot(acum.index, acum["fic"], label="BB Acoes Small Caps FIC", color="#1f3b73", ls=":", lw=0.9)
ax.plot(acum.index, acum["bench"], label=NOME_BENCH, color="#c9a227")
ax.set_ylabel("Valor acumulado (base 1)")
ax.legend(frameon=False)
fig.savefig(os.path.join(PASTA, "fig_acumulado.pdf"))
plt.close(fig)

dd_serie = pd.DataFrame({"Fundo": acum["fundo"] / acum["fundo"].cummax() - 1,
                         NOME_BENCH: acum["bench"] / acum["bench"].cummax() - 1})
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axvspan(DATA_ASSUNCAO, DATA_SAIDA_GESTAO, color="gray", alpha=0.12)
ax.fill_between(dd_serie.index, dd_serie["Fundo"], 0, color="#1f3b73", alpha=0.35, label="Fundo")
ax.plot(dd_serie.index, dd_serie[NOME_BENCH], color="#c9a227", lw=1, label=NOME_BENCH)
ax.set_ylabel("Drawdown desde o pico")
ax.legend(frameon=False, loc="lower left")
fig.savefig(os.path.join(PASTA, "fig_drawdown.pdf"))
plt.close(fig)

vol_movel = base[["fundo", "bench"]].rolling(63).std() * np.sqrt(DU)
fig, ax = plt.subplots(figsize=(9, 4))
ax.axvspan(DATA_ASSUNCAO, DATA_SAIDA_GESTAO, color="gray", alpha=0.12)
ax.plot(vol_movel.index, vol_movel["fundo"], label="Fundo", color="#1f3b73")
ax.plot(vol_movel.index, vol_movel["bench"], label=NOME_BENCH, color="#c9a227")
ax.set_ylabel("Volatilidade movel 63d (a.a.)")
ax.legend(frameon=False, loc="upper left")
ax2 = ax.twinx()
ax2.plot(vol_movel.index, vol_movel["fundo"] / vol_movel["bench"], color="firebrick", lw=0.8)
ax2.set_ylabel("Vol. fundo / vol. benchmark", color="firebrick")
fig.savefig(os.path.join(PASTA, "fig_vol_movel.pdf"))
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (a, c) in zip(axs, comparacoes):
    ax.hist(boot_did[(a, c)], bins=60, color="#1f3b73", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_title(f"{a} vs {c}")
    ax.set_xlabel("Delta Sharpe fundo menos Delta Sharpe bench")
axs[0].set_ylabel("Frequencia (bootstrap)")
fig.savefig(os.path.join(PASTA, "fig_bootstrap.pdf"))
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, per in zip(axs, ORDEM):
    d = pares_df[pares_df["periodo"] == per]
    ax.scatter(d["vol"], d["sharpe"], color="lightgray", label="Pares small caps")
    if CNPJ_FUNDO in d.index:
        ax.scatter(d.loc[CNPJ_FUNDO, "vol"], d.loc[CNPJ_FUNDO, "sharpe"],
                   color="#1f3b73", s=80, label="BB Top Small Caps")
    ax.set_title(f"{per.capitalize()} ({len(d)} fundos)")
    ax.set_xlabel("Volatilidade anual")
axs[0].set_ylabel("Sharpe")
axs[0].legend(frameon=False)
fig.savefig(os.path.join(PASTA, "fig_pares.pdf"))
plt.close(fig)

if pesos is not None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvspan(DATA_ASSUNCAO, DATA_SAIDA_GESTAO, color="gray", alpha=0.12)
    ax.step(comp.index, comp["n_acoes"], where="post", label="Numero de acoes", color="#1f3b73")
    ax.step(comp.index, comp["n_efetivo"], where="post", label="Numero efetivo (1/soma w2)", color="#c9a227")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(PASTA, "fig_composicao.pdf"))
    plt.close(fig)

print("Graficos salvos em", PASTA)

#%% 9. RECONSTRUCAO DA OTIMIZACAO MEDIA-VARIANCIA NA ASSUNCAO + TESTE FORA DA AMOSTRA
if os.path.exists(ARQ_PRECOS):
    precos = pd.read_csv(ARQ_PRECOS, sep=";", parse_dates=["data"], dayfirst=True,
                         index_col="data").sort_index()
    rets = precos.pct_change()
    est = rets[rets.index < DATA_ASSUNCAO].tail(DU)
    est = est.loc[:, est.notna().mean() > 0.9].fillna(0)
    ativos = est.columns
    n = len(ativos)
    mu = est.mean().values * DU
    S = LedoitWolf().fit(est.values).covariance_ * DU
    rf = (1 + cdi[cdi.index < DATA_ASSUNCAO].tail(DU)).prod() - 1

    limites = [(0, PESO_MAX)] * n
    soma1 = {"type": "eq", "fun": lambda w: w.sum() - 1}
    w0 = np.repeat(1 / n, n)
    w_mv = minimize(lambda w: w @ S @ w, w0, bounds=limites,
                    constraints=[soma1], method="SLSQP").x
    w_ms = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ S @ w), w0, bounds=limites,
                    constraints=[soma1], method="SLSQP").x

    fronteira = []
    for alvo in np.linspace(w_mv @ mu, np.sort(mu)[-int(1 / PESO_MAX):].mean(), 40):
        r = minimize(lambda w: w @ S @ w, w0, bounds=limites, method="SLSQP",
                     constraints=[soma1, {"type": "eq", "fun": lambda w, a=alvo: w @ mu - a}])
        if r.success:
            fronteira.append((np.sqrt(r.x @ S @ r.x), alvo))
    fronteira = np.array(fronteira)

    carteiras = {"Minima variancia": w_mv, "Maximo Sharpe": w_ms, "Pesos iguais": w0}
    if pesos is not None:
        data_cda = pesos.index.get_level_values(0)
        ult = data_cda[data_cda < DATA_ASSUNCAO].max()
        w_h = pesos.loc[ult].reindex(ativos).fillna(0).values
        if w_h.sum() > 0:
            carteiras["Carteira herdada"] = w_h / w_h.sum()
            print(f"Carteira herdada: CDA de {ult.date()}, cobertura de {w_h.sum():.1%} do bloco de acoes")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fronteira[:, 0], fronteira[:, 1], color="#1f3b73", label="Fronteira eficiente")
    ax.scatter(np.sqrt(np.diag(S)), mu, color="lightgray", s=12, label="Acoes individuais")
    for nome_c, w in carteiras.items():
        ax.scatter(np.sqrt(w @ S @ w), w @ mu, s=70, label=nome_c)
    ax.set_xlabel("Volatilidade anual (ex-ante)")
    ax.set_ylabel("Retorno esperado anual (ex-ante)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(PASTA, "fig_fronteira.pdf"))

    oos = rets[rets.index >= DATA_ASSUNCAO].head(JANELA_OOS)[ativos].fillna(0)
    cdi_oos = cdi.reindex(oos.index).fillna(0)
    linhas = []
    for nome_c, w in carteiras.items():
        r = oos.values @ w
        e = r - cdi_oos.values
        linhas.append({"carteira": nome_c,
                       "n_acoes_peso>0,5%": int((w > 0.005).sum()),
                       "vol_ex_ante": np.sqrt(w @ S @ w),
                       "vol_realizada": r.std() * np.sqrt(DU),
                       "sharpe_realizado": e.mean() / e.std() * np.sqrt(DU)})
    oos_tab = pd.DataFrame(linhas).set_index("carteira")
    print(oos_tab.to_string())
    oos_tab.to_csv(os.path.join(PASTA, "tab_otimizacao_oos.csv"))
else:
    print(f"{ARQ_PRECOS} nao encontrado: secao 9 (fronteira e teste fora da amostra) ignorada.")

print("Arquivos gerados em:", os.path.abspath(PASTA))