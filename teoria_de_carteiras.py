# %% [markdown]
# Teoria de Carteiras: código passo a passo para a aula
# Dependências: pip install yfinance requests numpy pandas matplotlib scipy scikit-learn

# %% 1. Dados
# Baixamos preços ajustados do Ibovespa e de alguns ativos. PETR4 será "a ação" dos slides.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True

inicio = "2021-10-01"
fim = "2026-09-30"

ticker_acao = "PETR4.SA"
tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "ABEV3.SA", "RENT3.SA", "BBAS3.SA", "SUZB3.SA"]

dados = yf.download(["^BVSP"] + tickers, start=inicio, end=fim, auto_adjust=True, progress=False)["Close"]
dados = dados.dropna()

ibov = dados["^BVSP"]
acao = dados[ticker_acao]
precos = dados[tickers]
print(dados.tail())

# %% 2. Retorno discreto e retorno contínuo (slide 1)
# Retorno discreto: P_t / P_{t-1} - 1. Retorno contínuo: ln(P_t) - ln(P_{t-1}).
ret_disc = acao / acao.shift(1) - 1
ret_cont = np.log(acao) - np.log(acao.shift(1))

comparacao = pd.DataFrame({"discreto": ret_disc, "continuo": ret_cont}).dropna()
print(comparacao.head(10))
print("Diferença máxima absoluta entre os dois:", (comparacao["discreto"] - comparacao["continuo"]).abs().max())

# Propriedade útil em sala: retornos contínuos somam no tempo, os discretos se compõem.
print("Soma dos retornos contínuos (exp - 1):", np.exp(comparacao["continuo"].sum()) - 1)
print("Composição dos retornos discretos    :", (1 + comparacao["discreto"]).prod() - 1)
print("Retorno total direto pelos preços    :", acao.iloc[-1] / acao.iloc[0] - 1)

# %% 3. Risco como volatilidade, passo a passo (slide 2)
# Calculamos variância, covariância e correlação "na mão" e conferimos com o pandas.
r_ibov = ibov.pct_change().dropna()
r_acao = acao.pct_change().dropna()
T = len(r_acao)

media_acao = r_acao.sum() / T
media_ibov = r_ibov.sum() / T

var_acao = ((r_acao - media_acao) ** 2).sum() / (T - 1)
var_ibov = ((r_ibov - media_ibov) ** 2).sum() / (T - 1)
cov_acao_ibov = ((r_acao - media_acao) * (r_ibov - media_ibov)).sum() / (T - 1)
rho_acao_ibov = cov_acao_ibov / (np.sqrt(var_acao) * np.sqrt(var_ibov))

print("Variância da ação (manual / pandas):", var_acao, r_acao.var())
print("Covariância (manual / pandas)      :", cov_acao_ibov, r_acao.cov(r_ibov))
print("Correlação (manual / pandas)       :", rho_acao_ibov, r_acao.corr(r_ibov))
print("Volatilidade diária da ação        :", np.sqrt(var_acao))
print("Volatilidade anualizada da ação    :", np.sqrt(var_acao) * np.sqrt(252))

# %% 4. Comportamento do Ibovespa: nível, retorno, histograma (slides 3 a 5)
fig, ax = plt.subplots()
ax.plot(ibov.index, ibov.values, color="navy")
ax.set_title("Ibovespa - nível")
ax.set_ylabel("Pontos")
fig.savefig("ibovpreco.png", dpi=200, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.plot(r_ibov.index, r_ibov.values, color="navy", linewidth=0.7)
ax.set_title("Ibovespa - retorno diário")
fig.savefig("ibovret.png", dpi=200, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.hist(r_ibov.values, bins=60, color="navy", alpha=0.7, density=True)
x = np.linspace(r_ibov.min(), r_ibov.max(), 300)
ax.plot(x, stats.norm.pdf(x, r_ibov.mean(), r_ibov.std()), color="firebrick", label="Normal com mesma média e desvio")
ax.set_title("Ibovespa - histograma dos retornos diários")
ax.legend()
fig.savefig("ibovhist.png", dpi=200, bbox_inches="tight")
plt.show()

print("Assimetria:", stats.skew(r_ibov), " Curtose em excesso:", stats.kurtosis(r_ibov))

# %% 5. Comportamento de uma ação: nível, retorno, histograma (slides 6 a 8)
fig, ax = plt.subplots()
ax.plot(acao.index, acao.values, color="darkgreen")
ax.set_title(f"{ticker_acao} - nível")
ax.set_ylabel("R$")
fig.savefig("acaopreco.png", dpi=200, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.plot(r_acao.index, r_acao.values, color="darkgreen", linewidth=0.7)
ax.set_title(f"{ticker_acao} - retorno diário")
fig.savefig("acaoret.png", dpi=200, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.hist(r_acao.values, bins=60, color="darkgreen", alpha=0.7, density=True)
x = np.linspace(r_acao.min(), r_acao.max(), 300)
ax.plot(x, stats.norm.pdf(x, r_acao.mean(), r_acao.std()), color="firebrick", label="Normal com mesma média e desvio")
ax.set_title(f"{ticker_acao} - histograma dos retornos diários")
ax.legend()
fig.savefig("acaohist.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 6. Comparação risco/retorno (slide 9)
# Anualização: média x 252 e volatilidade x raiz de 252.
ret_diarios = precos.pct_change().dropna()
mu_anual = ret_diarios.mean() * 252
sigma_anual = ret_diarios.std() * np.sqrt(252)

mu_ibov = r_ibov.mean() * 252
sigma_ibov = r_ibov.std() * np.sqrt(252)

fig, ax = plt.subplots()
ax.scatter(sigma_anual, mu_anual, color="darkgreen", s=60)
for nome in sigma_anual.index:
    ax.annotate(nome.replace(".SA", ""), (sigma_anual[nome], mu_anual[nome]), textcoords="offset points", xytext=(6, 4))
ax.scatter([sigma_ibov], [mu_ibov], color="navy", s=90, marker="s")
ax.annotate("Ibovespa", (sigma_ibov, mu_ibov), textcoords="offset points", xytext=(6, 4))
ax.set_xlabel("Risco (volatilidade anualizada)")
ax.set_ylabel("Retorno médio anualizado")
ax.set_title("Risco x retorno")
fig.savefig("riscoretorno.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 7. Retorno e risco de uma carteira com dois ativos (slides 13 e 14)
# Usamos PETR4 (A) e VALE3 (B). Primeiro com a correlação observada.
mu_A = mu_anual["PETR4.SA"]
mu_B = mu_anual["VALE3.SA"]
sig_A = sigma_anual["PETR4.SA"]
sig_B = sigma_anual["VALE3.SA"]
rho_AB = ret_diarios["PETR4.SA"].corr(ret_diarios["VALE3.SA"])

X_A = 0.6
X_B = 1 - X_A

ret_cart = X_A * mu_A + X_B * mu_B
var_cart = X_A**2 * sig_A**2 + X_B**2 * sig_B**2 + 2 * X_A * X_B * sig_A * sig_B * rho_AB
print("Correlação observada A,B:", rho_AB)
print("Retorno da carteira 60/40:", ret_cart)
print("Risco da carteira 60/40  :", np.sqrt(var_cart))
print("Média ponderada dos riscos (se rho = 1):", X_A * sig_A + X_B * sig_B)

# Conferência com a série de retornos da carteira rebalanceada diariamente
serie_cart = X_A * ret_diarios["PETR4.SA"] + X_B * ret_diarios["VALE3.SA"]
print("Risco anualizado pela série:", serie_cart.std() * np.sqrt(252))

# %% 8. Combinações com rho = 1, -1 e 0,5 (slides 10 a 12)
# Varremos os pesos de A de 0 a 1 e traçamos a curva risco x retorno para cada correlação.
pesos = np.linspace(0, 1, 101)
ret_curva = pesos * mu_A + (1 - pesos) * mu_B

for rho, nome_arq, titulo in [(1.0, "rho1.png", r"Combinações com $\rho = 1$"),
                              (-1.0, "rho-1.png", r"Combinações com $\rho = -1$"),
                              (0.5, "rho05.png", r"Combinações com $\rho = 0,5$")]:
    var_curva = pesos**2 * sig_A**2 + (1 - pesos) ** 2 * sig_B**2 + 2 * pesos * (1 - pesos) * sig_A * sig_B * rho
    sig_curva = np.sqrt(var_curva)
    fig, ax = plt.subplots()
    ax.plot(sig_curva, ret_curva, color="navy", linewidth=2)
    ax.scatter([sig_A, sig_B], [mu_A, mu_B], color="firebrick", zorder=3)
    ax.annotate("PETR4", (sig_A, mu_A), textcoords="offset points", xytext=(6, 4))
    ax.annotate("VALE3", (sig_B, mu_B), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Risco")
    ax.set_ylabel("Retorno esperado")
    ax.set_title(titulo)
    fig.savefig(nome_arq, dpi=200, bbox_inches="tight")
    plt.show()

# Para rho = -1 existe um peso que zera o risco: X_A = sig_B / (sig_A + sig_B)
peso_risco_zero = sig_B / (sig_A + sig_B)
print("Peso de A que zera o risco quando rho = -1:", peso_risco_zero)

# %% 9. Conjunto de mínima variância e fronteira eficiente de ativos arriscados (slides 15 a 18)
# Entradas: vetor de retornos esperados, matriz de covariância e limites para os pesos.
# Para cada retorno-alvo, achamos a carteira de menor variância. Isso gera o conjunto de mínima variância (a "bala" inteira).
# A fronteira eficiente é apenas a parte de cima, a partir da carteira de mínima variância global.
mu = mu_anual.values
cov = ret_diarios.cov().values * 252
n = len(mu)
limites = [(0.0, 1.0)] * n   # sem venda a descoberto

w0 = np.ones(n) / n
alvos = np.linspace(mu.min(), mu.max(), 60)
sig_front = []
pesos_front = []

for alvo in alvos:
    restricoes = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                  {"type": "eq", "fun": lambda w, alvo=alvo: w @ mu - alvo}]
    res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=limites, constraints=restricoes)
    sig_front.append(np.sqrt(res.fun))
    pesos_front.append(res.x)

sig_front = np.array(sig_front)
pesos_front = np.array(pesos_front)

# Carteira de mínima variância global
res_mv = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=limites,
                  constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}])
w_mv = res_mv.x
mu_mv = w_mv @ mu
sig_mv = np.sqrt(res_mv.fun)

# Separação: parte eficiente (retorno acima do da carteira de mínima variância) e parte ineficiente (abaixo)
mascara_ef = alvos >= mu_mv
sig_ef = np.concatenate([[sig_mv], sig_front[mascara_ef]])
ret_ef = np.concatenate([[mu_mv], alvos[mascara_ef]])
sig_inef = np.concatenate([sig_front[~mascara_ef], [sig_mv]])
ret_inef = np.concatenate([alvos[~mascara_ef], [mu_mv]])

# Carteiras aleatórias para mostrar a "nuvem" de possibilidades
rng = np.random.default_rng(42)
w_rand = rng.dirichlet(np.ones(n), size=5000)
mu_rand = w_rand @ mu
sig_rand = np.sqrt(np.einsum("ij,jk,ik->i", w_rand, cov, w_rand))

fig, ax = plt.subplots()
ax.scatter(sig_rand, mu_rand, s=4, color="lightgray", label="Carteiras aleatórias")
ax.plot(sig_inef, ret_inef, color="gray", linestyle="--", linewidth=2, label="Mínima variância (parte ineficiente)")
ax.plot(sig_ef, ret_ef, color="navy", linewidth=3, label="Fronteira eficiente")
ax.scatter([sig_mv], [mu_mv], color="firebrick", s=70, zorder=3, label="Carteira de mínima variância global")
ax.scatter(sigma_anual.values, mu, color="darkgreen", s=30, label="Ativos individuais")
ax.set_xlabel("Risco")
ax.set_ylabel("Retorno esperado")
ax.set_title("Fronteira eficiente de ativos arriscados")
ax.legend()
fig.savefig("fronteira.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 10. Fronteira eficiente: exemplo prático com a composição dos pesos (slide 19)
# Gráfico de área empilhada mostrando como os pesos mudam ao longo da fronteira eficiente.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(sig_ef, ret_ef, color="navy", linewidth=2)
ax1.scatter([sig_mv], [mu_mv], color="firebrick", s=70, zorder=3)
ax1.set_xlabel("Risco")
ax1.set_ylabel("Retorno esperado")
ax1.set_title("Fronteira eficiente")

ax2.stackplot(sig_front[mascara_ef], pesos_front[mascara_ef].T, labels=[t.replace(".SA", "") for t in tickers])
ax2.set_xlabel("Risco da carteira na fronteira")
ax2.set_ylabel("Peso")
ax2.set_title("Composição ao longo da fronteira eficiente")
ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.savefig("fronteiraexemplo.png", dpi=200, bbox_inches="tight")
plt.show()

tabela_mv = pd.Series(w_mv, index=[t.replace(".SA", "") for t in tickers]).round(3)
print("Pesos da carteira de mínima variância:")
print(tabela_mv)

# %% 11. Ajuste ao modelo de Markowitz: shrinkage na covariância (slide 20)
# Ledoit-Wolf encolhe a covariância amostral em direção a uma matriz estruturada.
from sklearn.covariance import LedoitWolf

lw = LedoitWolf().fit(ret_diarios.values)
cov_lw = lw.covariance_ * 252
print("Intensidade de shrinkage:", lw.shrinkage_)

res_mv_lw = minimize(lambda w: w @ cov_lw @ w, w0, method="SLSQP", bounds=limites,
                     constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}])
comparacao_pesos = pd.DataFrame({"amostral": w_mv, "shrinkage": res_mv_lw.x},
                                index=[t.replace(".SA", "") for t in tickers]).round(3)
print(comparacao_pesos)

# %% 12. Taxa livre de risco: CDI mensal do Banco Central
# O CDI mensal (série 4391 do SGS) é buscado diretamente na API do Banco Central.
# Ele serve de taxa livre de risco nas fronteiras com ativo livre de risco e, mais adiante, no CAPM.
import requests
from io import StringIO

data_ini = pd.Timestamp(inicio).strftime("%d/%m/%Y")
data_fim = pd.Timestamp(fim).strftime("%d/%m/%Y")
url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json&dataInicial={data_ini}&dataFinal={data_fim}"
cabecalho = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

resposta = requests.get(url, headers=cabecalho, timeout=60)
print("Status da resposta do BCB:", resposta.status_code)
if resposta.status_code != 200 or not resposta.text.strip().startswith("["):
    print("Conteúdo recebido:", resposta.text[:500])
    raise RuntimeError("A API do Banco Central não devolveu JSON. Veja o conteúdo acima.")

cdi_bruto = pd.read_json(StringIO(resposta.text))
cdi_bruto["data"] = pd.to_datetime(cdi_bruto["data"], format="%d/%m/%Y")
cdi_bruto["valor"] = cdi_bruto["valor"].astype(float) / 100   # CDI acumulado no mês, em decimal, nominal
cdi_mensal = cdi_bruto.set_index("data")["valor"]
cdi_mensal.index = cdi_mensal.index.to_period("M")
cdi_mensal.name = "cdi"

# Taxa livre de risco anual, nominal (média mensal x 12, consistente com a anualização dos retornos dos ativos)
rf = cdi_mensal.mean() * 12
print("Taxa livre de risco anual (CDI nominal):", rf)

# %% 13. Fronteira eficiente com ativo livre de risco (aplicação e captação à mesma taxa)
# Com um ativo livre de risco, a nova fronteira é a reta que sai de (0, rf) e tangencia a fronteira dos ativos de risco.
# O trecho entre 0 e a carteira tangente é a aplicação à taxa livre de risco; o trecho além dela é a captação à taxa livre de risco.
restr_soma = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

# Carteira tangente: maximiza o índice de Sharpe (mesmo problema com limites de 0 a 1 nos pesos)
res_t = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), w0, method="SLSQP",
                 bounds=limites, constraints=restr_soma)
w_t = res_t.x
mu_t = w_t @ mu
sig_t = np.sqrt(w_t @ cov @ w_t)
sharpe_t = (mu_t - rf) / sig_t

print("Retorno esperado da carteira tangente:", mu_t)
print("Risco da carteira tangente           :", sig_t)
print("Índice de Sharpe da tangente         :", sharpe_t)
print(pd.Series(w_t, index=[t.replace(".SA", "") for t in tickers]).round(3))

# Reta de mercado de capitais: aplicação (0 até sig_t) e captação (além de sig_t)
sig_aplic = np.linspace(0, sig_t, 50)
ret_aplic = rf + sharpe_t * sig_aplic
sig_capt = np.linspace(sig_t, 1.8 * sig_t, 50)
ret_capt = rf + sharpe_t * sig_capt

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(sig_inef, ret_inef, color="gray", linestyle=":", linewidth=1.5,
        label="Mínima variância de ativos de risco (parte ineficiente)")
ax.plot(sig_ef, ret_ef, color="gray", linewidth=2,
        label="Fronteira eficiente de ativos de risco")
ax.plot(sig_aplic, ret_aplic, color="darkgreen", linewidth=3, label="Aplicação à taxa livre de risco")
ax.plot(sig_capt, ret_capt, color="firebrick", linewidth=3, label="Captação à taxa livre de risco")
ax.scatter(sigma_anual.values, mu, color="lightgray", edgecolor="gray", s=30, zorder=2)
ax.scatter([0], [rf], color="black", s=70, zorder=4)
ax.annotate("Ativo livre de risco", (0, rf), textcoords="offset points", xytext=(8, -14))
ax.scatter([sig_t], [mu_t], color="navy", s=90, marker="D", zorder=4)
ax.annotate("Carteira tangente", (sig_t, mu_t), textcoords="offset points", xytext=(-95, 10))
ax.set_xlim(left=0)
ax.set_xlabel("Risco")
ax.set_ylabel("Retorno esperado")
ax.set_title("Fronteira eficiente com ativo livre de risco")
ax.legend(loc="lower right")
fig.savefig("fronteira_rf.png", dpi=200, bbox_inches="tight")
plt.show()

# Exemplo para sala: um investidor que aceita risco 50% maior que o da tangente capta 50% do patrimônio a rf
fator = 1.5
print(f"Para risco igual a {fator} vezes o da tangente: peso na tangente = {fator:.2f}, peso no ativo livre de risco = {1 - fator:.2f}")
print("Retorno esperado correspondente:", rf + fator * (mu_t - rf))

# %% 14. Fronteira eficiente com taxa de aplicação (lending) menor que taxa de captação (borrowing)
# O investidor aplica a rl e capta a rb, com rl < rb. Cada taxa gera sua própria carteira tangente.
rl = rf
spread_captacao = 0.03          # spread assumido de 3 pontos percentuais ao ano; altere à vontade
rb = rl + spread_captacao
print("Taxa de aplicação rl:", rl, " Taxa de captação rb:", rb)

# Tangente a partir de rl (define o fim da reta de aplicação)
res_tl = minimize(lambda w: -(w @ mu - rl) / np.sqrt(w @ cov @ w), w0, method="SLSQP",
                  bounds=limites, constraints=restr_soma)
w_tl = res_tl.x
mu_tl = w_tl @ mu
sig_tl = np.sqrt(w_tl @ cov @ w_tl)
sharpe_l = (mu_tl - rl) / sig_tl

# Tangente a partir de rb (define o início da reta de captação)
res_tb = minimize(lambda w: -(w @ mu - rb) / np.sqrt(w @ cov @ w), w0, method="SLSQP",
                  bounds=limites, constraints=restr_soma)
w_tb = res_tb.x
mu_tb = w_tb @ mu
sig_tb = np.sqrt(w_tb @ cov @ w_tb)
sharpe_b = (mu_tb - rb) / sig_tb

print("Tangente de aplicação: retorno", mu_tl, " risco", sig_tl, " Sharpe", sharpe_l)
print("Tangente de captação : retorno", mu_tb, " risco", sig_tb, " Sharpe", sharpe_b)

# Arco da fronteira dos ativos de risco entre as duas tangentes
alvos_arco = np.linspace(mu_tl, mu_tb, 60)
sig_arco = []
for alvo in alvos_arco:
    restricoes = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                  {"type": "eq", "fun": lambda w, alvo=alvo: w @ mu - alvo}]
    res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=limites, constraints=restricoes)
    sig_arco.append(np.sqrt(res.fun))
sig_arco = np.array(sig_arco)

# Trechos da nova fronteira
sig_aplic2 = np.linspace(0, sig_tl, 50)
ret_aplic2 = rl + sharpe_l * sig_aplic2
sig_capt2 = np.linspace(sig_tb, 1.8 * sig_tb, 50)
ret_capt2 = rb + sharpe_b * sig_capt2

# Prolongamentos que deixam de valer por causa do spread
sig_prol_l = np.linspace(sig_tl, 1.8 * sig_tb, 50)
ret_prol_l = rl + sharpe_l * sig_prol_l
sig_prol_b = np.linspace(0, sig_tb, 50)
ret_prol_b = rb + sharpe_b * sig_prol_b

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(sig_ef, ret_ef, color="lightgray", linewidth=1.5,
        label="Fronteira eficiente de ativos de risco")
ax.plot(sig_prol_l, ret_prol_l, color="gray", linestyle="--", linewidth=1, label="Prolongamentos que deixam de valer")
ax.plot(sig_prol_b, ret_prol_b, color="gray", linestyle="--", linewidth=1)
ax.plot(sig_aplic2, ret_aplic2, color="darkgreen", linewidth=3, label="Aplicação à taxa rl")
ax.plot(sig_arco, alvos_arco, color="navy", linewidth=3, label="Carteiras de ativos de risco (sem aplicar nem captar)")
ax.plot(sig_capt2, ret_capt2, color="firebrick", linewidth=3, label="Captação à taxa rb")
ax.scatter([0], [rl], color="black", s=70, zorder=4)
ax.annotate("rl (aplicação)", (0, rl), textcoords="offset points", xytext=(8, -14))
ax.scatter([0], [rb], color="black", s=70, zorder=4)
ax.annotate("rb (captação)", (0, rb), textcoords="offset points", xytext=(8, 6))
ax.scatter([sig_tl], [mu_tl], color="darkgreen", s=90, marker="D", zorder=4)
ax.annotate("Tangente de rl", (sig_tl, mu_tl), textcoords="offset points", xytext=(-95, 10))
ax.scatter([sig_tb], [mu_tb], color="firebrick", s=90, marker="D", zorder=4)
ax.annotate("Tangente de rb", (sig_tb, mu_tb), textcoords="offset points", xytext=(8, -16))
ax.set_xlim(left=0)
ax.set_xlabel("Risco")
ax.set_ylabel("Retorno esperado")
ax.set_title("Fronteira eficiente com taxa de aplicação menor que a taxa de captação")
ax.legend(loc="lower right")
fig.savefig("fronteira_rl_rb.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 15. CAPM na prática: dados mensais (slides 21 e 22)
# Retornos mensais dos últimos 5 anos, Ibovespa como mercado e CDI na curva como ativo livre de risco.
precos_mensais = dados[["^BVSP", ticker_acao]].resample("ME").last()
ret_mensal = precos_mensais.pct_change().dropna()
ret_mensal.index = ret_mensal.index.to_period("M")

base = ret_mensal.join(cdi_mensal, how="inner").dropna()
base["premio_acao"] = base[ticker_acao] - base["cdi"]
base["premio_mercado"] = base["^BVSP"] - base["cdi"]
print(base.tail())
print("Número de observações mensais:", len(base))

# %% 16. Retornos da ação contra o Ibovespa (slide 23)
fig, ax = plt.subplots()
ax.scatter(base["^BVSP"], base[ticker_acao], color="navy", alpha=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Retorno mensal do Ibovespa")
ax.set_ylabel(f"Retorno mensal de {ticker_acao}")
ax.set_title("Retornos mensais: ação x Ibovespa")
fig.savefig("scatteracaoibov.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 17. Regressão do CAPM com prêmios de risco (slide 24)
# Y = beta0 + beta1 X + erro, com X o prêmio de risco de mercado e Y o prêmio de risco da ação.
reg = stats.linregress(base["premio_mercado"], base["premio_acao"])
beta0 = reg.intercept   # alfa de Jensen mensal
beta1 = reg.slope       # beta da ação
print("Alfa mensal :", beta0, " (erro padrão:", reg.intercept_stderr, ")")
print("Beta        :", beta1, " (erro padrão:", reg.stderr, ")")
print("R ao quadrado:", reg.rvalue**2)
print("p-valor do beta:", reg.pvalue)

# Conferência manual: beta = cov(X,Y) / var(X)
beta_manual = base["premio_mercado"].cov(base["premio_acao"]) / base["premio_mercado"].var()
print("Beta manual :", beta_manual)

xx = np.linspace(base["premio_mercado"].min(), base["premio_mercado"].max(), 100)
fig, ax = plt.subplots()
ax.scatter(base["premio_mercado"], base["premio_acao"], color="navy", alpha=0.7)
ax.plot(xx, beta0 + beta1 * xx, color="firebrick", linewidth=2, label=f"beta = {beta1:.2f}")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Prêmio de risco do mercado (Ibovespa - CDI)")
ax.set_ylabel(f"Prêmio de risco de {ticker_acao} (ação - CDI)")
ax.set_title("CAPM: regressão dos prêmios de risco")
ax.legend()
fig.savefig("regressao.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 18. Retorno esperado pelo CAPM (slide 21)
# r_i = r_f + beta (r_m - r_f), com tudo anualizado em termos nominais.
rf_anual = (1 + base["cdi"]).prod() ** (12 / len(base)) - 1
rm_anual = (1 + base["^BVSP"]).prod() ** (12 / len(base)) - 1
premio_mercado_anual = rm_anual - rf_anual

ret_esperado_capm = rf_anual + beta1 * premio_mercado_anual
ret_realizado_acao = (1 + base[ticker_acao]).prod() ** (12 / len(base)) - 1

print("Taxa livre de risco anual (CDI)      :", rf_anual)
print("Retorno anual do Ibovespa            :", rm_anual)
print("Prêmio de risco de mercado anual     :", premio_mercado_anual)
print("Retorno esperado pelo CAPM da ação   :", ret_esperado_capm)
print("Retorno anualizado realizado da ação :", ret_realizado_acao)