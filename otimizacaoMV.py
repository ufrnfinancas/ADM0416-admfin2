import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Dados fictícios consistentes
np.random.seed(42)
assets = ["Ativo A", "Ativo B", "Ativo C", "Ativo D", "Ativo E"]
num_assets = len(assets)
returns = np.array([0.10, 0.12, 0.08, 0.11, 0.09])  # Retornos esperados (10% a 12%)
cov_matrix = np.array([
    [0.04, 0.02, 0.01, 0.03, 0.02],
    [0.02, 0.05, 0.02, 0.01, 0.02],
    [0.01, 0.02, 0.03, 0.01, 0.01],
    [0.03, 0.01, 0.01, 0.04, 0.02],
    [0.02, 0.02, 0.01, 0.02, 0.03]
])  # Matriz de covariância (ajustada manualmente)

# Funções para cálculo de portfólio
def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.03):
    port_return = portfolio_return(weights, returns)
    port_risk = portfolio_risk(weights, cov_matrix)
    return -(port_return - risk_free_rate) / port_risk

# Restrições e pesos iniciais
constraints = (
    {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Soma dos pesos = 1
)
bounds = [(0, 1) for _ in range(num_assets)]  # Pesos entre 0 e 1
initial_weights = np.ones(num_assets) / num_assets

# Otimização para portfólio de máxima Sharpe Ratio
result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(returns, cov_matrix),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
optimal_weights = result.x

# Calculando a Fronteira Eficiente
risks, returns_list = [], []
weights_list = []
for target_return in np.linspace(returns.min(), returns.max(), 50):
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "eq", "fun": lambda x: portfolio_return(x, returns) - target_return},
    )
    result = minimize(
        portfolio_risk,
        initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if result.success:
        risks.append(result.fun)
        returns_list.append(target_return)
        weights_list.append(result.x)

# Plotando a Fronteira Eficiente
plt.figure(figsize=(10, 6))
plt.plot(risks, returns_list, label="Fronteira Eficiente", color="blue")
plt.scatter(
    portfolio_risk(optimal_weights, cov_matrix),
    portfolio_return(optimal_weights, returns),
    color="red",
    label="Portfolio Ótimo (Máx. Sharpe)",
    marker="*",
    s=200,
)
plt.title("Fronteira Eficiente e Portfolio Ótimo")
plt.xlabel("Risco (Desvio Padrão)")
plt.ylabel("Retorno Esperado")
plt.legend()
plt.grid(True)
plt.show()

# Pesos do portfólio ótimo
for asset, weight in zip(assets, optimal_weights):
    print(f"{asset}: {weight:.2%}")
