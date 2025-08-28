import yfinance as yf
import pandas as pd
from datetime import datetime

# Código do fundo - ajuste conforme necessário
ticker = "MSCI"  # ou o código correto do fundo Morgan Stanley

# Busca dados do fundo
fund = yf.Ticker(ticker)
hist = fund.history(period="1y")
info = fund.info

# Dados básicos
current_price = hist['Close'].iloc[-1]
initial_price = hist['Close'].iloc[0]
total_return = ((current_price - initial_price) / initial_price) * 100

# Retornos diários
daily_returns = hist['Close'].pct_change().dropna()
volatility = daily_returns.std() * (252 ** 0.5) * 100

# Métricas de performance
annualized_return = ((current_price / initial_price) ** (252/len(hist)) - 1) * 100
sharpe_ratio = (annualized_return - 2) / volatility if volatility > 0 else 0

# Drawdown
rolling_max = hist['Close'].cummax()
drawdown = (hist['Close'] / rolling_max - 1) * 100
max_drawdown = drawdown.min()

# Mostra resultados
print(f"Fundo: {ticker}")
print(f"Preço atual: {current_price:.2f}")
print(f"Retorno total: {total_return:.2f}%")
print(f"Retorno anualizado: {annualized_return:.2f}%")
print(f"Volatilidade: {volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Drawdown máximo: {max_drawdown:.2f}%")

# Salva em Excel
hist.to_excel('historico_fundo.xlsx')
print("Histórico salvo em historico_fundo.xlsx")