#histograma
import matplotlib.pyplot as plt
import numpy as np

# Dados das notas
notas = [4, 4, 6, 0, 2, 3, 9, 6, 4, 6, 7, 3, 4, 7, 2, 10, 5, 8, 4, 6, 4, 6, 4, 10, 8, 6, 4, 4, 7, 5, 3, 3, 6, 6, 6, 5, 10, 9, 5, 10, 4, 2, 9, 10, 1, 7, 5, 9, 10, 10, 2, 3]

# Criar o histograma
plt.figure(figsize=(10, 6))
plt.hist(notas, bins=11, range=(-0.5, 10.5), edgecolor='black', alpha=0.7, color='skyblue')

# Personalizar o gráfico
plt.title('Histograma das Notas', fontsize=16, fontweight='bold')
plt.xlabel('Nota', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xticks(range(0, 11))
plt.grid(axis='y', alpha=0.3)

# Adicionar estatísticas básicas
media = np.mean(notas)
mediana = np.median(notas)
plt.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')
plt.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
plt.legend()

# Mostrar informações estatísticas
print(f"Total de notas: {len(notas)}")
print(f"Média: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Desvio padrão: {np.std(notas):.2f}")
print(f"Nota mínima: {min(notas)}")
print(f"Nota máxima: {max(notas)}")

# Mostrar o gráfico
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Dados das notas e faltas
notas = [4, 4, 6, 0, 2, 3, 9, 0, 6, 4, 0, 6, 0, 7, 3, 4, 7, 2, 10, 5, 8, 4, 6, 4, 6, 4, 10, 8, 6, 4, 4, 7, 0, 5, 3, 3, 6, 6, 6, 5, 10, 9, 5, 10, 4, 0, 2, 9, 10, 1, 7, 5, 9, 0, 10, 10, 2, 3]
faltas = [5, 4, 0, 8, 8, 7, 8, 6, 9, 1, 5, 4, 22, 8, 2, 5, 0, 6, 0, 2, 8, 7, 0, 7, 4, 3, 3, 4, 3, 4, 8, 8, 8, 5, 8, 12, 12, 8, 5, 2, 7, 4, 15, 10, 6, 16, 7, 4, 4, 4, 8, 11, 13, 9, 6, 5, 10, 3]

# Verificar se os dados têm o mesmo tamanho
print(f"Número de notas: {len(notas)}")
print(f"Número de faltas: {len(faltas)}")
print(f"Datasets compatíveis: {len(notas) == len(faltas)}\n")

# Criar DataFrame para facilitar análise
df = pd.DataFrame({
    'Notas': notas,
    'Faltas': faltas
})

# ===== ANÁLISE ESTATÍSTICA DESCRITIVA =====
print("=== ESTATÍSTICAS DESCRITIVAS ===")
print("\nNotas:")
print(f"Média: {np.mean(notas):.2f}")
print(f"Mediana: {np.median(notas):.2f}")
print(f"Desvio padrão: {np.std(notas):.2f}")
print(f"Min: {min(notas)} | Max: {max(notas)}")

print("\nFaltas:")
print(f"Média: {np.mean(faltas):.2f}")
print(f"Mediana: {np.median(faltas):.2f}")
print(f"Desvio padrão: {np.std(faltas):.2f}")
print(f"Min: {min(faltas)} | Max: {max(faltas)}")

# ===== ANÁLISE DE CORRELAÇÃO =====
print("\n=== ANÁLISE DE CORRELAÇÃO ===")
# Correlação de Pearson
corr_pearson, p_value_pearson = stats.pearsonr(notas, faltas)
print(f"Correlação de Pearson: {corr_pearson:.4f}")
print(f"P-value (Pearson): {p_value_pearson:.4f}")

# Correlação de Spearman (não paramétrica)
corr_spearman, p_value_spearman = stats.spearmanr(notas, faltas)
print(f"Correlação de Spearman: {corr_spearman:.4f}")
print(f"P-value (Spearman): {p_value_spearman:.4f}")

# Interpretação da correlação
def interpretar_correlacao(r):
    if abs(r) < 0.1:
        return "Correlação muito fraca"
    elif abs(r) < 0.3:
        return "Correlação fraca"
    elif abs(r) < 0.5:
        return "Correlação moderada"
    elif abs(r) < 0.7:
        return "Correlação forte"
    else:
        return "Correlação muito forte"

print(f"\nInterpretação: {interpretar_correlacao(corr_pearson)}")
if p_value_pearson < 0.05:
    print("A correlação é estatisticamente significativa (p < 0.05)")
else:
    print("A correlação NÃO é estatisticamente significativa (p >= 0.05)")

# ===== REGRESSÃO LINEAR =====
print("\n=== ANÁLISE DE REGRESSÃO LINEAR ===")
slope, intercept, r_value, p_value_reg, std_err = stats.linregress(faltas, notas)
print(f"Equação da reta: Nota = {intercept:.2f} + ({slope:.2f} × Faltas)")
print(f"R² (coeficiente de determinação): {r_value**2:.4f}")
print(f"P-value da regressão: {p_value_reg:.4f}")
print(f"Erro padrão: {std_err:.4f}")

# ===== VISUALIZAÇÕES =====
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Scatter plot com linha de regressão
ax1.scatter(faltas, notas, alpha=0.6, color='blue', s=50)
x_reg = np.linspace(min(faltas), max(faltas), 100)
y_reg = slope * x_reg + intercept
ax1.plot(x_reg, y_reg, 'r--', linewidth=2, label=f'y = {intercept:.2f} + {slope:.2f}x')
ax1.set_xlabel('Faltas')
ax1.set_ylabel('Notas')
ax1.set_title(f'Relação Notas vs Faltas\n(r = {corr_pearson:.3f}, p = {p_value_pearson:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Histograma das notas
ax2.hist(notas, bins=11, range=(-0.5, 10.5), alpha=0.7, color='skyblue', edgecolor='black')
ax2.set_xlabel('Notas')
ax2.set_ylabel('Frequência')
ax2.set_title('Distribuição das Notas')
ax2.set_xticks(range(0, 11))

# 3. Histograma das faltas
ax3.hist(faltas, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
ax3.set_xlabel('Faltas')
ax3.set_ylabel('Frequência')
ax3.set_title('Distribuição das Faltas')

# 4. Box plot comparativo
box_data = [notas, faltas]
ax4.boxplot(box_data, labels=['Notas', 'Faltas'])
ax4.set_title('Box Plot Comparativo')
ax4.set_ylabel('Valores')

plt.tight_layout()
plt.show()

# ===== ANÁLISE POR CATEGORIAS =====
print("\n=== ANÁLISE POR CATEGORIAS DE FALTAS ===")
# Categorizar faltas
def categorizar_faltas(falta):
    if falta <= 3:
        return "Baixas (0-3)"
    elif falta <= 7:
        return "Médias (4-7)"
    elif falta <= 12:
        return "Altas (8-12)"
    else:
        return "Muito Altas (13+)"

df['Categoria_Faltas'] = df['Faltas'].apply(categorizar_faltas)

# Estatísticas por categoria
print("Nota média por categoria de faltas:")
for categoria in df['Categoria_Faltas'].unique():
    subset = df[df['Categoria_Faltas'] == categoria]
    print(f"{categoria}: {subset['Notas'].mean():.2f} (n={len(subset)})")

# ===== TESTE ANOVA =====
print("\n=== TESTE ANOVA ===")
grupos = [df[df['Categoria_Faltas'] == cat]['Notas'].values 
          for cat in df['Categoria_Faltas'].unique()]
f_stat, p_value_anova = stats.f_oneway(*grupos)
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value_anova:.4f}")

if p_value_anova < 0.05:
    print("Há diferença significativa entre as médias dos grupos (p < 0.05)")
else:
    print("NÃO há diferença significativa entre as médias dos grupos (p >= 0.05)")

# ===== RESUMO FINAL =====
print("\n" + "="*50)
print("RESUMO DA ANÁLISE")
print("="*50)
print(f"• Correlação entre notas e faltas: {corr_pearson:.3f}")
print(f"• Significância estatística: {'Sim' if p_value_pearson < 0.05 else 'Não'}")
print(f"• {interpretar_correlacao(corr_pearson)}")
print(f"• R² = {r_value**2:.3f} ({r_value**2*100:.1f}% da variação nas notas é explicada pelas faltas)")
print(f"• A cada falta adicional, a nota diminui em média {abs(slope):.2f} pontos")