import numpy as np
import matplotlib.pyplot as plt

def calcular_vpl(fluxos_de_caixa, taxa_desconto):
    vpl = 0
    for ano, fluxo in enumerate(fluxos_de_caixa):
        vpl += fluxo / (1 + taxa_desconto) ** ano
    return vpl

# Definindo os fluxos de caixa e taxas de desconto para diferentes projetos
projetos = [
    {"fluxos_de_caixa": [-100000, 30000, 40000, 40000, 30000, 20000], "taxa_desconto": 0.10},
    {"fluxos_de_caixa": [-120000, 40000, 50000, 50000, 40000, 30000], "taxa_desconto": 0.12},
    {"fluxos_de_caixa": [-80000, 20000, 30000, 40000, 50000, 60000], "taxa_desconto": 0.08}
]

# Calculando e exibindo o VPL para cada projeto
for i, projeto in enumerate(projetos, start=1):
    vpl = calcular_vpl(projeto["fluxos_de_caixa"], projeto["taxa_desconto"])
    print("Projeto {}: O Valor Presente Líquido (VPL) é: ${:,.2f}".format(i, vpl))

    # Criando gráfico de barras para os fluxos de caixa
    anos = range(len(projeto["fluxos_de_caixa"]))
    plt.figure(figsize=(8, 6))
    plt.bar(anos, projeto["fluxos_de_caixa"], color='blue', alpha=0.7)
    plt.xlabel('Ano')
    plt.ylabel('Fluxo de Caixa ($)')
    plt.title('Fluxo de Caixa do Projeto {}'.format(i))
    plt.xticks(anos)
    plt.grid(True)
    plt.show()
