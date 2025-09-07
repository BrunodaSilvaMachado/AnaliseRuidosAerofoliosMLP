import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import seaborn as sns
from scipy import stats

airfoil_self_noise = pd.read_table('./airfoil+self+noise/airfoil_self_noise.dat',
                                   names=["frequência", "angulo_ataque", "comprimento_corda", "velocidade_fluxo",
                                          "espessura_sucção", "pressão_sonora"])
X = airfoil_self_noise.iloc[:, 0:5]
y = airfoil_self_noise.iloc[:, 5:6]

# Gerar scatter plot com curva LOWESS para cada entrada
for coluna in airfoil_self_noise.columns[:-1]:  # Exclui a variável alvo
    g = sns.lmplot(x=coluna, y='pressão_sonora',
                   data=airfoil_self_noise, lowess=True, line_kws={'color': 'red'})
    g.ax.grid(True, axis='both')
    sns.despine(fig=None, ax=None, top=False, right=False,
                left=False, bottom=False, offset=None, trim=False)
    plt.title(f'{coluna} vs Pressão Sonora (com LOWESS)')
    plt.xlabel(coluna)
    plt.ylabel('pressão_sonora')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'img/t1_disperssao_lowess_{coluna}.png', format='png')
    plt.show()

# Calcular matriz de correlação
corr = airfoil_self_noise.corr()

# Visualizar com heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação de Pearson')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('img/t1_matriz_correlacao.png', format='png')
plt.show()

# 1. Detectar Outliers usando Z-score
# calcula z-score absoluto
z_scores = np.abs(stats.zscore(airfoil_self_noise))
outliers = z_scores > 3                        # define outliers como z > 3
outliers_por_variavel = pd.Series(
    np.sum(outliers, axis=0), index=airfoil_self_noise.columns)

# print("Quantidade de outliers por variável:")
print(outliers_por_variavel)

# Configurar estilo dos gráficos
sns.set(style="whitegrid", font_scale=1.5)

# Criar boxplots para cada variável
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # cria um grid 2x3
axes = axes.flatten()  # achata para iterar facilmente

for idx, coluna in enumerate(airfoil_self_noise.columns):
    sns.boxplot(data=airfoil_self_noise, x=coluna,
                ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Boxplot de {coluna}')
    axes[idx].set_xlabel("")  # remove o nome do eixo X para estética

plt.tight_layout()
plt.savefig('img/t1_boxplot.png', format='png')
plt.show()
