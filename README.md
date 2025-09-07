# Predição de Ruído em Aerofólios com Redes Neurais

Este projeto aplica um **Multi-Layer Perceptron (MLP)** para modelar o ruído em aerofólios utilizando o conjunto de dados [Airfoil Self-Noise](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self+Noise) da NASA. O trabalho inclui análise exploratória, implementação de rede neural em TensorFlow, experimentos comparativos e extrapolação de resultados.

## 📂 Estrutura do Repositório
- `notebooks/` → análises exploratórias e experimentos em Jupyter.  
- `src/` → scripts Python para treinamento e avaliação do modelo.  
- `report/` → arquivos LaTeX e PDF do relatório.  
- `models/` → modelo treinado (`mlp_model.h5`).  
- `requirements.txt` → dependências do projeto.  

## 🚀 Como Reproduzir
1. Clone o repositório:
   ```bash
   git clone https://github.com/BrunodaSilvaMachado/AnaliseRuidosAerofoliosMLP.git
   cd projeto-mlp-aerofolio

## Modelo

O modelo desenvolvido nesse trabalho possui as seguintes configurações:
- 1 camada intermediária com 10 neurônios;
- Função de ativação sigmoide;
- Otimizador Adam com taxa de aprendizado com decaimento exponencial;
- Função de perda Erro Quadrático Médio (MSE);
- Mini-batch com 32 amostras;
- Treinamento realizado em 1000 épocas.

## Resultados

|Métrica| Valor|
|---|---|
|MSE |0,1595|
|MAE |0,2929|
|RAE |0,3481|
|R2 |0,8485|

## Relatório

O relatório completo está disponível em report/projeto_final.pdf.

## 📌 Referências

UCI Machine Learning Repository – Airfoil Self-Noise Dataset.

Bishop, C. M. Neural Networks for Pattern Recognition.

LeCun, Y. et al. Deep Learning.
