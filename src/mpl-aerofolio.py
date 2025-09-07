import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score

# 1. Carregar os dados normalizados
airfoil_self_noise = pd.read_table('./airfoil+self+noise/airfoil_self_noise.dat',
                                   names=["frequência", "angulo_ataque", "comprimento_corda", "velocidade_fluxo",
                                          "espessura_sucção", "pressão_sonora"])
df_normalizado = (airfoil_self_noise - airfoil_self_noise.mean()
                  ) / airfoil_self_noise.std()

X = df_normalizado.drop(columns=['pressão_sonora']).values
y = df_normalizado['pressão_sonora'].values

# 2. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Definir taxa de aprendizado com decaimento exponencial
initial_lr = 0.01
lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr,
                               decay_steps=100, decay_rate=0.96, staircase=True
                               )

# 4. Construir o modelo
model = Sequential([
    Dense(10, activation='relu', input_shape=(X.shape[1],)), #,kernel_regularizer=regularizers.l2(0.001)
    Dense(1, activation='linear')  # saída contínua
])

# 5. Compilar o modelo
model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss='mse', metrics=['mae'])

# 6. Treinar o modelo
history = model.fit(X_train, y_train, epochs=2000,
                    batch_size=32, validation_split=0.2, verbose=1
                    )

# 7. Avaliar no conjunto de teste
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Erro quadrático médio no teste: {test_loss:.4f}")
print(f"Erro absoluto médio no teste: {test_mae:.4f}")

# Vamos ver como foi o treino?
fig, ax = plt.subplots(1, 2, figsize=(10, 8))
ax[0].semilogx(history.history['loss'], color='b', label="Treinamento loss")
ax[0].semilogx(history.history['val_loss'], color='r', label="Validação loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].semilogx(history.history['mae'], color='b', label="Treinamento mae")
ax[1].semilogx(history.history['val_mae'], color='r', label="Validação mae")
legend = ax[1].legend(loc='best', shadow=True)
plt.grid(True)
plt.savefig('img/t1_training_validation.png', format='png')

# 1. Predições do modelo
y_pred = model.predict(X_test).flatten()

# 2. Coeficiente de determinação (R^2)
r2 = r2_score(y_test, y_pred)
print(f"Coeficiente de determinação R^2: {r2:.4f}")

# 3. Erro Absoluto Médio Relativo (RAE)
rae = np.sum(np.abs(y_test - y_pred)) / \
    np.sum(np.abs(y_test - np.mean(y_test)))
print(f"Erro Absoluto Médio Relativo (RAE): {rae:.4f}")

# 4. Análise gráfica de resíduos
residuos = y_test - y_pred

plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuos, alpha=0.5, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos (y_real - y_pred)')
plt.title('Análise de Resíduos')
plt.grid(True)
plt.savefig('img/t1_residuos.png', format='png')
plt.show()

y_pred = model.predict(X_test).flatten()
r2 = r2_score(y_test, y_pred)

# Gráfico: valores reais vs preditos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(),
         y_test.max()], 'r--', label='Ideal (y = x)')

# Adiciona R^2 no gráfico
plt.text(x=min(y_test), y=max(y_pred),
         s=f"$R^2$ = {r2:.4f}", fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Valores Reais vs Preditos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('img/t1_valores_reais_preditos_leg.png', format='png')
plt.show()

# Experimentos

# Lista para armazenar resultados
resultados = []
neuronios_testados = []

# Loop para testar diferentes quantidades de neurônios
for neuronios in range(10, 10000, 100):
    # Aprendizado com decaimento exponencial
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01,
                                   decay_steps=100, decay_rate=0.96, staircase=True
                                   )

    # Modelo MLP
    model = Sequential([
        Dense(neuronios, activation='sigmoid', input_shape=(X.shape[1],)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse')
    model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

    # Avaliação
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    resultados.append(r2)
    neuronios_testados.append(neuronios)

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.semilogx(neuronios_testados, resultados, marker='o', linestyle='--')
plt.xlabel("Número de Neurônios na Camada Oculta")
plt.ylabel(f"$R^2$ no Teste")
plt.title("Impacto da Quantidade de Neurônios na Performance")
plt.grid(True)
plt.savefig('img/t1_impacto_neuronio_performace.png', format='png')
plt.show()
