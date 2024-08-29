# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:34:39 2024

@author: María Eugenia
"""

#Importamos las librerías
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Estamos importando nuestra base de datos, para su correcta lectura añadimos delimiter = ';'
df = pd.read_csv('C:/Users/maria/OneDrive/Escritorio/Ingeniería Informática/2º CARRERA/Sistemas inteligentes/predict+students+dropout+and+academic+success/data.csv', delimiter=';')

# Importar las librerías necesarias para el modelo de vecinos más cercanos y la validación cruzada
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Seleccionamos las columnas que son atributos (X), y después la columna objetivo (y)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

n_divisiones = 10  # 10 divisiones
kf = KFold(n_divisiones, shuffle=True, random_state=2)

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'n_neighbors': list(range(1, 20)), 
              'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=kf, scoring='f1_weighted')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_n_neighbors = best_params['n_neighbors']
best_metric = best_params['metric']

# Ahora vamos a crear una lista donde vamos a guardar los valores de cada una de nuestras métricas en cada una de las 10 iteraciones
valor_precision = []
valor_f1 = []
valor_exactitud = []

for train_index, test_index in kf.split(X):
    # Renombramos
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights='distance', algorithm='auto', metric=best_metric, p=2)
    model.fit(X_train, y_train)  # Ajusta el modelo al training data
    y_pred = model.predict(X_test)  # Predicciones

    # Guardamos los valores en las listas previamente inicializadas
    valor_precision.append(precision_score(y_test, y_pred, average='weighted'))  # Como nuestra base de datos no es binaria tomo como average='weighted'
    valor_f1.append(f1_score(y_test, y_pred, average='weighted'))
    valor_exactitud.append(accuracy_score(y_test, y_pred))

# Imprimimos nuestras listas con los valores ya establecidos
print("Valores de precisión:", valor_precision)
print("Valores de F1:", valor_f1)
print("Valores de exactitud:", valor_exactitud)

# A continuación calculamos la media de cada una de las medidas de rendimiento
precision_mean = np.mean(valor_precision)
f1_mean = np.mean(valor_f1)
exactitud_mean = np.mean(valor_exactitud)

# Imprimimos la mejor media de cada métrica
print(f"Mejor precisión: {precision_mean:.4f}")
print(f"Mejor F1: {f1_mean:.4f}")
print(f"Mejor exactitud (accuracy): {exactitud_mean:.4f}")

# Imprimimos el mejor número de vecinos
print(f"Mejor número de vecinos: {best_n_neighbors}")
print(f"Mejor métrica: {best_metric}")