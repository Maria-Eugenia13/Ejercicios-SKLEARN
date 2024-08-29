# -*- coding: utf-8 -*-
"""
Created on Fri May 31 02:17:49 2024

@author: María Eugenia
"""
#Importamos las librerías
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Estamos importando nuestra base de datos, para su correcta lectura añadimos delimiter = ';'
df = pd.read_csv('C:/Users/maria/OneDrive/Escritorio/Ingeniería Informática/2º CARRERA/Sistemas inteligentes/predict+students+dropout+and+academic+success/data.csv', delimiter=';')

#Seleccionamos las columnas que son atributos (X), y después la columna objetivo (y)  
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

n_divisiones = 10 #10 divisiones
kf = KFold(n_divisiones, shuffle=True, random_state=2)

TreeClassifier = DecisionTreeClassifier(random_state=30)
    
param = {'criterion': ['gini','entropy', 'log_loss'], 'splitter': ['best'], 'max_depth': range(3, 10), 'min_samples_split': range(2, 9), 'min_samples_leaf': range(1, 6)}

grid_search = GridSearchCV(TreeClassifier, param_grid=param, cv=kf, scoring='accuracy')

grid_search.fit(X, y)

# Para conseguir el mejor modelo
best_tree = grid_search.best_estimator_

# Encuentra los mejores hiperparámetros
best_params = grid_search.best_params_

model = DecisionTreeClassifier(**best_params)
model.fit(X, y)

# Ahora vamos a crear como una lista donde vamos a guardar los valores de cada una de nuestras métricas en cada una de las 10 iteraciones
valor_precision = []
valor_f1 = []
valor_exactitud = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train) # Ajusta el modelo al training data
    y_pred = model.predict(X_test) #Predicciones
    
    # Guardamos los valores en las listas previamente inicializadas
    valor_precision.append(precision_score(y_test, y_pred, average='weighted')) #Como nuestra base de datos no es binaria tomo como average='weighted'
    valor_f1.append(f1_score(y_test, y_pred, average='weighted'))
    valor_exactitud.append(accuracy_score(y_test, y_pred))
    
#Imprimimos nuestras listas con los valores ya establecidos
print("Valores de precisión:", valor_precision)
print("Valores de F1:", valor_f1)
print("Valores de exactitud:", valor_exactitud)
  
# A continuación calculamos la media de de cada una de las medidas de rendimiento
precision_mean = np.mean(valor_precision)
f1_mean = np.mean(valor_f1)
exactitud_mean = np.mean(valor_exactitud)
    

# Mejores hiperparámetros
print("Los mejores hiperparámetros son:", best_params)

#Imprimimos la mejor media de cada métrica
print(f"Mejor precisión: {precision_mean:.4f}")
print(f"Mejor f1: {f1_mean:.4f}")
print(f"Mejor exactitud (accuracy): {exactitud_mean:.4f}")

# Grafica el mejor árbol de decisión
plt.figure(figsize=(200, 100))
plot_tree(best_tree, filled=True)
plt.show()