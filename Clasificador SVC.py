# -*- coding: utf-8 -*-
"""
Created on Fri May 31 02:23:11 2024

@author: María Eugenia
"""
#Importamos las librerías
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#Estamos importando nuestra base de datos, para su correcta lectura añadimos delimiter = ';'
df = pd.read_csv('C:/Users/maria/OneDrive/Escritorio/Ingeniería Informática/2º CARRERA/Sistemas inteligentes/predict+students+dropout+and+academic+success/data.csv', delimiter=';')

# Importamos las librerías para el modelo de la validación cruzada, Naïve Bayes, y además importamos las métricas que vamos a usar
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Seleccionamos las columnas que son atributos (X), y después la columna objetivo (y)  

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]


n_divisiones = 10 #10 divisiones
kf = KFold(n_divisiones, shuffle=True, random_state=2)

# Definimos los parametros 
parametros = {'C' : [0.1, 1, 10, 100], 'gamma' : ['scaled', 'auto'] + [0.001, 0.01, 0.1, 1], 'kernel' : ['sigmoid', 'rbf'], 'decision_function_shape':['ovo']}

clasificadorSVC = SVC()

grid_search = GridSearchCV(clasificadorSVC, parametros, cv = kf, scoring='accuracy')

# Entrenamos el modelo 
grid_search.fit(X, y)

# Mejores hiperparámetros
best_params = grid_search.best_params_

# Imprimimos los mejores hiperparametros 
print("Los mejores hiperparámetros son: ", best_params)

# Entrena el mejor modelo 
mejor_svc = SVC(**best_params)
mejor_svc.fit(X, y)

# Ahora vamos a crear una lista donde vamos a guardar los valores de cada una de nuestras métricas en cada una de las 10 iteraciones
valor_precision = []
valor_f1 = []
valor_exactitud = []

# Realiza la validación cruzada manualmente con los mejores hiperparámetros
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    mejor_svc.fit(X_train, y_train)
    y_pred = mejor_svc.predict(X_test)
    
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

