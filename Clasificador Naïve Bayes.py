# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:26:09 2024

@author: María Eugenia Ramiro Gutiérrez
"""

#Importamos las librerías
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#Estamos importando nuestra base de datos, para su correcta lectura añadimos delimiter = ';'
df = pd.read_csv('C:/Users/maria/OneDrive/Escritorio/Ingeniería Informática/2º CARRERA/Sistemas inteligentes/predict+students+dropout+and+academic+success/data.csv', delimiter=';')

# Importamos las librerías para el modelo de la validación cruzada, Naïve Bayes, y además importamos las métricas que vamos a usar
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Seleccionamos las columnas que son atributos (X), y después la columna objetivo (y)  
X = df.iloc[:,0:-1]  
y = df.iloc[:,-1]  

model = GaussianNB()

n_divisiones = 10 #10 divisiones
kf = KFold(n_divisiones, shuffle=True, random_state=2)

# Ahora vamos a crear como una lista donde vamos a guardar los valores de cada una de nuestras métricas en cada una de las 10 iteraciones
valor_precision = []
valor_f1 = []
valor_exactitud = []


for train_ind, test_ind in kf.split(X, y):
    #Renombramos
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
    
    model.fit(X_train, y_train) # Ajusta el modelo al training data
    y_pred = model.predict(X_test) #Predicciones
    
    precision = precision_score(y_test, y_pred, average='weighted') #Como nuestra base de datos no es binaria tomo como average='weighted'
    f1 = f1_score(y_test, y_pred, average='weighted')
    exactitud = accuracy_score(y_test, y_pred)

    
    # Guardamos los valores en las listas previamente inicializadas
    valor_precision.append(precision)
    valor_f1.append(f1)
    valor_exactitud.append(exactitud)

#Imprimimos nuestras listas con los valores ya establecidos
print (valor_precision)
print (valor_f1)
print (valor_exactitud)

# A continuación calculamos la media de de cada una de las medidas de rendimiento
precision_mean = np.mean(valor_precision)
f1_mean = np.mean(valor_f1)
exactitud_mean = np.mean(valor_exactitud)

#Imprimimos la media de cada métrica
print("La media de la métrica precision es:", precision_mean)
print("La media de la métrica f1 es:", f1_mean)
print("La media de la métrica accuracy es:", exactitud_mean)