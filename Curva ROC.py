# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:43:00 2024

@author: María Eugenia
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

# Establecer la semilla
np.random.seed(42)

#Estamos importando nuestra base de datos, para su correcta lectura añadimos delimiter = ';'
df = pd.read_csv('C:/Users/maria/OneDrive/Escritorio/Ingeniería Informática/2º CARRERA/Sistemas inteligentes/predict+students+dropout+and+academic+success/data.csv', delimiter=';')

#Seleccionamos las columnas que son atributos (X), y después la columna objetivo (y)  
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

# X e y tienen que ser arrays numpy
X = np.array(X)
y = np.array(y)

clases = np.unique(y)
y_binary = label_binarize(y, classes=clases)
n_clases = y_binary.shape[1]

# Inicializar clasificadores con OneVsRestClassifier
clasificadores = {
    'GaussianNB': OneVsRestClassifier(GaussianNB()),
    'KNeighbors': OneVsRestClassifier(KNeighborsClassifier()),
    'DecisionTree': OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
    'SVC': OneVsRestClassifier(SVC(probability=True, random_state=42))
}

n_divisiones = 10 #10 divisiones
kf = KFold(n_divisiones, shuffle=True, random_state=42)

roc_auc_scores = {nombre: [] for nombre in clasificadores}
accuracy_scores = {nombre: [] for nombre in clasificadores}
precision_scores = {nombre: [] for nombre in clasificadores}
f1_scores = {nombre: [] for nombre in clasificadores}

# Realizar validación cruzada
for train_ind, test_ind in kf.split(X):
    X_train, X_test = X[train_ind], X[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]
    
    y_train_binary = label_binarize(y_train, classes=clases)
    y_test_binary = label_binarize(y_test, classes=clases)
    
    for nombre, clf in clasificadores.items():
        clf.fit(X_train, y_train_binary)
        y_score = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        
        # Calcular métricas
        fpr, tpr, _ = roc_curve(y_test_binary.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        accuracy = accuracy_score(y_test_binary, y_pred)
        precision = precision_score(y_test_binary, y_pred, average='weighted')
        f1 = f1_score(y_test_binary, y_pred, average='weighted')
        
        roc_auc_scores[nombre].append(roc_auc)
        accuracy_scores[nombre].append(accuracy)
        precision_scores[nombre].append(precision)
        f1_scores[nombre].append(f1)

# Mostrar resultados
for nombre in clasificadores:
    print(f"Clasificador: {nombre}")
    print(f"Media ROC AUC: {np.mean(roc_auc_scores[nombre]):.2f}")
    print(f"Media Accuracy: {np.mean(accuracy_scores[nombre]):.2f}")
    print(f"Media Precision: {np.mean(precision_scores[nombre]):.2f}")
    print(f"Media F1 Score: {np.mean(f1_scores[nombre]):.2f}\n")

def plot_roc_curve_manual(clasificadores, X_test, y_test_binary):
    plt.figure()
    
    # Colores para las curvas ROC
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
    
    # Iterar sobre clasificadores y graficar sus curvas ROC
    for i, (nombre, clf) in enumerate(clasificadores.items()):
        y_score = clf.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test_binary.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'curva ROC de {nombre} (area = {roc_auc:0.2f})')
    
    # Configuración final del gráfico
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC para clasificación multiclase')
    plt.legend(loc="lower right")
    plt.show()

# Llamada a la función con los clasificadores y los datos de prueba
plot_roc_curve_manual(clasificadores, X_test, y_test_binary)