from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier
from KNN import KNN_Generator
from decisionTreeEntropy import TreeEntropy
from decisionTreeGini import TreeGini
from KMeans import algorithmKMeans
from MLP import algorithmMLP

base = "./bases/balance-scale.data"

dataset = pds.read_csv(base, header=None)

index_Y = 0
index_inicial = 1
index_final = len(dataset.columns)

y = dataset[index_Y] # extrai a primeira coluna, que é o label
X = dataset.loc[:,index_inicial:index_final-1]
X.head(5)

# 20% teste e 80% treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

# KNN
KNN_Generator(X_train, X_test, y_train, y_test, "euclidean", "balance-scale")
KNN_Generator(X_train, X_test, y_train, y_test, "manhattan", "balance-scale")
KNN_Generator(X_train, X_test, y_train, y_test, "minkowski", "balance-scale")

#Árvore
TreeEntropy(X_train, X_test, y_train, y_test,"balance-scale")
TreeGini(X_train, X_test, y_train, y_test, "balance-scale")

#K-Means
algorithmKMeans(X_train, X_test, y_train, y_test,"balance-scale")

# MLP
algorithmMLP(X_train, X_test, y_train, y_test,"balance-scale","tanh",(6,4,2))
algorithmMLP(X_train, X_test, y_train, y_test,"balance-scale","tanh",(5,3))

algorithmMLP(X_train, X_test, y_train, y_test,"balance-scale","logistic",(6,4,2))
algorithmMLP(X_train, X_test, y_train, y_test,"balance-scale","logistic",(5,3))