from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier
from KNN import KNN_Generator
from decisionTreeEntropy import TreeEntropy
from decisionTreeGini import TreeGini
from KMeans import algorithmKMeans
from MLP import algorithmMLP

base = "./bases/iris.data"

dataset = pds.read_csv(base, header=None)

index_Y = 0
index_inicial = 1
index_final = len(dataset.columns)

y =  dataset.loc[:, index_final-1] # extrai a primeira coluna, que é o label
X = dataset.loc[:,index_Y:index_final-2]
X.head(5)

# 20% teste e 80% treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

# KNN
KNN_Generator(X_train, X_test, y_train, y_test, "euclidean", "iris")
KNN_Generator(X_train, X_test, y_train, y_test, "manhattan", "iris")
KNN_Generator(X_train, X_test, y_train, y_test, "minkowski", "iris")

#Árvore
TreeEntropy(X_train, X_test, y_train, y_test,"iris")
TreeGini(X_train, X_test, y_train, y_test, "iris")

#K-Means
algorithmKMeans(X_train, X_test, y_train, y_test,"iris")

# MLP
algorithmMLP(X_train, X_test, y_train, y_test,"iris","tanh",(6,4,2))
algorithmMLP(X_train, X_test, y_train, y_test,"iris","tanh",(5,3))

algorithmMLP(X_train, X_test, y_train, y_test,"iris","logistic",(6,4,2))
algorithmMLP(X_train, X_test, y_train, y_test,"iris","logistic",(5,3))