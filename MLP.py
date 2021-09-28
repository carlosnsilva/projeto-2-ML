from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.neural_network import MLPClassifier

def algorithmMLP(X_train, X_test, y_train, y_test, base, arch):
    model = MLPClassifier(hidden_layer_sizes=(6,4,2), activation=arch, max_iter=3000)
    model = model.fit(X_train, y_train)

    result = model.predict(X_test)

    acc = metrics.accuracy_score(result, y_test)

    show = round(acc * 100)
    print("Resultado para a base {}, utilizando a arquitetura {} foi de : {}%\n".format(base,arch,show))

    classes = model.classes_
    camadas = model.n_layers_
    saidas = model.n_outputs_

    vetorPeso = len(model.coefs_)
    pesoCamada = model.coefs_[0]
    bias = len(model.intercepts_)
    pesoCamadaEntrada = model.intercepts_[0]

    print("Quantidade de classes: ", classes)
    print("Quantidade de camadas: ", camadas)
    print("Quantidade de saídas: ", saidas)
    print("Tamanho do vetor de pesos: ", vetorPeso)
    print("Pesos da camada de entrada:\n ", pesoCamada)
    print("Tamanho do vetor dos limiares: ", bias)
    print("Pesos da camada de entrada: ", pesoCamadaEntrada)
    print()
    return