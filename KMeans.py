from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
import pandas as pd

def algorithmKMeans(X_train, X_test, y_train, y_test,base):
    myset = set(y_train)
    clusters = len(myset)

    model = KMeans(n_clusters = clusters)
    model = model.fit(X_train)

    # Pegar os labels dos padrões de Treinamento
    labels = model.labels_

    map_labels = []

    for i in range(clusters):
        map_labels.append([])

    new_y_train = y_train.to_list()

    for i in range(len(y_train)):
        for c in range(clusters):
            if labels[i] == c:
                map_labels[c].append(new_y_train[i])

    #print(map_labels)

    # Criar dicionário com os labells a serem mapeados
    mapping = {}

    for i in range(clusters):
        final = Counter(map_labels[i]) # contar a classe que mais aparece
        value = final.most_common(1)[0][0] # retorna a classe com maior frequência
        mapping[i] = value


    result = model.predict(X_test)
    result = [mapping[i] for i in result]

    acc = metrics.accuracy_score(result, y_test)
    show = round(acc * 100)
    print("K-Means na Base {}, resultado: {}%".format(base,show))
    print()
    return
