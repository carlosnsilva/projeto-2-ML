from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

def KNN_Generator(X_train, X_test, y_train, y_test, metrica, base):
        k = [5,10] 

        for i in k:

            model = KNeighborsClassifier(n_neighbors=i, metric=metrica, algorithm='brute')
            model = model.fit(X_train, y_train)

            result = model.predict(X_test)

            acc = metrics.accuracy_score(result, y_test)

            show = round(acc * 100)
            print("Resultado do Algoritmo KNN para a base {} com a metrica {} e k = {}: {}%\n".format(base,metrica,i,show))
        
        return