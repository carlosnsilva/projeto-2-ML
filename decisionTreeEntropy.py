from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier
    
def TreeEntropy(X_train, X_test, y_train, y_test, base):
    # Treinando a árvore

    model = tree.DecisionTreeClassifier(criterion="entropy")
    model = model.fit(X_train, y_train)

    resultado = model.predict(X_test)

    result_final = metrics.accuracy_score(resultado, y_test)

    final = round(result_final * 100)
    
    print("Resultado da árvore de decisão na base {} com criterio entropy: {}%\n".format(base,final))
    
    return 
