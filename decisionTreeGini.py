from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pds
from sklearn.neighbors import KNeighborsClassifier

def TreeGini(X_train, X_test, y_train, y_test, base):
    
    model = tree.DecisionTreeClassifier(criterion="gini")
    model = model.fit(X_train, y_train)

    resultado = model.predict(X_test)

    result_final = metrics.accuracy_score(resultado, y_test)

    final = round(result_final * 100)

    print("Resultado da árvore de decisão {} com o criterio gini {}% \n".format(base,final))

    
    return

