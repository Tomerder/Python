from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from enum import Enum    

# ---------------------- Enums ---------------------
class Model(Enum):
    KNN = 1
    LOG_REG = 2
# --------------------------------------------------

def Prediction(_dataSet, _toPredict, _useModel):
    X = _dataSet.data
    y = _dataSet.target

    if _useModel==Model.KNN:
        model = KNeighborsClassifier()
    elif _useModel==Model.LOG_REG:
        model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=11111)

    #fit (train) 
    model.fit(X,y)

    #predict with KNN and logReg
    prediction = model.predict(_toPredict)

    #return prediction
    return prediction

# ---------------------- main ----------------------
print('Here we GO!')	

#load dataset
iris = load_iris()

#excute model for prediction
toPredict = [[2,4,3,1]]
predictionKnn = Prediction(iris, toPredict, Model.KNN)
predictionLogReg = Prediction(iris, toPredict, Model.LOG_REG)

#print results
print(iris.target_names[predictionKnn])
print(iris.target_names[predictionLogReg])

# --------------------------------------------------
