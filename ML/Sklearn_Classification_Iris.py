from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from enum import Enum    

#---------------------- Enums ---------------------

class Model(Enum):
    KNN = 1
    LOG_REG = 2

#---------------------- Consts ---------------------

TEST_SIZE = 0.2

#--------------------- Inner imp ----------------------

def InitModel(_model):
    if _model==Model.KNN:
        model = KNeighborsClassifier()
    elif _model==Model.LOG_REG:
        model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=11111)
    
    return model

#--------------------- APIs ------------------------

def Prediction(_dataSet, _toPredict, _useModel):
    X = _dataSet.data
    y = _dataSet.target

    model = InitModel(_useModel)

    #fit (train) 
    model.fit(X,y)

    #predict
    prediction = model.predict(_toPredict)

    #return prediction
    return prediction

#--------------------------------------------------

def PredictionAccuracy(_dataSet, _useModel, _testSize):
    X = _dataSet.data
    y = _dataSet.target
    #split DS
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = _testSize)

    model = InitModel(_useModel)

    #fit (train) 
    model.fit(X_train,y_train)

    #predict 
    y_prediction = model.predict(X_test)

    #calc accuracy
    accuracy = metrics.accuracy_score(y_test, y_prediction)

    #return prediction
    return accuracy

#---------------------- main ----------------------

print('Here we GO!\n')	

#load dataset
iris = load_iris()

#excute model for prediction
toPredict = [[2,4,3,1]]
predictionKnn = Prediction(iris, toPredict, Model.KNN)
predictionLogReg = Prediction(iris, toPredict, Model.LOG_REG)

#print prediction results
print("Results for prediction :")
print("Knn = " + str(iris.target_names[predictionKnn])  )
print("LogReg = " +  str(iris.target_names[predictionLogReg]) + "\n")

#excute model for calc accuracy
accurKnn = PredictionAccuracy(iris, Model.KNN, TEST_SIZE)
accurLogReg = PredictionAccuracy(iris, Model.KNN, TEST_SIZE)

#print accuracy results
print("Results for accuracy :")
print("Knn = " + str(accurKnn) )
print("LogReg = " + str(accurLogReg) + "\n")

#--------------------------------------------------
