import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#-------------------------------------------

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#-------------------------------------------

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#-------------------------------------------

def my_split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#-------------------------------------------

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

 #-------------------------------------------

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

#attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
#housing_extra_attribs = attr_adder.transform(housing.values)

#-------------------------------------------

def PrintRmse(y, predictions):
    lin_mse = mean_squared_error(y, predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Avarage RMSE error is : ", lin_rmse)

#-------------------------------------------

def DisplayScores(model, X, y):
    scores = cross_val_score(model, X, y,scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    print("Scores:", tree_rmse_scores)
    print("Mean:", tree_rmse_scores.mean())
    print("Standard deviation:", tree_rmse_scores.std())

#-------------------------------------------
#main
#-------------------------------------------

#---------------- Get data  ---------------------
print("\n*** Fetching data set from github...")
#fetch_housing_data()

print("\n*** Load data set from csv...\n")
houseDataSet = load_housing_data()

print("\n*** Dataset Info : ")
print(houseDataSet.info())
print(houseDataSet.head())

print("\n*** Show data hist : ")
#houseDataSet.hist(bins=50, figsize=(20,15))
#plt.show()

print("\n*** Create test set : ")
#train_set, test_set = my_split_train_test(houseDataSet, 0.2)
train_set, test_set = train_test_split(houseDataSet, test_size=0.2, random_state=42)
print("\n*** train len= " + str(len(train_set)) + " *** test len=" + str(len(test_set)))

#---------------- Explor data  ---------------------

print("\n*** Explor and visualize the data : ")
#houseDataSet.plot(kind="scatter", x="longitude", y="latitude")
#houseDataSet.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#houseDataSet.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#    s=houseDataSet["population"]/100, label="population", figsize=(10,7),
#    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
#plt.legend()
#plt.show()

print("\n*** Corralations : ")
corr_matrix = houseDataSet.corr()
corr = corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr)

print("\n*** Corralations visualize : ")
#attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
#scatter_matrix(houseDataSet[attributes], figsize=(12, 8))
#plt.show()
#houseDataSet.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
#plt.show()

#---------------- Data for model  ---------------------

X = train_set.drop("median_house_value", axis=1)
y = train_set["median_house_value"].copy()
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

#---------------- Clean data  ---------------------

print("\n*** Use pipelines to Clean and prepare data for model : ")

housing_num = X.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

#pipeline for nums
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),      #fill empty cells
        #('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),                   #scale data
    ])

#pipeline for text
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder()),
    ])

#merge pipelines
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

#execute pipelines for train and test
X_prepared = full_pipeline.fit_transform(X)
print(X_prepared.shape)
print(X_prepared)

X_test_prepared = full_pipeline.fit_transform(X_test)

#---------------- train model  ---------------------

lin_reg = LinearRegression()
lin_reg.fit(X_prepared, y)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_prepared, y)

forest_reg = RandomForestRegressor()
forest_reg.fit(X_prepared, y)

#---------------- Predict house from train  ---------------------

print("\n*** Predict house from train : ")
some_data_prepared = X_prepared[0]
some_labels = y.iloc[:1]

prediction_linReg = lin_reg.predict(some_data_prepared)
prediction_treeReg = tree_reg.predict(some_data_prepared)
prediction_forestReg = forest_reg.predict(some_data_prepared)
print("Predictions:", prediction_linReg)
print("Predictions:", prediction_treeReg)
print("Predictions:", prediction_forestReg)
print("Labels:", list(some_labels))

#---------------- Predict house from test  ---------------------

print("\n*** Predict house from test : ")
some_data_prepared = X_test_prepared[0]
some_labels = y_test.iloc[:1]

prediction_linReg = lin_reg.predict(some_data_prepared)
prediction_treeReg = tree_reg.predict(some_data_prepared)
prediction_forestReg = forest_reg.predict(some_data_prepared)
print("Predictions: ", prediction_linReg)
print("Predictions: ", prediction_treeReg)
print("Predictions: ", prediction_forestReg)
print("Labels: ", list(some_labels))

#---------------- Calculate error on training set ---------------------

print("\n*** Calculate RMSE error on training set : ")
print("linear")
predictions = lin_reg.predict(X_prepared)
PrintRmse(y, predictions)
#DisplayScores(lin_reg, X_prepared, y)
print("tree")
predictions = tree_reg.predict(X_prepared)
PrintRmse(y, predictions)
#DisplayScores(tree_reg, X_prepared, y)
print("forest")
predictions = forest_reg.predict(X_prepared)
PrintRmse(y, predictions)
#DisplayScores(forest_reg, X_prepared, y)

#---------------- Calculate error on test set ---------------------

print("\n*** Calculate RMSE error on test set : ")
print("linear")
predictions = lin_reg.predict(X_test_prepared)
PrintRmse(y_test, predictions)
#DisplayScores(lin_reg, X_test_prepared, y_test)
print("tree")
predictions = tree_reg.predict(X_test_prepared)
PrintRmse(y_test, predictions)
#DisplayScores(tree_reg, X_test_prepared, y_test)
print("forest")
predictions = forest_reg.predict(X_test_prepared)
PrintRmse(y_test, predictions)
#DisplayScores(forest_reg, X_test_prepared, y_test)

#---------------- Random forest preform best - find best params ---------------------

#print("\n*** Find best params for Random forest : ")
#param_grid = [
#    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}
#  ]

#rand_search = RandomizedSearchCV(forest_reg, cv=5, scoring='neg_mean_squared_error', param_distributions=param_grid)
#rand_search.fit(X_prepared, y)
#print(rand_search.best_params_)
