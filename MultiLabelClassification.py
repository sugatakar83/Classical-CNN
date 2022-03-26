import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy import asarray
from sklearn.multioutput import ClassifierChain


# create train and test dataset
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')
print(train_csv.shape, test_csv.shape)


# remove the rows where clearsky DHI > 400 and wind speed >11 as they seems to be outliers
train_csv.drop(train_csv[train_csv['Clearsky DHI'] > 400].index, inplace=True)
train_csv.drop(train_csv[train_csv['Wind Speed'] > 11].index, inplace=True)
X1 = test_csv.drop(["Year","Month","Day","Hour","Minute","Clearsky DHI","Clearsky DNI","Clearsky GHI"], axis=1).to_numpy()


# create the X train and y train
def get_dataset():
    y = train_csv[["Clearsky DHI","Clearsky DNI","Clearsky GHI"]]
    X = train_csv.drop(["Year","Month","Day","Hour","Minute","Clearsky DHI","Clearsky DNI","Clearsky GHI"], axis=1)
    return X.to_numpy(), y.to_numpy()



# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(40, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        base_lr = LogisticRegression(solver='saga', random_state=0)
        chain = ClassifierChain(base_lr, order='random', random_state=0)
        # model = get_model(n_inputs, n_outputs)
        # fit model
        # model.fit(X_train, y_train, verbose=0, epochs=5)
        chain.fit(X_train, y_train).predict(X_test)
        # make a prediction on the test set
        # yhat = model.predict(X_test)
        yhat = chain.predict_proba(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        yhatr = np.argmax(yhat, axis=1)
        ytest = np.argmax(y_test, axis=1)
        acc = accuracy_score(ytest, yhatr)
        # store result
        print('>%.3f' % acc)
        results.append(acc)
    return results


# load dataset
X, y = get_dataset()
# evaluate model
results = evaluate_model(X, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))


# load dataset
X, y = get_dataset()
n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, y, verbose=0, epochs=2)
# make a prediction for new data
# row = [2009, 3, 6, 7, 8, 7, 19, 17, 1100, 91.1, 105, 3, 200, 2.7, 100]
for row in X1:
    newX = asarray([row])
    yhat = model.predict(newX)
    print('Predicted: %s' % yhat[0])

