# Script to read neon plant data and classify plant species using an SVM 

from sklearn.svm import SVC
import pandas as pd
import re
import numpy as np

fileName = "../data/neon_plants.csv"
df = pd.read_csv(fileName, skiprows = 0)

#Extract taxonid column and assign an integer to each unique species. This becomes y vector
def getTarget(df):
	species = df['taxonid']
	labels, levels = pd.factorize(species)
	y = labels
	return y

#Extract all columns to be used as features in the svm
def getFeatures(df):
	ident = 'nm_'
	names = ['chm_height']
	for column in df:
		if re.match(ident, column):
			names.append(column)
	X = df[names]
	X = X.as_matrix()
	return X

y = getTarget(df)
X = getFeatures(df)

#Normalize features
def featureNorm(X):
	X_shape = X.shape
	X_norm = X
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis = 0)
	for i in range(X_shape[1]):
		X_norm[:,i] = (X[:,i] - mu[i])/sigma[i]
	return X_norm

#Randomize row order
def randomizeVals(X, y):
	X_shape = X.shape
	arr = np.column_stack((X, y))
	np.random.shuffle(arr)
	X = arr[:,0:X_shape[1]]
	y = arr[:,-1]
	return(X, y)

#Seperate data into a training set, cross validation set and test set
def getSets(X, y):
	train_stop = round(.6 * y.size)
	cv_stop = y.size - round(.2 * y.size)
	X_train = X[0:train_stop, :]
	y_train = y[0:train_stop]
	X_cv = X[train_stop + 1:cv_stop, :]
	y_cv = y[train_stop + 1:cv_stop]
	X_test = X[cv_stop + 1:-1, :]
	y_test = y[cv_stop + 1:-1]
	return(X_train, X_test, X_cv, y_train, y_test, y_cv)

X = featureNorm(X)
(X, y) = randomizeVals(X, y)
(X_train, X_test, X_cv, y_train, y_test, y_cv) = getSets(X, y)

#optimize constant parameters of cost function on the cross validation set
def findParams(X_train, y_train, X_cv, y_cv):
	accuracy = 0
	params = np.array([.001, .01, .03, .1, .3, 1, 3, 10, 30, 100])
	for i in range(params.size):
		for j in range(params.size):
			C = params[i]
			gamma = params[j]
			clf = SVC(C=C, gamma=gamma, decision_function_shape='ovr')
			clf.fit(X_train, y_train)
			temp_acc = clf.score(X_test, y_test)
			if temp_acc > accuracy:
				clf_ideal = clf
				C_ideal = C
				gamma_ideal = gamma
				accuracy = temp_acc
	print("C:", C_ideal, "gamma:", gamma_ideal)
	return clf_ideal

#check accuracy
clf_ideal = findParams(X_train, y_train, X_cv, y_cv)
print("Accuracy:", clf_ideal.score(X_test, y_test))
