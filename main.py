#This program implements a multi-layer perceptron using sklearn library to predict diabetics 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import matplotlib.pyplot as plt


diabetes = pd.read_csv("diabetes.csv")
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'],diabetes['Outcome'], test_size = 0.2, random_state=66)


#create a MLP having 3 hidden layers of 100 neurons, alpha is the learning rate and solver is stochastic gradient descent

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy=",accuracy_score(y_test, y_pred))




filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

print([6,148,72,35,0,33.6,0.627,50],"->",clf.predict([[6,148,72,35,0,33.6,0.627,50]]))


