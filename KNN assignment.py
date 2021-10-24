# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:24:07 2021

@author: Nishan Kapoor
"""
######### Question 1 ###########

import pandas as pd
import numpy as np

glass = pd.read_csv(r"E:\KNN Assignment\glass.csv")

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
glass_n = norm_func(glass.iloc[:, :9])
glass_n.describe()

X = np.array(glass_n.iloc[:,:]) # Predictors 
Y = np.array(glass['Type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable acc means accuracy
acc = []

# running KNN algorithm for 3 to 29 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,29,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,29,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,29,2),[i[1] for i in acc],"bo-")


##############################################################################

######## Question 2 ###########

import pandas as pd
import numpy as np

zoo = pd.read_csv(r"E:\KNN Assignment\Zoo.csv")


zoo = zoo.iloc[:, 1:] # Excluding animal name column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data), excluding type column as that is our target column
zoo_n = norm_func(zoo.iloc[:, :16])
zoo_n.describe()

X = np.array(zoo_n.iloc[:,:]) # Predictors 
Y = np.array(zoo['type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable acc means accuracy
acc = []

# running KNN algorithm for 3 to 25 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,25,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,25,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,25,2),[i[1] for i in acc],"bo-")

########### completed ######################


