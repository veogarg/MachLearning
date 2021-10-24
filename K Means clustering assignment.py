# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 13:36:03 2021

@author: Royal computer
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:20:19 2021

@author: Nishan Kapoor
"""
###  question 2

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on crime Data set 
crime = pd.read_csv((r"E:\K means datasets\crime_data (1).csv"))

crime.describe()
crime1 = crime.drop(["State"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime1.iloc[:, 0:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 6))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime1['clust'] = mb # creating a  new column and assigning it to new column 

crime1.head()
df_norm.head()

crime1 = crime1.iloc[:, :]
crime1.head()

crime1.iloc[:, 2:6].groupby(crime1.clust).mean()
crime1.to_csv(r"E:\K means datasets\"crime_data (1).csv", encoding = "utf-8")

import os
os.getcwd()


##############################################################################
##############################################################################

  #  question 1

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y="Y", kind = "scatter")

model1 = KMeans(n_clusters = 5).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on airlines Data set 
airways1 = pd.read_excel(r"E:\K means datasets\EastWestAirlines (1).xlsx")

airways1.describe()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airways1.iloc[:, 0:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airways1['clust'] = mb # creating a  new column and assigning it to new column 

airways1.head()
df_norm.head()

airways1 = airways1.iloc[:, :]
airways1.head()

airways1.iloc[:, 0:12].groupby(airways1.clust).mean()

airways1.to_excel(r"E:\K means datasets\EastWestAirlines (1).xlsx", encoding = "utf-8")

import os
os.getcwd()

##############################################################################
##############################################################################

   #  question 3

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 


# Kmeans on Insurance Data set 
insurance1 = pd.read_csv(r"E:\K means datasets\Insurance Dataset.csv")

insurance1.describe()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(insurance1.iloc[:, 0:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ 
# getting the labels of clusters assigned to each row 

mb = pd.Series(model.labels_)
# converting numpy array into pandas series object 

insurance1['clust'] = mb
# creating a  new column and assigning it to new column 

insurance1.head()
df_norm.head()

insurance1 = insurance1.iloc[:, :]
insurance1.head()

insurance1.iloc[:, :].groupby(insurance1.clust).mean()

import os
os.getcwd()
#####################################################################
#####################################################################

   # question 4

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 


# Kmeans on University Data set 
df = pd.read_excel(r"E:\K means datasets\Telco_customer_churn (1).xlsx")


df.describe()
df.info()
df.head()

#check null value
df.isnull().sum()

# drop column
df1 = df.drop(["Customer ID"], axis=1)
df1.describe()

# Normalization function 
# Range converts to: 0 to 1
def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)


# Create dummy variables
df_new = pd.get_dummies(df)
df_new_1 = pd.get_dummies(df, drop_first = True)
# we have created dummies for all categorical columns

# Normalized data frame (considering the numerical part of data)
df2_norm = norm_func(df_new_1.iloc[:, 1:])
df2_norm.describe()


###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df2_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df2_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column 

df1.head()
df2_norm.head()

df1 = df1.iloc[:,[14,0,1,2,3,4,5,6]]
df1.head()

df1.iloc[:, 2:14].groupby(df1.clust).mean()

df1.to_excel(r"E:\K means datasets\Telco_customer_churn (1).xlsx", encoding = "utf-8")

import os
os.getcwd()

##############################################################################
##############################################################################

     # question 5

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Kmeans on University Data set 
df = pd.read_csv(r"E:\K means datasets\AutoInsurance (1).csv")

df.describe()
df.info()
df.columns
df.head()
df1 = df.drop(["Customer","State"], axis=1)

#check null value
df.isnull().sum()

# Create dummy variables
df_new = pd.get_dummies(df)
df_new_1 = pd.get_dummies(df, drop_first = True)
# we have created dummies for all categorical columns

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
#df_norm = norm_func(df1.iloc[:, 1:])
#df_norm.describe()

df_norm = norm_func(df_new_1.iloc[:, 1:])
df_norm.describe()



###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 15))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df1['clust'] = mb # creating a  new column and assigning it to new column 

df1.head()
df_norm.head()

df1 = df1.iloc[:,[9,0,1,2,3,4,5,6]]
df1.head()

df1.iloc[:, 2:10].groupby(df1.clust).mean()

df1.to_csv(r"E:\K means datasets\AutoInsurance (1).csv", encoding = "utf-8")

import os
os.getcwd()
##############################################################################
# k means clustering assignment completed


























