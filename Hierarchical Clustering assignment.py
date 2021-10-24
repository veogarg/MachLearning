# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:15:44 2021

@author: Nishan Kapoor
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv(r"E:\clustering assignments\crime_data.csv")
df
type(df)
plt.bar(height = df.Rape, x = np.arange(5, 50, 5)) # initializing the parameter
plt.hist(df.Murder)
plt.boxplot(df.Assault)

df.describe()
df.info()
df["State"].value_counts().head()

# iloc means accessing the rows and columns by numbers
df.iloc[0:5, 0:4]
df.iloc[1:3, [2,4]]
df.iloc[:, 1:]

df.loc[:, ["Murder","Rape"]]
# or we can simply write the following command
df[["Assault","UrbanPop"]]
df.describe()



df.Murder = df.Murder.astype('int64') 
df.dtypes

df.Rape = df.Rape.astype('int64')
df.dtypes

dfnew = df.drop(["State"], axis=1)
dfnew.info()

import pandas as pd
import matplotlib.pylab as plt


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(df.iloc[:, 1:])
df.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Crime')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 8 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(dfnew) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df['clust'] = cluster_labels # creating a new column and assigning it to new column 

df = df.iloc[:, [0,1,2,3,4,5]]
df.head()

# Aggregate mean of each cluster
df.iloc[:, 0:].groupby(df.clust).mean()

# creating a csv file 
df.to_csv("crime_data.csv", encoding = "utf-8")

import os
os.getcwd()


##########################################################

##########################################################



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

airways = pd.read_csv(r"E:\clustering assignments\airlines dataset.csv")
airways.describe()
airways.info()
airways.head()

#check null value
airways.isnull().sum()

# drop columns
airways1 = airways.drop(["ID#","Balance"], axis = 1)
airways1.describe()
plt.hist(airways1.Balance)

airways1.head()
airways1.shape

#from sklearn.preprocessing import MinMaxScaler
#import numpy as np


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
airways2_norm = norm_func(airways1.iloc[:, 0:])
airways2_norm.describe()


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 

plt.figure(figsize = (20,6))

z = linkage(airways2_norm, method="complete", metric="euclidean")

import seaborn as sns
sns.boxplot(z)


# Dendrogram
plt.figure(figsize=(30, 15));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,
    leaf_font_size = 10
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(airways2_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

# create new column to store cluster
airways['clust'] = cluster_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
airways1.iloc[:, 2:].groupby(airways1.clust).mean()

# creating a csv file 
airways1.to_csv(r"E:\clustering assignments\airlines dataset.csv",encoding = "utf-8")

import os
os.getcwd()
################################################################################
################################################################################

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

df = pd.read_excel(r"E:\clustering assignments\Telco_customer_churn.xlsx")
type(df)

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


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_new_1, method = "complete", metric = "euclidean")

import seaborn as sns
sns.boxplot(z)

# Dendrogram
plt.figure(figsize=(30, 15));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,
    leaf_font_size = 10
)
plt.show()

from sklearn.cluster import AgglomerativeClustering
#use AgglomerativeClustering clustering


h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df2_norm) 
h_complete.labels_


cluster_labels = pd.Series(h_complete.labels_)
#creat new column to store cluster
df1['clust'] = cluster_labels # creating a new column and assigning it to new column 




#creat group of cluster
df1.iloc[:, 2:].groupby(df1.clust).mean()

#store data to new csv
df1.to_excel(r"C:\Users\pushk\Desktop\Data Science\08_Data Mining Unsupervised Learning-Hierarchical Clustering\New folder\Telco_customer_churn.xlsx", encoding = "utf-8")




##############################################################################
##############################################################################
import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv(r"E:\clustering assignments\AutoInsurance.csv")

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



# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0, 
    leaf_font_size = 10 
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df1['clust'] = cluster_labels # creating a new column and assigning it to new column 

# Aggregate mean of each cluster
df.iloc[:, 2:].groupby(df.clust).mean()

# creating a csv file 
df1.to_csv(r"E:\clustering assignments\AutoInsurance.csv", encoding = "utf-8")

import os
os.getcwd()

#############################################################################













