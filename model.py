import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('MallCustomers.csv')

# first 5 rows in the dataframe
customer_data.head()

#finding the no of rows and columns
customer_data.shape

# getting some informations about the dataset
customer_data.info()

# checking for missing values
customer_data.isnull().sum()

X = customer_data.iloc[:,[3,4]].values

# finding wcss value for different number of clusters
#WCSS = Within Clusters Sum of Squares

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)

def result(income,spending):
    test = np.array([income,spending])
    result = kmeans.predict([test])
    print(result)
    return result