#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import csv
import math
import sys


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


import scipy.cluster.hierarchy as sch


# In[4]:


PATH = os.getcwd()
DATA = "data_1.csv"
ORIGINAL_DATA = "data.csv"

SINGLE_LINKAGE = "single"
COMPLETE_LINKAGE = "complete"


# In[5]:


def load_data():
    data_path = os.path.join(PATH , DATA)
    df = pd.read_csv(data_path, header = None)
    data = df.to_numpy()
    return data


# In[6]:


def load_original_data():
    data_path = os.path.join(PATH , ORIGINAL_DATA)
    df = pd.read_csv(data_path, header = None)
    data = df.to_numpy()
    
    X = data[:,:-1]
    Y = data[:,-1:]
    return X, Y


# In[7]:


class Hierarchical_Clustering(object):
    def __init__(self, x, cluster_algo):
        print("Clustering algorihtm : ", cluster_algo)
        self.x = x
        self.cluster_algo = cluster_algo
        
        for i in range (2, 5):
            print("--------------No of cluster : ", i, " ----------------")
            self.no_of_clusters = i
            # Clustering
            self.clustering()
    
    def clustering(self):
        # Distance Matrix
        dist_matrix = self.get_distance_matrix(self.x)
        #print(dist_matrix)
        
        # Set diagonal = infinity 
        np.fill_diagonal(dist_matrix, sys.maxsize)
        clusters = self.find_clusters(dist_matrix)
        
        # Get n clusters  
        # Saved in clusters array in bakcward 
        n = dist_matrix.shape[0] - self.no_of_clusters
        output = clusters[n]
        print("Output : ", output)
        
        # Get individual cluster
        arr = np.unique(output)
        n_clusters = []
        for x in np.nditer(arr):
            n_clusters.append(np.where(output == x))
        
        for cls in range(0, len(n_clusters)):
            print("Cluster-", cls + 1, " : ", n_clusters[cls][0])
        
        # Plot clusters
        self.plot_cluster(n_clusters)
        
        # Calulate error
        self.calculate_error(n_clusters)
        print("\n\n")
        
    def find_clusters(self, dist_matrix):
        clusters = {}
        cluster_idx = []
        row_idx = -1
        col_idx = -1
        
        # Cluster index, [0, 1, 2, ..., 13]
        for n in range(dist_matrix.shape[0]):
            cluster_idx.append(n)
        #print(cluster_idx)
        
        clusters[0] = cluster_idx.copy()
        merge_cluster = [[i] for i in range(dist_matrix.shape[0])]
        #print(merge_cluster)
        
        # Min from the distance matrix
        for k in range(1, dist_matrix.shape[0]):
            min_val = sys.maxsize
            
            for i in range(0, dist_matrix.shape[0]):
                for j in range(0, dist_matrix.shape[1]):
                    if(dist_matrix[i][j] <= min_val):
                        min_val = dist_matrix[i][j]
                        row_idx = i
                        col_idx = j
            #print("Min value : ", min_val)
            
            # Merge cluster
            #print("Merged clusters - ")
            #print("Cluster 1 : ", merge_cluster[row_idx])
            #print("Cluster 2 : ", merge_cluster[col_idx])
            merge_cluster[row_idx].append(merge_cluster[col_idx])
            merge_cluster[col_idx] = merge_cluster[row_idx]
            #print(merge_cluster)
            #print("\n")
            
            # Update the distance matrix
            # Calculating the new distances from the merged cluster to all points
            for i in range(0,dist_matrix.shape[0]):
                if(i != col_idx):
                    # Min for Single Linkage
                    # Max for Complete Linkage
                    if(self.cluster_algo == SINGLE_LINKAGE):
                        temp = min(dist_matrix[col_idx][i],dist_matrix[row_idx][i])
                    else:
                        temp = max(dist_matrix[col_idx][i],dist_matrix[row_idx][i])
                    
                    # Symmetric update of distance matrix
                    dist_matrix[col_idx][i] = temp
                    dist_matrix[i][col_idx] = temp
            
            for i in range (0,dist_matrix.shape[0]):
                dist_matrix[row_idx][i] = sys.maxsize
                dist_matrix[i][row_idx] = sys.maxsize
            #print("dist_matrix")
            #print(dist_matrix)
            
            # Cluster formation, clusters 
            # if k=0,then all datapoints are clusters
            minimum = min(row_idx,col_idx)
            maximum = max(row_idx,col_idx)
            #print("minimum", minimum)
            #print("maximum", maximum)
            for n in range(len(cluster_idx)):
                if(cluster_idx[n] == maximum):
                    cluster_idx[n] = minimum
            clusters[k] = cluster_idx.copy()
            #print(cluster_idx)
            
        #print("clusters")
        #print(clusters)
        return clusters
    
    # Distance matrix
    def get_distance_matrix(self, x):
        dist_matrix = np.zeros((len(x),len(x)))
        for i in range(0, dist_matrix.shape[0]):
            for j in range(0, dist_matrix.shape[0]):
                dist_matrix[i][j] = self.get_euclidean_distance(x[i], x[j])
        return dist_matrix
    
    # Euclidean distance
    def get_euclidean_distance(self, x1, x2):
        d = 0.0
        for i in range(0, len(x1)):
            d = d + (x1[i] - x2[i]) ** 2
        #print(d)
        return math.sqrt(d)
        
    def plot_cluster(self, n_clusters):
        fig = plt.figure()
        fig.suptitle('Scatter Plot')
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Height')
        ax.set_ylabel('Weight')
        
        p=0
        # Color for scatter plot blobs
        color = ['r','g','b','y','c','m','k','w']
        for i in range(0,len(n_clusters)):
            for j in np.nditer(n_clusters[i]):
                   ax.scatter(self.x[j,0], self.x[j,1], c = color[p])
            p = p + 1

        plt.show()
        
    def calculate_error(self, n_clusters):
        print("\nError Calculation")
        x, y = load_original_data()
        
        #print(n_clusters)
        total_error = 0
        total_percentage = 0
        for cls in range(0, len(n_clusters)):
            sub_cluster = n_clusters[cls][0]
            print("Cluster-", cls + 1, " : ", np.take(y, sub_cluster))
            
            # Get men/women count from each sub-cluster
            men = 0
            women = 0
            for i in range(0, len(sub_cluster)):
                idx = sub_cluster[i]
                if(y[idx] == 'M'):
                    men += 1
                else:
                    women += 1
            print("Men : ", men, ", Women : ", women)
            if(men > women):
                freq_class = 'M'
                freq_count = men
            else:
                freq_class = 'W'
                freq_count = women
            
            error = len(sub_cluster) - int(freq_count)
            percentage = float((len(sub_cluster) - int(freq_count)) / len(sub_cluster)) * 100
            print("Error count : ", error, ", Error percentage : ", percentage, "%")
            total_error += error
            total_percentage += percentage
        
        print("Total error : ", total_error, ", Total error percentage : ", total_percentage, "%")


# In[8]:


def main():
    
    X = load_data()
    Hierarchical_Clustering(X, SINGLE_LINKAGE)
    Hierarchical_Clustering(X, COMPLETE_LINKAGE)
        
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




