#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import csv
import math
import sys


import matplotlib.pyplot as plt


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# In[ ]:


PATH = os.getcwd()
OUTPUT_PATH = os.path.join(PATH, "Output/") 
PLOT_PATH = os.path.join(PATH, "ScatterPlots/")
TREE_PATH = os.path.join(PATH, "Tree/")

DATA = "seeds_dataset.txt"
TRAIN_DATA = "Train Data"
TEST_DATA = "Test Data"

SINGLE_LINKAGE = "single"
COMPLETE_LINKAGE = "complete"


# In[ ]:


FILE_NAME = ""


# In[ ]:


def load_data():
    data_path = os.path.join(PATH , DATA)
    df = pd.read_csv(data_path, header = None, delim_whitespace=True)
    data = df.to_numpy()
    
    # Each row = one node [0, 1, ..., 209]
    # Add row index as first column of data
    col = []
    for i in range(data.shape[0]):
        col.append(i)
    data = np.column_stack((col, data))
    
    # Shuffle to mixup data
    np.random.shuffle(data)
    
    # Split into train & test data set, train = 190, test = 20
    train, test = data[:190,:], data[190:,:]
    
    x_train = train[:,:-1]
    y_train = train[:,-1:]
    x_test = test[:,:-1]
    y_test = test[:,-1:]
    
    return x_train, y_train, x_test, y_test


# In[ ]:


class Nearest_Neighbor(object):
    def __init__(self, x, x_test, K):
        print("\nNearest_Neighbor Algorithm : ", K)
        self.x = x
        self.x_test = x_test
        self.K = K
        
        self.neighbors = self.nearest_neighbors()
    
    def nearest_neighbors(self):
        # For each test data node, return nearest neighbor list
        # key - node, value - nearest neighbor list
        neighbors = {}
        for n in range(0, len(self.x_test)):
            node = self.x_test[n][0]
            neighbors[node] = self.KNN_Algo(self.x, self.x_test[n], self.K)
            #print("Node : ", node, ", Neighbors : ", neighbors[node])
        
        return neighbors
        
    # KNN Algorithm
    def KNN_Algo(self, x, test_data, k):
        # Column 0 = node id
        #print("Test Node : ", test_data[0])
        nodes = x[:,0]
        p = np.full((len(x), len(test_data)), test_data)
        
        #### 1: Calculate distance
        # From the given point to each data point of the cluster
        distance = np.power((x[:,1:] - p[:,1:]), 2)
        distance = np.sqrt(np.sum(distance, axis=1))
        
        # Include nodes as column 0
        distance = np.column_stack((nodes, distance))
        #print("Distance")
        #print(distance)
        
        #### 2. Neighbour Selection
        # Sorted array
        neighbors = distance[np.argsort(distance[:, 1])]
        # Pick the first K rows into a vector
        neighbors = neighbors[:k, :];
        #print("K neighbor")
        #print(neighbors)
        
        # Return the nearest nodes as neighbors
        return neighbors[:,0]


# In[ ]:


class Hierarchical_Clustering(object):
    def __init__(self, cluster_algo, no_of_clusters, x, y):
        print("Clustering algorihtm : ", cluster_algo, ", No of cluster : ", no_of_clusters)
        self.cluster_algo = cluster_algo
        self.no_of_clusters = no_of_clusters
        self.x = x
        self.y = y
        # Clustering
        # List of clusters, index = cluster id
        self.clusters = self.clustering()
            
    def clustering(self):
        # Nodes : cloumn = 0
        nodes = self.x[:,0]
        #print(nodes)
        
        # Distance Matrix
        #print(self.x.shape, x1.shape)
        dist_matrix = self.get_distance_matrix(self.x)
        #print(dist_matrix.shape)
        #print(dist_matrix)
        # Set diagonal = infinity 
        np.fill_diagonal(dist_matrix, sys.maxsize)
        clusters = self.find_clusters(dist_matrix)
        #print("All the clusters")
        #print(clusters)
        
        # Get n clusters indexes
        # Saved in clusters array in bakcward 
        n = dist_matrix.shape[0] - self.no_of_clusters
        output = clusters[n]
        #print("Output : ", output)
        
        # Get individual cluster indexes
        arr = np.unique(output)
        #print("Unique : ")
        #print(arr)
        
        # List of cluster nodes, first element = cluster id
        n_clusters = np.empty(self.no_of_clusters, object)
        cluster_idx = 0
        for x in np.nditer(arr):
            idx = np.where(output == x)
            cluster = nodes[tuple(idx)]
            # With nodes index list and ground truth y, get most frequent cluster id for the nodes of cluster
            cluster_id = get_most_frequent_cluster_id(np.array(idx), np.array(self.y.flatten().astype(int)))
            #print("Cluster id : ", cluster_id)
            #print("Cluster : ", cluster)
            n_clusters[cluster_idx] = np.append(cluster_id, cluster)
            cluster_idx += 1
        
        #print(n_clusters)
        return n_clusters
        
    def find_clusters(self, dist_matrix):
        clusters = {}
        cluster_idx = []
        row_idx = -1
        col_idx = -1
        
        # Cluster index, [0, 1, 2, ..., 189]
        for n in range(dist_matrix.shape[0]):
            cluster_idx.append(n)
        #print(cluster_idx)
        
        clusters[0] = cluster_idx.copy()
        merge_cluster = [[i] for i in range(dist_matrix.shape[0])]
        #print(merge_cluster)
        
        #finding minimum value from the distance matrix
        #loop will always return minimum value from bottom triangle of matrix
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
            
            #once we find the minimum value, we need to update the distance matrix
            #updating the matrix by calculating the new distances from the cluster to all points
            for i in range(0,dist_matrix.shape[0]):
                if(i != col_idx):
                    # Calculate the distance of every data point from newly formed cluster 
                    
                    # Min for Single Linkage
                    # Max for Complete Linkage
                    if(self.cluster_algo == SINGLE_LINKAGE):
                        temp = min(dist_matrix[col_idx][i],dist_matrix[row_idx][i])
                    else:
                        temp = max(dist_matrix[col_idx][i],dist_matrix[row_idx][i])
                    
                    #we update the matrix symmetrically as our distance matrix should always be symmetric
                    dist_matrix[col_idx][i] = temp
                    dist_matrix[i][col_idx] = temp
            
            #set the rows and columns for the cluster with higher index i.e. the row index to infinity
            #Set input[row_index][for_all_i] = infinity
            #set input[for_all_i][row_index] = infinity
            for i in range (0,dist_matrix.shape[0]):
                dist_matrix[row_idx][i] = sys.maxsize
                dist_matrix[i][row_idx] = sys.maxsize
            #print("dist_matrix")
            #print(dist_matrix)
            
            #Manipulating the dictionary to keep track of cluster formation in each step
            #if k=0,then all datapoints are clusters
            minimum = min(row_idx,col_idx)
            maximum = max(row_idx,col_idx)
            #print("minimum", minimum)
            #print("maximum", maximum)
            for n in range(len(cluster_idx)):
                if(cluster_idx[n] == maximum):
                    cluster_idx[n] = minimum
            clusters[k] = cluster_idx.copy()
            #print(cluster_idx)
            
        #print(merge_cluster[1])
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


# In[ ]:


def prediction(clusters, neighbors):
    # List of clusters, index = cluster id
    pred_cluster = np.empty(len(clusters), object) 
    
    for node in neighbors:
        nn_list = neighbors[node]
        #print("Node : ", node, ", nn_list : ", nn_list)
        nn_cluster = []
        
        # Nearest neighbor belongs to which cluster
        for n in range(0, len(nn_list)):
            c = get_cluster(nn_list[n], clusters)
            #print("node : ", nn_list[n], ", cluster : ", c)
            nn_cluster.append(c)
        # Find most frequent cluster index
        bin = np.bincount(nn_cluster)
        arg_max = np.argwhere(bin == np.amax(bin))
        if(len(arg_max) != 1):
            cluster_idx = nn_cluster[0]
        else:
            cluster_idx = np.bincount(nn_cluster).argmax()
        #cluster_idx = np.bincount(nn_cluster).argmax()
        #print("Majority cluster idx : ", cluster_idx)
        
        # Add node the frequent cluster
        if(pred_cluster[cluster_idx]):
            cluster = pred_cluster[cluster_idx]
        else:
            cluster_id = clusters[cluster_idx][0]
            cluster = [cluster_id]
        cluster.append(node)
        #print("cluster : ", cluster)
        pred_cluster[cluster_idx] = cluster
        #print("pred_cluster : ")
        #print(pred_cluster)
    
    #print(pred_cluster)
    return pred_cluster

# Get the most frequent class as cluster id for the specific cluster
# y = y_train
def get_most_frequent_cluster_id(cluster_idx, y):
    #print("Cluster idx : ", cluster_idx, str(type(cluster_idx)))
    #print("Y : ", y, str(type(y)))
    output = np.take(y, cluster_idx) #y[tuple(cluster_idx)]
    #print("Output : ", output, str(type(output)))
    
    # Check dimension
    if(output.ndim > 1):
        output = output.flatten()
    
    #print("Output : ", output, str(type(output)))
    
    # Find most frequent cluster id
    cluster_id = np.bincount(output).argmax()
    #print("Freq cluster id : ", cluster_id)
    return cluster_id

# Get cluuster given a specific node
def get_cluster(node, clusters):
    cluster_idx = -1
    # For each cluster
    for cls_idx in range(0, len(clusters)):
        cluster = clusters[cls_idx]
        if(cluster is None):
            print("")
        else:
            cls_id = cluster[0]
            cluster = cluster[1:]
            if(node in cluster):
                cluster_idx = cls_idx
                break       
    return cluster_idx

# Dictionary clusters: key-> cluster_id, value -> cluster nodes
def print_clusters(clusters, data):
    print("\n----------- Clusters : ", data, " -----------\n")
    for i in range(0, len(clusters)):
        cluster = clusters[i]
        if(cluster is None):
            print("")
        else:
            print("\nCluster : ", i + 1, ", Cluster id : ", cluster[0], ", size : ", len(cluster[1:]))
            print(cluster[1:])
            
def plot_clusters(clusters, x, fig_name):
    fig = plt.figure(fig_name)
    fig.suptitle('Scatter Plot')
    ax = fig.add_subplot(1,1,1)
    
    nodes = x[:,0]
    p=0
    color = ['r','g','b','y','c','m','k','w']
    for cluster_idx in range(0, len(clusters)):
        cluster = clusters[cluster_idx]
        if(cluster is None):
            print("")
        else:
            cluster_id = cluster[0]
            cluster = cluster[1:]
            for i in range(0, len(cluster)):
                idx = np.where(nodes == cluster[i])
                ax.scatter(x[idx,0], x[idx,1], c = color[p])
            p = p + 1
    plt.show()
    
    global FILE_NAME
    image_name = FILE_NAME + ".png" 
    image = os.path.join(PLOT_PATH , image_name) 
    fig.savefig(image)
    
def calculate_accuracy(x, y, clusters, data):
    print("\nAccuracy : ", data)
    accuracy = 0
    for cluster_idx in range(0, len(clusters)):
        cluster = clusters[cluster_idx]
        if(cluster is None):
            print("")
        else:
            # First item = actual cluster id
            # From 2nd to end = cluster nodes
            cluster_id = cluster[0]
            cluster = cluster[1:]
            #print("Clusters")
            #print(cluster)
            for i in range(0, len(cluster)):
                # Check node belong to cluster_id
                idx = np.where(x[:, 0] == cluster[i])
                if(y[idx] == cluster_id):
                    accuracy += 1
                
    percentage = float(accuracy / len(y)) * 100
    print("Accuracy : ", percentage, "%")


# In[ ]:


def Sklearn_Clustering(cluster_algo, no_of_clusters, x, y):
    model = AgglomerativeClustering(n_clusters = no_of_clusters, affinity='euclidean', linkage=cluster_algo)
    model.fit(x[:,1:])
    labels = model.labels_
    #print(labels)
    
    # List of clusters indexes
    cluster_idxes = np.empty(no_of_clusters, object) 
    nodes = x[:,0]
    for l in range(0, len(labels)):
        # Add index of the clusters
        node = l
        cluster_id = labels[l]
        cluster = []
        # Add node to cluster
        if(cluster_idxes[cluster_id]):
            cluster = cluster_idxes[cluster_id]
        else:
            cluster = []
        cluster.append(node)
        cluster_idxes[cluster_id] = cluster 
        #print("Cluster : ", cluster_id, ", Node idx : ", node)
    
    #print(cluster_idxes)
    # Assign cluster id with most frequent output label
    # List of clusters, index = cluster id
    clusters = np.empty(no_of_clusters, object) 
    cluster_idx = 0
    for cls_id in range(0, len(cluster_idxes)):
        idx_list = cluster_idxes[cls_id]
        #print("Cluster index list ", idx_list)
        cluster_id = get_most_frequent_cluster_id(np.array(idx_list), np.array(y.flatten().astype(int)))
        cluster = np.take(nodes, idx_list) #nodes[tuple(idx)]
        clusters[cluster_idx] = np.append(cluster_id, cluster)
        cluster_idx += 1
    
    print_clusters(clusters, "SKLEARN")
    
    # Calculate accuracy
    calculate_accuracy(x, y, clusters, "SKLEARN")
    
    # Plot clusters
    plot_clusters(clusters, x, "SKLEARN")


# In[ ]:

def DendogramTree(cluster_algo, x):
    fig = plt.figure(cluster_algo)
    dendrogram = sch.dendrogram(sch.linkage(x, method = cluster_algo))
    
    image_name = cluster_algo + ".png" 
    image = os.path.join(TREE_PATH , image_name) 
    fig.savefig(image)

def open_file(s, algo, no_of_clusters, k):
    global FILE_NAME
    FILE_NAME = str(s) + "algo=" + algo + "_" + "cluster=" + str(no_of_clusters) + "_" + "k=" + str(k)
    print(FILE_NAME)
    
    filename = FILE_NAME + ".txt"
    file = os.path.join(OUTPUT_PATH, filename)
    sys.stdout = open(file, 'w')


# In[ ]:


def main():
    
    # Command line argument
    # seed, clustering algo, no of cluster, k
    args = sys.argv
    if(len(args) != 5):
        return
      
    print(args)
    s = int(args[1])
    algo = args[2] 
    no_of_clusters = int(args[3])
    K = int(args[4])
    
    print(s, algo, no_of_clusters, K)
    np.random.seed(s)
    
    # Open file to write all outputs
    open_file(s, algo, no_of_clusters, K)
    
    x_train, y_train, x_test, y_test = load_data()
    
    # Hierarchical Clustering
    hc = Hierarchical_Clustering(algo, no_of_clusters, x_train, y_train)
    clusters = hc.clusters
    print_clusters(clusters, TRAIN_DATA)
    # Calculate accuracy
    calculate_accuracy(x_train, y_train, clusters, TRAIN_DATA)
    
    
    # Nearest Neighbor Classification
    # neighbors: key -> node, value -> neighbor list
    nn = Nearest_Neighbor(x_train, x_test, K)
    neighbors = nn.neighbors
    
    # Get predicted clusters for test data
    # y_pred: key -> cluster, value -> predicted cluster
    pred_clusters = prediction(clusters, neighbors)
    print_clusters(pred_clusters, TEST_DATA)
    # Calculate accuracy
    calculate_accuracy(x_test, y_test, pred_clusters, TEST_DATA)
    
    # Plot clusters
    plot_clusters(pred_clusters, x_test, TEST_DATA)
    
    # Compare
    Sklearn_Clustering(algo, no_of_clusters, x_test, y_test)
    
    # Tree
    #DendogramTree(algo, x_test)

    sys.stdout.close()
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




