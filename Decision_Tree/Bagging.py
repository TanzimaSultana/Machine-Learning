#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from random import seed
from random import random
from random import randrange
from pprint import pprint
import os


# In[3]:


PATH = os.getcwd() + "\\"
TRAIN_DATA = "tic-tac-toe_train.csv"
TEST_DATA = "tic-tac-toe_test.csv"


# In[ ]:





# In[ ]:





# In[4]:


def get_entropy(target_col):
    elements,counts = np.unique(target_col,return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy
  
def get_information_gain(data,split_attribute_name,target_name="10"):
    total_entropy = get_entropy(data[target_name])
    vals,counts = np.unique(data[split_attribute_name],return_counts=True)
    #cal the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*get_entropy(data.where(data[split_attribute_name]==vals[i]).
                                dropna()[target_name])for i in range(len(vals))])
    
    #formula for information gain
    Information_Gain = total_entropy-Weighted_Entropy
    return Information_Gain


# In[5]:


def decision_tree(depth, max_depth, data,originaldata,features,target_attribute_name="10", parent_node_class=None):
    # If depth > max_depth
    if(depth > max_depth):
        return
    
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],
                                                                       return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class 
    
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],
                                                                           return_counts=True)[1])]

    #Select the feature which best splits the dataset
    item_values = [get_information_gain(data,feature,target_attribute_name)for feature in features] #Return the infgain values
    best_feature_index = np.argmax(item_values)
    best_feature = features[best_feature_index]

    #Create the tree structure
    tree = {best_feature:{}}
    features = [i for i in features if i!= best_feature]
    for value in np.unique(data[best_feature]):
        value = value
        # Data fro subtree
        sub_data = data.where(data[best_feature]==value).dropna()
        #Recursive algorithm
        global dataset1
        subtree = decision_tree(depth, max_depth, sub_data,dataset1,features,target_attribute_name,parent_node_class)
        #Add the subtree
        tree[best_feature][value] = subtree
        # Increase depth of tree
        depth += 1
        
    return(tree)

#Predict
def predict(query,tree,default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
            
##check the accuracy
def evaluation(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient="records")
    actual = data["10"]
    
    predicted = pd.DataFrame(columns=["predicted"])
    #calculation of accuracy
    for i in range(len(data)):
        pred = predict(queries[i],tree,1.0)
        if(pred == 1):
            pred = 'win'
        if(pred == None):
            pred = 'no-win'
        predicted.loc[i,"predicted"] = pred
        
    accuracy = (np.sum(predicted["predicted"]==data["10"])/len(data))*100
    return 100 - accuracy


# In[6]:


dataset2 = pd.read_csv(PATH + TEST_DATA,names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
test_data = dataset2
global dataset1

# ------------- Bagging ------------- #
# Create a random subsample from the dataset with replacement
def subsample(dataset):
    sample = []
    n_sample = len(dataset)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))

def bagging():
    seed(1)
    
    dataset = pd.read_csv(PATH + TRAIN_DATA,names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    #print("Original dataset")
    #print(dataset)
    
    # Covert string to numeric value
    dataset = np.where(dataset == 'x', 11, dataset)
    dataset = np.where(dataset == 'o', 22, dataset)
    dataset = np.where(dataset == 'b', 33, dataset)
    dataset = np.where(dataset == 'win', 1, dataset)
    dataset = np.where(dataset == 'no-win', 0, dataset)

    original_mean = mean([row[0] for row in dataset])
    print("Original dataset mean : ", original_mean)
    
    for size in [10, 50, 100]:
        sample_means = list()
        for i in range(size):
            sample = subsample(dataset)
            sample_mean = mean([row[0] for row in sample])
            sample_means.append(sample_mean)
        
        print("Sample len : ", len(sample))
        #Convert back the dataset
        df = pd.DataFrame(sample)
        df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        df = np.where(df == 11, 'x', df)
        df = np.where(df == '22', 'o', df)
        df = np.where(df == '33', 'b', df)
        df = np.where(df == '1', 'win', df)
        df = np.where(df == '0', 'no-win', df)
        df = pd.DataFrame(df, columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        
        print("Bagging sample : ", size, ", Mean : ", mean(sample_means))
        
        # ------- Decision Tree -------- #
        global dataset1
        dataset1 = df
        train_data = dataset1
        tree = decision_tree(-1, 6, train_data,train_data,train_data.columns[:-1])
        #pprint(tree)
        print("Train error rate : ", evaluation(train_data,tree))
        print("Test error rate : ", evaluation(test_data,tree))


# In[7]:


def main():
    
    bagging()
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




