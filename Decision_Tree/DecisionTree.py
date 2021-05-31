#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from pprint import pprint


# In[2]:


PATH = os.getcwd() + "\\"
TRAIN_DATA = "tic-tac-toe_train.csv"
TEST_DATA = "tic-tac-toe_test.csv"


# In[3]:


dataset1 = pd.read_csv(PATH + TRAIN_DATA,names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
dataset2 = pd.read_csv(PATH + TEST_DATA,names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
train_data = dataset1
test_data = dataset2


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
    return accuracy


# In[6]:


def main():
    
    for d in range(1, 10):
        tree = decision_tree(-1, d, train_data,train_data,train_data.columns[:-1])
        #pprint(tree)
        print("Depth : ", d)
        print("Train accuracy : ", evaluation(train_data,tree))
        print("Test accuracy : ", evaluation(test_data,tree))

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




