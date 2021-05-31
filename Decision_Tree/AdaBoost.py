#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from random import seed
from random import random
from random import randrange
from pprint import pprint
import os


# In[2]:


PATH = os.getcwd() + "\\"
TRAIN_DATA = "tic-tac-toe_train.csv"
TEST_DATA = "tic-tac-toe_test.csv"


# In[ ]:





# In[ ]:





# In[3]:


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


# In[4]:


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
    
    #accuracy = (np.sum(predicted["predicted"]==data["10"])/len(data))*100
    accuracy = (np.sum(predicted["predicted"].reset_index(drop=True) == data["10"].reset_index(drop=True))/len(data))*100
    return (predicted["predicted"], accuracy)


# In[5]:


dataset2 = pd.read_csv(PATH + TEST_DATA,names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
test_data = dataset2
global dataset1


def bagging():
    seed(1)
    
    dataset = pd.read_csv(PATH + TRAIN_DATA,names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    
    #print("Original dataset")
    #print(dataset)

    for size in [1]:
        #Initially assign same weights to each records in the dataset
        weight = 1/(dataset.shape[0])
        alphas = []
        for i in range(size):
            # Initial weight
            probCol = ('probR'+str(i))
            dataset[probCol] = weight
            #simple random sample with replacement
            temp_dataset = dataset.sample(len(dataset), replace = True, weights = dataset[probCol])
            
            temp_dataset = temp_dataset.drop(columns = [probCol])
            # ------- Decision Tree -------- #
            global dataset1
            dataset1 = temp_dataset
            train_data = dataset1
            tree = decision_tree(-1, 3, train_data,train_data,train_data.columns[:-1])
            #pprint(tree)
            pred, accuracy = evaluation(train_data,tree)
            
            pred[pred=='win'] = 1
            pred[pred=='no-win'] =-1
            predCol = 'pred'+str(i)
            dataset[predCol] = pred
            
            #misclassified = 0 if the label and prediction are same
            dataset.loc[dataset['10'] != dataset[predCol], 'misclassified'] = 1
            dataset.loc[dataset['10'] == dataset[predCol], 'misclassified'] = 0
            
            # Error & Alpha
            err = sum(dataset['misclassified'] * dataset[probCol])
            alpha = 0.5*np.log((1-err)/err)
            alphas.append(alpha)
            
            label = dataset['10']
            label[label=='win'] = 1
            label[label=='no-win'] =-1
            
            # Updated weight
            new_weight = dataset[probCol]*np.exp(-1*alpha*np.dot(label, pred))
            #normalized weight
            z = sum(new_weight)
            normalized_weight = new_weight/sum(new_weight)
            weight = round(normalized_weight,4)
            #print("Train accuracy : ", evaluation(train_data,tree))
            #print("Test accuracy : ", evaluation(test_data,tree))


# In[6]:


def main():
    
    bagging()
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




