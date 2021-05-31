#!/usr/bin/env python
# coding: utf-8

# In[15]:


#!/usr/bin/env python
# coding: utf-8

# In[334]:


import numpy as np


# In[335]:


M = 1
W = 2


# In[336]:


def load_data():
    data = np.array([
    [170, 57, 32, W],
    [190, 95, 28, M],
    [150, 45, 35, W],
    [168, 65, 29, M],
    [175, 78, 26, M],
    [185, 90, 32, M],
    [171, 65, 28, W],
    [155, 48, 31, W],
    [165, 60, 27, W],
    [182, 80, 30, M],
    [175, 69, 28, W],
    [178, 80, 27, M],
    [160, 50, 31, W],
    [170, 72, 30, M]
    ])
    return data


# In[337]:


def cartesian_distance(x1, x2):
    row, col = x1.shape

    y = np.power((x1[:, :-1] - x2), 2)
    y = np.sqrt(np.sum(y, axis=1))
    
    #y = np.linalg.norm(x1[:, :-1] - x2) 
    
    return y


# In[338]:

def evaluation(kc, k):
    ev = kc / k
    return ev


def KNN_Algo(data, y, points, k):
    
    row, col = data.shape
    
    # Points vector, M * N-1 vector
    # M = No of rows of original dataset, N = No of columns of original dataset
    # Each row is same for vector operation convinience with 'data' vector
    
    p = np.full((row, col - 1), points) #[1.25 ,1.75]
    
    #### 1. Distance Calculation
    
    # Cartesian distance vector, M*1 vector
    # M = No of rows of original dataset
    # Each row = Cartesian distance between each row of data set and the given query vector x
    
    distance = cartesian_distance(data, p)
    
    # Merge distance and y vector
    y = np.column_stack((distance, y))
    
    #### 2. Neighbour Selection
    
    # Sorted array
    output = y[np.argsort(y[:, 0])]
    
    # Pick the first K rows into a vector
    output = output[:k, :];
    
    #### 3. Prediction
    
    # Select the most frequent occured item
    output = np.array(output[:, -1], dtype = int)
    
    values, counts = np.unique(output, return_counts=True)
    result = np.argmax(counts)
    
    ev = evaluation(np.max(counts), k)
    
    print("Data points : ", p[0], "K = ", k, " Predicted Gender : ")
    
    if(result == M):
        print("M")
    else:
        print("W")
    
    print("Evaluation : ", ev)

# In[351]:


def main():

    # Load data
    data = load_data()
    
    # Output vector, the last column of the dataset
    y = np.array(data[:, -1])
    
    points = np.array([
        [162, 53, 28],
        [168, 75, 32], 
        [175, 70, 30],
        [180, 85, 29]
    ])
    
    K = [1, 3, 5]
    
    for p in points:
        for k in K:
            KNN_Algo(data, y, p, k)
    
if __name__ == '__main__':
    main()



# In[ ]:





# In[ ]:




