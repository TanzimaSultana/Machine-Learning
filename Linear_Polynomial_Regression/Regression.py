import numpy as np
import pandas as pd
import os


# In[2]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


# In[3]:


PATH = os.getcwd() + "\\"


# In[4]:


def load_train_data():
    
    data_path = PATH + "PolyTrain.txt"
    data = pd.read_csv(data_path, header = None, sep = "   ", engine='python')
    
    return data


# In[5]:


def load_test_data():
    data_path = PATH + "PolyTest.txt"
    data = pd.read_csv(data_path, header = None, sep = "   ", engine='python')
    
    return data


# In[6]:


def process_data():
    # Load train data
    df = load_train_data()
    train_data = df.to_numpy()
    #train_data = train_data[np.argsort(train_data[:, 0])]
    
    # Input & Output
    X = train_data[:,:-1].astype(float)
    Y = train_data[:,-1:].astype(float)
    
    # Load test data
    df = load_test_data()
    test_data = df.to_numpy()
    #test_data = test_data[np.argsort(test_data[:, 0])]
    
    x_test = test_data[:,:-1].astype(float)
    y_test = test_data[:,-1:].astype(float)
    
    # Feature Normalization
    X_norm = feature_normalization(X)
    Y_norm = feature_normalization(Y)
    x_test_norm = feature_normalization(x_test)
    y_test_norm = feature_normalization(y_test)
    
    return X_norm, Y_norm, x_test_norm, y_test_norm


# In[7]:


def feature_normalization(X):
    
    mu = np.mean(X)
    sigma = np.std(X)
    X = (X - mu)/sigma
    X_norm = X
    
    return X_norm


# In[8]:


def transform(X, d):
    # Input data X (X1, X2, X1, X2) = (X1, X2, X3 = X1, X4 = X2)
    X_ = X
    for i in range(1,d):
        P = np.power(X, i + 1)
        X_ = np.column_stack((X_, P))
    #print(X_[:2, :])
    return X_


# In[9]:


def prediction(x_test, y_test, theta):

    m = len(x_test)
    ones = np.ones((m,1))
    x_test = np.column_stack((ones, x_test))  
    
    y_pred = np.dot(x_test, theta)
    #print(pred)
    
    test_error = mean_sqr_error(y_test, y_pred)
    #print("Test error ", test_error)
    
    return y_pred, test_error


# In[10]:


def mean_sqr_error(y1, y2):
    m = len(y1)
    sqrError = np.power(y1-y2, 2)
    mse = (1/2) * sum(sqrError)
    return mse


# In[11]:


def plot_2D(X, Y, theta, degree):
    
    X = X[np.argsort(X[:, 1])]
    x = np.array(X[:, 1])
    
    fig = plt.figure()
    plt.scatter(x, Y)
    
    x = X[:20, :]
    y = Y[:20, :]
    y_pred, error = prediction(x, y, theta)
    plt.plot(x, y_pred)
    plt.show()


# In[12]:


def plot_3D(X, Y, theta, degree):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = np.tile(np.arange(start=-2, stop=2, step=0.1), (40,1))
    ys = np.tile(np.arange(start=-2, stop=2, step=0.1), (40,1)).T
    
    xx = np.reshape(len(xs), 1)
    print(xx)
    #y_pred = np.dot(xs, theta)
    
    # Scatter plot
    ax.scatter(X[:, 0], X[:, 1], Y, marker = 'o', color='green', s=20)
    ax.set_title("Linear Regression,degree = :" + str(degree))
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    
    if(degree == 1):
        zs = theta[0] + xs*theta[1] + ys*theta[2]
    elif(degree == 2):
        zs = theta[0] + xs*theta[1] + ys*theta[2] + np.power(xs,2)*theta[3] + np.power(ys,2)*theta[4]
    elif(degree == 3):
        zs = theta[0] + xs*theta[1] + ys*theta[2] + np.power(xs,2)*theta[3] + np.power(ys,2)*theta[4]
        + np.power(xs,3)*theta[5] + np.power(ys,3)*theta[6]
    elif(degree == 4):
        zs = theta[0] + xs*theta[1] + ys*theta[2] + np.power(xs,2)*theta[3] + np.power(ys,2)*theta[4]
        + np.power(xs,3)*theta[5] + np.power(ys,3)*theta[6] + np.power(xs,4)*theta[7] + np.power(xs,4)*theta[8]
    
    ax.plot_surface(xs,ys,zs,alpha=0.5)
    plt.show()


# In[13]:


def LinearRegression(X, Y, x_test, y_test):
    
    print("LINEAR REGRESSION")
    
    # Linear Regression with X & Y
    linear_reg = Regression(X, Y)
    theta = linear_reg.theta
    train_error = linear_reg.train_error
    
    # Prediction 
    #print(logistic_reg.theta)
    y_pred, test_error = prediction(x_test, y_test, theta)
    #print("Prediction")
    #print(y_pred)
    
    # Error
    print("Train error : ", train_error, ", Test error : ", test_error)
    print("Error Diff : ", abs(train_error - test_error))
    
    # Plot function
    plot_3D(X, Y, theta, 1)


# In[14]:


def PolynomialRegression(X, Y, x_test, y_test):
        
    degree = 4
    for d in range(2, 5):
        print("POLYNOMIAL REGRESSION : ", d)
        
        # Add polynomial feature
        X_ = transform(X, d)
        x_test_ = transform(x_test, d)
        
        # Polynomial regression
        polynomial_reg = Regression(X_, Y)
        theta = polynomial_reg.theta
        train_error = polynomial_reg.train_error
        
        # Prediction 
        #print(logistic_reg.theta)
        y_pred, test_error = prediction(x_test_, y_test, theta)
        #print("Prediction")
        #print(y_pred)
        
        # Error
        print("Train error : ", train_error, ", Test error : ", test_error)
        print("Error Diff : ", abs(train_error - test_error))
        
        # Plot function
        plot_3D(X_, Y, theta, d)


# In[15]:


class Regression():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.no_iter = 50
        
        # 1. Initialize x and theta
        self.m = len(x)
        ones = np.ones((self.m,1))
        self.x = np.column_stack((ones, self.x))  

        no_feature = self.x.shape[1]
        self.theta = np.zeros((no_feature, 1))
        
        # Keep cost with each iteration
        self.J = np.ones((self.no_iter, 1))
        
        #print(self.x)
        #print("Initial theta ", self.theta)
        
        self.theta = self.gradient_descent(self.theta)
        self.train_error = self.calculate_train_error(self.theta)
    
    ## --- Cost function --- ##
    def cost_function(self, error):
        m = len(error)
        sqrError = np.power(error, 2)
        cost = (1/ 2 * m) * np.sum(sqrError)
        return cost

    ## --- Gradient Descent --- ##
    def gradient_descent(self, theta):
        alpha = 0.1
        
        #print("Initial theta ", theta)
        for i in range(0, self.no_iter):
            # 2. Hypothesis function
            h = np.dot(self.x, theta)
            #print("Hypothesis", h.shape)
            #print(h)
            
            # 3. Theta update
            # theta = theta - alpha * (1/m) * sum((h - y) * x)
            error = h - self.y
            theta = theta - alpha * (1/self.m) * np.dot(self.x.T, error)

            cost = self.cost_function(error)
            #print("i : ", i, "Cost : ", cost, " Theta : ", theta)

            self.J[i] = cost
        #print(theta)
        return theta
    
    ## --- Train error --- ##
    def calculate_train_error(self, theta):
        train_error = mean_sqr_error(self.y, np.dot(self.x, theta))
        return train_error


# In[16]:


def main():
    
    X, Y, x_test, y_test = process_data()
    
    #print(X)
    #print(Y)
    #print(x_test)
    #print(y_test)
    
    LinearRegression(X, Y, x_test, y_test)
    PolynomialRegression(X, Y, x_test, y_test)

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




