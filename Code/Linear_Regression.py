import pandas as pd
import numpy  as np
import matplotlib.pylab as plt

class LinearRegression:
    
    def __init__(self,X,y,iteration,alpha,add_intecept = True):   
        self.X             = X
        self.y             = y
        self.add_intercept = add_intecept
        self.iter          = iteration
        self.alpha         = alpha
        
        #adding an intercept column if the add intercept variable is true
        if self.add_intercept ==  True:
            self.X.insert(0,"Intercept",1)
        #converting X & Y to matrix form
        self.X = np.matrix(self.X)
        self.y = np.matrix(self.y).T      
        #no of elemements in the dataset         
        self.m = len(self.X)       
       #intialize the theta or the intercept and slope values
        self.theta = np.matrix(np.zeros(shape = self.X.shape[1]))
                 
    def computecost(self,X,y,theta):
        prediction = X * theta.T
        error      = np.power((prediction - y),2)
        cost       = (np.sum(error))/(2 * self.m)
        return cost
    
    def gradientdecent(self):
        cost       = np.zeros(shape = self.iter)
        parameters = self.theta.size
        
        for i in range(self.iter):
            prediction = self.X * self.theta.T
            error      = prediction - self.y
            for p in range(parameters):
                temp1      =  np.multiply(error , self.X[:,p])
                self.theta[0,p] = self.theta[0,p] - ((self.alpha/self.m) * np.sum(temp1))            
            cost[i] = self.computecost(self.X,self.y,self.theta)
        return self.theta,cost
            
    def predict(self,X_Test):

        if self.add_intercept == True:
            X_Test.insert(0,"Intercept",1)
            
        theta,cost = self.gradientdecent()
        x_predict  = np.matrix(X_Test) * theta.T
        
        return x_predict
    
    def plot_error(self,cost):
        fig,ax = plt.subplots(figsize =(12,10))
        ax.plot(np.arange(self.iter),cost,"r")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        ax.set_title("Error vs Training")
        
    def model_accuracy(self,y_pred):
        #calculate root mean squared
        score =  np.sqrt(np.sum(np.power((self.y - y_pred),2))/self.m)
        return score
    
    
class Preprocessing: 
    def __init__(self,dataset, feature_normalization = True):        
        self.dataset = dataset
        self.feature_normalization = feature_normalization
    
    def normalization(self):
        #normailizing the features if the feature_normalization is True
        if self.feature_normalization is True:
            self.dataset.iloc[:,0:self.dataset.shape[1]-1] = (self.dataset - self.dataset.mean()) /self.dataset.std() 
        return self.dataset

    def data_prep(self,dataset_type):        
        dataset = self.normalization()
        #converting the dataset into x and y .please keep in mind this code assumes the last column of a given dataset is always the target varaible.
        if dataset_type == "Train":    
            X = dataset.iloc[:,0:dataset.shape[1]-1]
            y = dataset.iloc[:,dataset.shape[1]-1]
            return X,y
        else :
            X = dataset.iloc[:,0:dataset.shape[1]-1]
            return X


#reading training and test dataset 
training_dataset         = pd.read_csv(r"C:\Users\sauravghosh\Desktop\MachineLearning\Linear_Regression\MultiVariate_Regression\DataSet\Training_DataSet.txt",header = None)
training_dataset.columns = ["Area","NoOfBedRooms","Price"]

test_dataset             = pd.read_csv(r"C:\Users\sauravghosh\Desktop\MachineLearning\Linear_Regression\MultiVariate_Regression\DataSet\Test_DataSet.txt",header = None)
test_dataset.columns     = ["Area","NoOfBedRooms","Price"]

dataset_type_train = "Train"
dataset_type_test  = "Test"
#intialize parameters for linear regression model
add_intercept         = True   #add_intecept can take only two values : True or False
feature_normalization = True   #feature normalization can take only two values : True or False
iteration             = 10000
alpha                 = 0.001

#create an instance of an class which can be used for performing preprocessing of the dataset
obj                   = Preprocessing(training_dataset,feature_normalization)
x_train,y_train       = obj.data_prep(dataset_type_train)
x_test                = obj.data_prep(dataset_type_test)

#create an instance of the class linear regression which will be used for prediction
obj1                  = LinearRegression(x_train,y_train,iteration,alpha,add_intercept)
theta, cost           = obj1.gradientdecent()
predictedvalue        = obj1.predict(x_test)
obj1.plot_error(cost)
print obj1.model_accuracy(predictedvalue)






