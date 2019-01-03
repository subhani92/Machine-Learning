# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:30:58 2018

@author: Asus
"""
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.svm import SVC  

def importdata(): 
    df = pd.read_csv("bankruptcy.csv") 
      
    print ("Dataset Lenght: ", len(df)) 
    print ("Dataset Shape: ", df.shape) 
      
    print ("Dataset: \n",df.head()) 
    return df

def splitdataset(df): 

   
    X = df.drop('Class', axis=1)
    Y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state =0 ) 
   # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.50)  
   
      
    return X, Y, X_train, X_test, y_train, y_test 


# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
     
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
#Function using svm

def train_using_svm(X_train, X_test, y_train): 
    
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)  

    return svclassifier

# Function to make predictions 
def prediction(X_test, clf_object): 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix:\n ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ('Accuracy :',format(
    accuracy_score(y_test,y_pred)*100, '0.5f'))
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 

def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
    clf_svm=train_using_svm(X_train,X_test, y_train)
    
    
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    print("Results Using SVM:")
    #result using SVM
    y_pred_svm = prediction(X_test, clf_svm)
    cal_accuracy(y_test, y_pred_svm)
      
# Calling main function 
if __name__=="__main__": 
    main() 
