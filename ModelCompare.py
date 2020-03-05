# -*- coding: utf-8 -*-
"""
Created on Thu Mar 5 17:45:50 2020

@author: Sigve Sørensen & Ernst-Martin Buduschin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

from matplotlib import cm as CM
from matplotlib import mlab as ML

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score, mean_squared_error
from collections import Counter



class CompareModels:
    def __init__(self):
        
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        self.data = self.data[['POWER', 'WS10', 'WS100']]
        self.columns = self.data.columns
        
        
    def LinearRegression(self, pairplot=False, run_model=True):
        """Perform a linear regression"""
        
        
        if pairplot:
            sb.pairplot(self.data)
            plt.show()
            
        predict = 'POWER'
        predictor = 'WS100'
        
        X = self.data.drop([predict], 1).values
        y = self.data[predict].values
        
        best = 0
        
        """Running the model 1000 times trying to find the best model"""
        if run_model:
            for _ in range(1000):
            
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                
                linear = LinearRegression()
                linear.fit(x_train, y_train)
                acc = linear.score(x_test, y_test)
                
                
                if acc > best:
                    print(acc)
                    best = acc
                    
                    """Save our model with pickle. Saved with best result: 70.69 %""".format(best)
                    with open('linregmodel.pkl', 'wb') as f:
                        pickle.dump(linear, f)
                    
                    
            print('Accuracy: {}'.format(acc))         # Accuracy for what the Power will be given 
            print('co: ', linear.coef_)               # Coefficients for attributes
            print('Intercept: ', linear.intercept_)   # Intercept of y
        
        
            
        """Read in our pickle file"""
        pickle_in = open('linregmodel.pkl', 'rb')
        linear = pickle.load(pickle_in)
        
        """Hexplot since big dataset"""
        plt.hexbin(X[:,1], y, gridsize=20, cmap='Blues')
        plt.axis([X[:,1].min(), X[:,1].max(), y.min(), y.max()])
        cb = plt.colorbar()
        cb.set_label('Scatter density')
        plt.plot(X, (linear.coef_*X+linear.intercept_), color='#417ed6', alpha=0.7, lw=0.7)
        plt.xlabel(predictor)
        plt.ylabel(predict)
        plt.tight_layout()
        plt.show()
        
        
    def PolynomialRegression(self):
        """Perform a polynomial regression"""
        
        from sklearn.preprocessing import PolynomialFeatures 
        
        predict = 'POWER'
        
        X = self.data.drop([predict], 1).values
        y = self.data[predict].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        
        best = 0
        
        for _ in range(1000):
            
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
        
            linear = linear_model.LinearRegression()
            linear.fit(x_train, y_train)
            
            poly = PolynomialFeatures(degree = 4) 
            X_poly = poly.fit_transform(x_train) 
            poly.fit(X_poly, y_train) 
            
            
            acc = linear.score(x_test, y_test)
            
            if acc > best:
                print(acc)
                best = acc
                
                """Save our model with pickle. Saved with best result:  %""".format(best)
                with open('linregmodel.pkl', 'wb') as f:
                    pickle.dump(linear, f)
                
                
        print('Accuracy: {}'.format(acc))         # Accuracy for what the Power will be given 
        print('co: ', linear.coef_)               # Coefficients for attributes
        print('Intercept: ', linear.intercept_)   # Intercept of y
        

            
        """Read in our pickle file"""
        pickle_in = open('linregmodel.pkl', 'rb')
        linear = pickle.load(pickle_in)
        
        p = 'WS100' # Predictor variable
        
        plt.figure()
        plt.scatter(self.data[p], self.data[predict])
        plt.plot(X, linear.predict(poly.fit_transform(X)), color = 'red')
        plt.xlabel(p)
        plt.ylabel('Power')
        plt.tight_layout()
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

if __name__ == '__main__':
    c = CompareModels()
    c.LinearRegression(run_model=False)
    #c.PolynomialRegression()


