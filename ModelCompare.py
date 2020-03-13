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


import sklearn
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
        
        
    def scatterplot_winds(self):
        
        import matplotlib
        
        f = plt.figure(figsize=(8, 6), dpi=100)
        ax = f.add_subplot(111)
        
        x1 = self.data['WS10']
        x2 = self.data['WS100']
        y = self.data['POWER']
        
        ax.scatter(x1, y, color='b', alpha=0.3)
        ax.scatter(x2, y, color='g', alpha=0.1)
        
        ax.axis([x1.min(), x2.max(), y.min(), y.max()])
        #pcm = ax.get_children()[0]
        #cb = plt.colorbar(pcm, ax=ax,)
        #cb.set_label('Data point density')
        plt.xlabel(predictor)
        plt.ylabel(predict)
        plt.title('Linear Regression Fit')
        plt.tight_layout()
        plt.show()       
        
    def LinearRegression(self, pairplot=False, run_model=True, runs=1000):
        """Perform a linear regression"""
        import matplotlib
        
        if pairplot:
            sb.pairplot(self.data)
            plt.show()
            
        predict = 'POWER'
        predictor = 'WS100'
        
        X = self.data[predictor].values.reshape(-1,1)
        y = self.data[predict].values
        
        best_acc = 0
        
        """Running the model 1000 times trying to find the best model"""
        if run_model:
            for _ in range(runs):
            
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                linear = LinearRegression()
                linear.fit(X_train, y_train)
                acc = linear.score(X_test, y_test)
                
                
                if acc > best_acc:
                    print(acc)
                    best_acc = acc
                    
                    """Save our model with the best accuracy"""
                    with open('linregmodel.pkl', 'wb') as handle:
                        pickle.dump(linear, handle)
                    
                    
            print('Model saved')       
            print('Accuracy: {}'.format(best_acc))    # Accuracy for what the Power will be given 
            print('co: ', linear.coef_)               # Coefficients for attributes
            print('Intercept: ', linear.intercept_)   # Intercept of y
        
        else:
            try:
                """Read in our pickle file"""
                pickle_in = open('linregmodel.pkl', 'rb')
                linear = pickle.load(pickle_in)
                print('Polynomial regression model imported')
            except:
                raise ValueError('run_model set to False, missing file "linregmodel.pkl". Set run_model True to run and save model.')
        
        """Hexplot since big dataset"""
        f = plt.figure(figsize=(8, 6), dpi=100)
        ax = f.add_subplot(111)
        
        ax.hexbin(X.flatten(), y, gridsize=28, cmap='Blues', norm=matplotlib.colors.LogNorm())
        ax.axis([X.min(), X.max(), y.min(), y.max()])
        pcm = ax.get_children()[0]
        cb = plt.colorbar(pcm, ax=ax,)
        cb.set_label('Data point density')
        ax.plot(X, (linear.coef_*X+linear.intercept_), color='k', alpha=0.4, lw=3)
        plt.xlabel(predictor)
        plt.ylabel(predict)
        plt.title('Linear Regression Fit')
        plt.tight_layout()
        plt.show()       
        
        
        
    def PolynomialRegression(self, run_model=True, runs=1):
        """Perform a polynomial regression"""
        
        from sklearn.preprocessing import PolynomialFeatures 
        
        predict = 'POWER'
        predictor = 'WS100'
        
        X = self.data[predictor].values.reshape(-1,1)
        y = self.data[predict].values
        
        best_acc = 0
        
        if run_model:
            for _ in range(runs):
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                
                poly_reg = PolynomialFeatures(degree = 6)
                X_poly = poly_reg.fit_transform(X) 
                poly_reg.fit(X_poly, y)
                reg = LinearRegression()
                reg.fit(X_poly, y)
                
                acc = round(reg.score(X_poly, y),4)
                
                if acc > best_acc:
                    print(acc)
                    best_acc = acc
                    
                    """Save our model with pickle. Saved with best result: 70.69 %"""
                    with open('linregmodel.pkl', 'wb') as handle:
                        pickle.dump(poly_reg, handle)
                
            print('Model saved')      
            print('Accuracy: {}'.format(acc))      # Accuracy for what the Power will be given 
            print('co: ', reg.coef_)               # Coefficients for attributes
            print('Intercept: ', reg.intercept_)   # Intercept of y
        
        else:
            try:
                """Read in our pickle file"""
                pickle_in = open('polyregmodel.pkl', 'rb')
                poly_reg = pickle.load(pickle_in)
                print('Polynomial regression model imported')
            except:
                raise ValueError('run_model set to False, missing file "polyregmodel.pkl". Set run_model True to run and save model')
        
        """Hexplot since big dataset"""
        plt.hexbin(X, y, gridsize=20, cmap='Blues')
        plt.axis([X.min(), X.max(), y.min(), y.max()])
        cb = plt.colorbar()
        cb.set_label('Scatter density')
        plt.plot(X, reg.predict(poly_reg.fit_transform(X)), color='#417ed6', alpha=0.7, lw=0.7)
        plt.xlabel(predictor)
        plt.ylabel(predict)
        plt.tight_layout()
        plt.show()
        
        
    def PolynomialRegressionNumpy(self, degree=6, plot=True):
        """Polynomial regression"""
        
        import matplotlib
        
        predict = 'POWER'
        predictor = 'WS100'
        
        X = self.data[predictor].values
        y = self.data[predict].values
        
        polynomial_order = degree
        
        """Curve fit the test data"""
        fitted_parameters = np.polyfit(X, y, polynomial_order)
        print('Fitted Parameters:', fitted_parameters)
        
        model_predictions = np.polyval(fitted_parameters, X)
        abs_error = model_predictions - y
        
        """Model evaluation"""
        se = np.square(abs_error) # squared errors
        mse = np.mean(se) # mean squared errors
        rmse = np.sqrt(mse) # Root Mean Squared Error, RMSE
        r_squared = 1.0 - (np.var(abs_error) / np.var(y))
        print('RMSE:', rmse)
        print('R-squared:', r_squared)
        
        x_model = np.linspace(min(X), max(X))
        y_model = np.polyval(fitted_parameters, x_model)
        
        
        if plot:
            
            f = plt.figure(figsize=(8, 6), dpi=100)
            ax = f.add_subplot(111)
            
            ax.hexbin(X, y, gridsize=28, cmap='PuBu', norm=matplotlib.colors.LogNorm())
            ax.axis([X.min(), X.max(), y.min(), y.max()])
            pcm = ax.get_children()[0]
            cb = plt.colorbar(pcm, ax=ax)
            cb.set_label('Data point density')
            ax.plot(x_model, y_model, color='k', alpha=0.4, lw=4)
            plt.xlabel(predictor)
            plt.ylabel(predict)
            plt.title('Polynomial Regression, Polynomial order {}'.format(degree))
            plt.tight_layout()
            plt.show() 
            
    
    def KNN(self):
        pass    
            
            
            
            
    def LinearRegressionNumpy(self, plot=True, model=True):
        
        predict = 'POWER'
        predictor = 'WS100'
        
        X = self.data.drop(self.data[predict], 1).values
        y = self.data[predict].values
        
        
        
        
    def SupportVectorRegression(self, plot=True, model=True):
        
        from sklearn.svm import SVR
        import matplotlib
        
        predict = 'POWER'
        predictor = 'WS100'
        
        if model:
            
            X = self.data[predictor][:500].values.reshape(-1,1)
            y = self.data[predict][:500].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   
            
            
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma = 0.1)
            #svr_lin = SVR(kernel='linear', C=1e3)
            #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
            y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
            #y_lin = svr_lin.fit(X, y).predict(X)
            #y_poly = svr_poly.fit(X, y).predict(X)
            
            # linear = LinearRegression()
            # acc = linear.score(X, y)
            # print(acc)
        else:
            pass
            
        if plot:  
            
            
            f = plt.figure(figsize=(8, 6), dpi=100)
            ax = f.add_subplot(111)
            
            ax.hexbin(X.flatten(), y, gridsize=28, cmap='Blues', norm=matplotlib.colors.LogNorm())
            ax.axis([X.min(), X.max(), y.min(), y.max()])
            pcm = ax.get_children()[0]
            cb = plt.colorbar(pcm, ax=ax)
            cb.set_label('Data point density')
            ax.plot(X, y_rbf, color='k', alpha=0.4, lw=4)
            plt.xlabel(predictor)
            plt.ylabel(predict)
            plt.title('Support Vector Regression')
            plt.tight_layout()
            plt.show() 
            
            
        #     plt.show()
        #     plt.scatter(X, y, color='k', label=predictor)
        #     plt.plot(X, y_rbf, color='r', lw=3, label='RBF Model')
        #     #plt.plot(X, y_lin, color='c', lw=lw, linestyle='-.', label='Linear Model')
        #     #plt.plot(X, y_poly, color='navy', lw=lw, linestyle='--', label='Polynomial Model')
        #     plt.xlabel(predictor)
        #     plt.ylabel(predict)
        #     plt.title('Support Vector Regression')
        #     plt.legend()
        #     plt.show()
        # else:
            pass
            
            
            
        #     for _ in range(1000):
                
        #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            
        #         clf = svm.SVC()
        #         clf.fit(X_train, y_train)
                
        #         best_acc = 0
                
        #         acc = clf.score(X_test, y_test)
                
        #         if acc > best_acc:
        #             print(acc)
        #             best_acc = acc
                    
        #             """Save our model with pickle. Saved with best result"""
        #             with open('svm-model.pkl', 'wb') as f:
        #                 pickle.dump(clf, f)
                    
        # else:
        #     try:
        #         """Read pickled SVM model"""
        #         pickle_in = open('svm-model.pkl', 'rb')
        #         clf = pickle.load(pickle_in)
        #         print('Polynomial regression model imported')
        #     except:
        #         raise ValueError('run_model set to False, missing file "svm-model.pkl", set run_model True')
                
        

if __name__ == '__main__':
    c = CompareModels()
    #c.LinearRegression()
    #c.PolynomialRegression()
    c.PolynomialRegressionNumpy(degree=6)
    #c.SupportVectorRegression()
    #c.scatterplot_winds()


