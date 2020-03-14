#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 00:32:35 2020

@author: alfredo
"""

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class LogReg:
    def __init__(self, x_val=['WS100', 'V100']):
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        self.data = self.data#[['POWER', 'WS10', 'WS100']]
        self.columns = self.data.columns
        self.data['POWER'] = pd.cut(self.data['POWER']*100, bins=[0, 70, 100], labels=[0, 1], include_lowest=True)
        self.reg, self.train, self.test = self._preprocessing(x_val)
    
    def _preprocessing(self, x_val, test_size=.25, ):
        """Create LogisticReg Model
        
        Be aware of assumptions: https://www.statisticssolutions.com/assumptions-of-logistic-regression/
        
        Parameters
        ----------
        x_val:  list of str
            Contains variabele on which to build model
        test_size:  float
            Percentage of the test data size        
        """
        # Data Selection
        X = self.data.loc[:, x_val].values
        y = self.data.loc[:, 'POWER'].values
        # Encoding
        labelenc_y = LabelEncoder()
        y = labelenc_y.fit_transform(y)
        # Train/Test Selection
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state = 0)
        # print('Train: ' + str(X_train.mean(axis=0))+ ' / ' + str(X_train.std(axis=0)))
        # print('Test: ' + str(X_test.mean(axis=0))+ ' / ' + str(X_test.std(axis=0)))
        # Feature Scaling
        sc = StandardScaler()
        train = {
            'X': sc.fit_transform(X_train),
            'y': y_train,
            }
        test = {
            'X': sc.transform(X_test),
            'y': y_test,
            }
        # Logistic Regression Fitting
        reg = LogisticRegression(random_state=0)
        reg.fit(train['X'], train['y'])
        # print(reg.intercept_)
        # print(reg.coef_)
        # print(reg.score(test['X'], test['y']))
        return reg, train, test
        
        # # Predict y test results
        # y_pred = classifier.predict(X_test)
        
        # fig = plt.figure(figsize=(18, 6))
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax2 = fig.add_subplot(1, 2, 2)
        # # Confusion Matrix
        # cm_test = confusion_matrix(y_test, y_pred)
        # cm_train = confusion_matrix(y_train, classifier.predict(X_train))
        # sns.heatmap(cm_test, cmap="RdYlGn", fmt='g',
        #             ax=ax1, annot=True, square=True)
        # sns.heatmap(cm_test, cmap="RdYlGn", fmt='g',
        #             ax=ax1, annot=True, square=True)
        # cm_test = plot_confusion_matrix(
        #     classifier, X_test, y_test, cmap="RdYlGn", values_format='g', 
        #     ax=ax1, display_labels=['Not High Power', 'High Power'],
        #     )
        # cm_test.im_.colorbar.remove()
        # cm_test = cm_test.confusion_matrix
        # cm_train = plot_confusion_matrix(
        #     classifier, X_train, y_train, cmap="RdYlGn", values_format='g', 
        #     ax=ax2, display_labels=['Not High Power', 'High Power'],
        #     )
        # cm_train.im_.colorbar.remove()
        # cm_train = cm_train.confusion_matrix
        # ax1.xaxis.set_label_position('top')
        # ax1.xaxis.tick_top()
        # ax2.xaxis.set_label_position('top')
        # ax2.xaxis.tick_top()
        # sns.heatmap(cm_test, cmap="RdYlGn",
        #             ax=ax2, annot=True, square=True)
        # plt.scatter(X_train[:, 0], y_train, color='r')
        # plt.scatter(X_test[:, 0], y_test, color='k')
        
    def compute_confusion_matrix(self):
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        # Confusion Matrix
        cm_train_disp = plot_confusion_matrix(
            self.reg, self.train['X'], self.train['y'], cmap="RdYlGn", values_format='g', 
            ax=ax2, display_labels=['Not High Power', 'High Power'],
            )
        # cm_train_disp.im_.colorbar.remove()
        cm_test_disp = plot_confusion_matrix(
            self.reg, self.test['X'], self.test['y'], cmap="RdYlGn", values_format='g', 
            ax=ax1, display_labels=['Not High Power', 'High Power'],
            )
        # cm_test_disp.im_.colorbar.remove()
        ax1.xaxis.set_label_position('top')
        ax1.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()
        cm_train = self._confusion_analaysis(cm_train_disp.confusion_matrix)
        cm_test = self._confusion_analaysis(cm_test_disp.confusion_matrix)
        print('Train:\n  Accuracy: ' + str(cm_train['accuracy']) +
              '\nTest:\n  Accuracy: ' + str(cm_test['accuracy']))
        
    def _confusion_analaysis(self, cm):
        true_neg = cm[0, 0]     # has 'Not High Power'
        false_pos = cm[0, 1]    # has not 'High Power'
        false_neg = cm[1, 0]    # has not 'Not High Power'
        true_pos = cm[1, 1]     # has 'High Power'
        n = true_neg + false_pos + false_neg + true_pos
        result = {
            'cm': cm,
            'accuracy': (true_neg + true_pos) / n,          # overall accuracy 
            'sensivity': true_pos / (true_pos + false_neg), # accuracy of 'High Power' detection
            'specifity': true_neg / (true_neg + false_pos), # accuracy of 'Not High Power' detection
            }
        return result
    
if __name__ == '__main__':
    obj = LogReg()
    #print(obj.data['POWER'])
    obj.compute_confusion_matrix()
    