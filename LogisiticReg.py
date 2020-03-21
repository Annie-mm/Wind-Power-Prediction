#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 00:32:35 2020

@author: alfredo
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns

class LogReg:
    def __init__(self, x_val=['WS100', 'V100'], test_size=.25):
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
        
        self.test_size = test_size
        self.data = self.data#[['POWER', 'WS10', 'WS100']]
        self.columns = self.data.columns
        self.data['POWER'] = pd.cut(self.data['POWER']*100, bins=[0, 70, 100], labels=[0, 1], include_lowest=True)
        self.reg, self.train, self.test = self._preprocessing(x_val, test_size)
    
    def _preprocessing(self, x_val, test_size):
        """Create LogisticReg Model
        
        Be aware of assumptions: https://www.statisticssolutions.com/assumptions-of-logistic-regression/
        Especially: only non-correlating variables as input
        
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
    
    def compare_test_size(self):
        """Compare algrithm based on different test sizes"""
        sizes = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65]
        train = []
        test = []
        for val in sizes:
            reg, dtrain, dtest = self._preprocessing(['WS100', 'V100'], val)
            train.append(reg.score(dtrain['X'], dtrain['y']))
            test.append(reg.score(dtest['X'], dtest['y']))
        tmp = np.subtract(train, test)
        best_sample_size = sizes[np.argmin(abs(tmp))]
        print('Optimal Sample Size: ' + str(best_sample_size))
        # print(min(abs(tmp)))
    
    def visualise_2d_data(self, X_set, y_set):
        """"Display Logistic Regression in cartheisic axis"""
        from matplotlib.colors import ListedColormap
        fig = plt.figure(figsize=(18, 12))
        ax = fig.gca()
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        ax.contourf(X1, X2, self.reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        ax.set_xlim(X1.min(), X1.max())
        ax.set_ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('black', 'blue'))(i), label = j)
        ax.set_xlabel('Age')
        ax.set_ylabel('Estimated Salary')
        plt.legend()
        
    def compute_confusion_matrix(self):
        """Confusion Matrix visualisation"""
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(1, 2, 1)        
        ax2 = fig.add_subplot(1, 2, 2)
        # Confusion Matrix
        cm_train_disp = plot_confusion_matrix(
            self.reg, self.train['X'], self.train['y'], cmap="RdYlGn", values_format='g', 
            ax=ax2, #display_labels=['Not High Power', 'High Power'],
            )
        # cm_train_disp.im_.colorbar.remove()
        cm_test_disp = plot_confusion_matrix(
            self.reg, self.test['X'], self.test['y'], cmap="RdYlGn", values_format='g', 
            ax=ax1, #display_labels=['Not High Power', 'High Power'],
            )
        # cm_test_disp.im_.colorbar.remove()
        ax1.xaxis.set_label_position('top')
        ax1.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()
        cm_train = self._confusion_analaysis(cm_train_disp.confusion_matrix)
        cm_test = self._confusion_analaysis(cm_test_disp.confusion_matrix)
        print('Train:\n  Accuracy: ' + str(cm_train['accuracy']) +
              '\n  Sensitivity: ' + str(cm_train['sensitivity']) +
              '\n  Specifity: ' + str(cm_train['specificity']) +
              '\nTest:\n  Accuracy: ' + str(cm_test['accuracy']) + 
              '\n  Sensitivity: ' + str(cm_test['sensitivity']) +
              '\n  Specificity: ' + str(cm_test['specificity'])
              )
        return fig, cm_train, cm_test
        
    def _confusion_analaysis(self, cm):
        """"Prepare Confusion Matrix and compute performance"""
        true_neg = cm[0, 0]     # has 'Not High Power'
        false_pos = cm[0, 1]    # has not 'High Power'
        false_neg = cm[1, 0]    # has not 'Not High Power'
        true_pos = cm[1, 1]     # has 'High Power'
        n = true_neg + false_pos + false_neg + true_pos
        result = {
            'cm': cm,
            'accuracy': (true_neg + true_pos) / n,          # overall accuracy 
            'sensitivity': true_pos / (true_pos + false_neg), # accuracy of 'High Power' detection
            'specificity': true_neg / (true_neg + false_pos), # accuracy of 'Not High Power' detection
            }
        return result
    
    def prep_for_report(self):
        """Prepare and print figures and values for export to latex."""
        loc = '../windforcasting_report/'
        size = 15
        pgf_with_latex = {                      # setup matplotlib to use latex for output
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": True,                # use LaTeX to write all text
            "font.family": 'serif',
            "font.serif": [],                   # blank entries should cause plots 
            "font.sans-serif": [],              # to inherit fonts from the document
            "font.monospace": [],
            "axes.labelsize": size,               # LaTeX default is 10pt font.
            "font.size": size,
            "legend.fontsize": size,               # Make the legend/label fonts 
            "xtick.labelsize": size,               # a little smaller
            "ytick.labelsize": size,
            # "figure.figsize": (12, 8),     # default fig size of 0.9 textwidth
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts 
                r"\usepackage[T1]{fontenc}",        # plots will be generated
                r"\usepackage[detect-all,locale=DE]{siunitx}",
                ]                                   # using this preamble
            }
        mp.rcParams.update(pgf_with_latex)
        fig, cm_train, cm_test = self.compute_confusion_matrix()
        fig.axes[0].text(-.7, -0.6, 'a) Training', fontweight='bold')
        fig.axes[1].text(-.7, -0.6, 'b) Testing', fontweight='bold')
        # saving confusion matrix
        fig.savefig(loc + 'figures/logistic_regression/cm.pgf')
        # saving trian and test performance evaluation values
        train = [round(cm_train['accuracy'] * 100, 3),
                 round(cm_train['sensitivity'] * 100, 3),
                 round(cm_train['specificity'] * 100, 3),]
        test = [round(cm_test['accuracy'] * 100, 3),
                 round(cm_test['sensitivity'] * 100, 3),
                 round(cm_test['specificity'] * 100, 3),]        
        print(
            ' & '.join(['\SI{' + str(val) + '}{\percent}' for val in train]), 
            file=open(loc + 'values/' + str(int(self.test_size*100)) + 'logreg_eval_train' + '.tex', 'w'),
        )
        print(
            ' & '.join(['\SI{' + str(val) + '}{\percent}' for val in test]),
            file=open(loc + 'values/' + str(int(self.test_size*100)) + 'logreg_eval_test' + '.tex', 'w'),
        )
        # saving test_size value
        print('\SI{' + str(int(self.test_size*100)) + '}{\percent}',
            file=open(loc + 'values/' + 'test_size.tex', 'w'),
        )
        plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']

    
if __name__ == '__main__':
    obj = LogReg(test_size=.30)
    obj.compute_confusion_matrix()
    # obj.visualise_2d_data(obj.train['X'], obj.train['y'])
    # obj.visualise_2d_data(obj.test['X'], obj.test['y'])
    # obj.compare_test_size()
    # obj.prep_for_report()
    