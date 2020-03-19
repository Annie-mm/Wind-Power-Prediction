# -*- coding: utf-8 -*-
"""
Created on Sun Mar 1 12:02:31 2020

@author: Sigve Sørensen & Ernst-Martin Buduschin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import warnings

class DataCleaner:
    """Identifies and handles missing data in dataset"""
    
    def __init__(self, clean_data=False):
        try:
            self.data = pd.read_pickle('data_raw.pkl')
        except:
            tmp = pd.read_csv(r'Assignment2.csv')
            tmp_df = pd.DataFrame(tmp)
            with open('data_raw.pkl', 'wb') as handle:
                pickle.dump(tmp_df, handle)
            with open('data_raw.pkl', 'rb') as handle:
                self.data = pickle.load(handle)
     
        self.columns = self.data.columns
        
        """Replace NAN strings with np.nan"""
        self.data.replace('NAN', np.nan, inplace=True)
        
        """Convert object column to numeric column float64"""
        self.data['PRESS'] = pd.to_numeric(self.data['PRESS'])
        
        """Convert to pandas datetime, to utilize Datetimeindex"""
        self.data['TIME'] = pd.to_datetime(self.data.index)
        self.data.set_index(self.data['TIME'], drop=True, inplace=True)
        self.data.drop('TIME', axis=1, inplace=True)
        
        
    def data_distribution(self, column_name=True, _all=True):
        """Distribution-plot to get a feeling of possible outliers"""
        
        if _all:
            for column in self.columns:
                plt.figure(figsize=(15, 8))
                sns.distplot(self.data[column], bins=30)
        else:
            try:
                plt.figure(figsize=(15, 8))
                sns.distplot(self.data[column_name], bins=30)
            except: 
                print('No valid column name given', self.columns)
                
    def sort_datetime_index(self):
        """Making sure datetime index is in correct order"""
        
        self.data.sort_index(inplace=True)
        print(self.data)
                       
                
    def identify_missing_values(self, column_name=None, _all=True, plot=True):
        """Identify zero-, outlier- and NaN values"""
        
        # Check for three types of missing data: 0, -9999, nan
        self.zeros = pd.Series(dtype='int64')
        self.outliers = pd.Series(dtype='int64') 
        self.nans = self.data.isnull().sum() 
        
        
        if _all:
            
            for col in self.columns:
                mean = self.data[col][self.data[col] != -9999].mean()
                std = self.data[col][self.data[col] != -9999].std()
                print(std)
                outlier_count = 0
                zero_count = 0
                for val in self.data[col]:
                    if abs(val) > mean + 8 * std:
                        outlier_count += 1
                    elif val == 0:
                        zero_count += 1
               
                self.zeros.loc[str(col)] = zero_count
                self.outliers.loc[str(col)] = outlier_count
                
            self.missing = pd.concat([self.zeros, self.outliers, self.nans], 
                                      axis=1, 
                                      keys=['Zeros', 'Outliers', 'NaNs'])
        
            
        if plot:

            f, ax = plt.subplots(figsize=(15, 6))
            
            ax.barh(self.missing.index, self.missing['Outliers'], 
                    color='#99FF99', label='Outliers', edgecolor='k', lw=0.5)
            
            ax.barh(self.missing.index, self.missing['NaNs'],
                    color='#fea05e', label='NaNs', edgecolor='k', lw=0.5,
                    left=self.missing['Outliers'])
            
            ax.barh(self.missing.index, self.missing['Zeros'],
                    color='#66b3ff', label='Zeros', edgecolor='k', lw=0.5,
                      left=np.array(self.missing['NaNs'])+
                      np.array(self.missing['Outliers']))
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(loc='upper right', handles = handles, labels=labels, 
                      frameon=True, edgecolor='black', facecolor='white',
                      fontsize=20)
            
            plt.tick_params(axis='both', which='major', labelsize=14.5)
            
            #ax.set_xscale('log') 
            plt.xlabel('Missing data', fontsize=20)
            plt.ylabel('Features', fontsize=20)
            plt.title('Missing values per feature', fontsize=20)
            plt.tight_layout()
            plt.show()
            
        else:
            return self.missing
            
                
    def handle_missing_values(self, method=None, outliers=True, zeros=False):
        """Choose method for handling missing data"""
        
        self.method_dict = {
                        'mean': 'Fill by column mean',           
                        'median': 'Fill by column median',         
                        'remove_row_all': 'Remove row if all NaN', 
                        'remove_row': 'Remove row NaN',     
                        'back-fill': 'Back-fill, propagate previous val',     
                        'forward-fill': 'Forward-fill, propagate next val',   
                        'constant': 'Fill with constant -9999',       
                        'random': 'Fill by random within one std',
                        'zeros': 'Fill by zeros',
                        }       
                        
        
        if outliers:
            """Replace -9999 values with NaN, before proceeding"""
            
            self.data.replace(-9999, np.nan, inplace=True)
            print("All -9999 values replaced by NaN")
            
            for col in self.columns:
                self.data[col] = self.data[col].mask(self.data[col].sub(self.data[col].mean()).div(self.data[col].std()).abs().gt(8))
            
        else:
            print("Proceeding without filling -9999 by NaN")
            
            
        if zeros:
            """Replace 0 values with NaN, before proceeding"""
            
            self.data.replace(0, np.nan, inplace=True)
            print("All 0 values replaced by NaN")
        else:
            print("Proceeding without filling 0s by NaN")
            
        
        if method == 'mean':
            for column in self.columns:
                self.data[column].fillna(self.data[column].mean(), 
                                         inplace=True)
            print("Replaced NaN by column mean")
                      
            
        elif method == 'median':
            for column in self.columns:
                self.data[column].fillna(self.data[column].median(), 
                                         inplace=True)
            print("Replaced NaN by column median")
                
            
        elif method == 'remove_row_all':
            self.data.dropna(how='all', inplace=True)
            print('Removed rows containing exclusively NaN')
            
            
        elif method == 'remove_row':
            self.data.dropna(inplace=True)
            print('Removed rows containing NaN')
            
            
        elif method == 'back-fill':
            self.data.fillna(method='bfill',inplace=True)
            if len(self.data.isna().sum()) > 0:
                warnings.warn('There still exist NaNs in dataset')
            else:
                print('All NaNs replaced by back-fill')
        
        
        elif method == 'forward-fill':
            self.data.fillna(method='ffill', inplace=True)
            if len(self.data.isna().sum()) > 0:
                warnings.warn('There still exist NaNs in dataset')
            else:
                print('All NaNs replaced by back-fill')
                        
                
        elif method == 'constant':
            for column in self.columns:
                self.data[column].fillna(-9999.0, inplace=True)
            print("Replaced NaNs by constant -9999")
        
        
        elif method == 'zeros':
            for column in self.columns:
                self.data[column].fillna(0, inplace=True)
            print("Replaced NaNs by constant 0")
            
        
        elif method == 'random':
            """WORK IN PROGRESS"""
            
            for column in self.columns:
                
                for idx, val in enumerate(self.data[column]):
                    if type(val) != float:
                        local_up = self.data[column][idx:idx+50].dropna()
                        local_down = self.data[column][idx-51:idx-1].dropna()
                        local = pd.concat([local_up, local_down], axis=1)
                        local_mean = local.mean()
                        self.data[column].replace(val, local_mean)
                        print(val, local_mean)
            
                        
                
                
                # #column = 'V10'
                # column_avg = self.data[column].mean()
                # #print('a')
                # column_std = self.data[column].std()
                # #print('a')
                # try:
                #     random = np.random.randint(column_avg - column_std , column_avg + column_std)
                # except:
                #     random = np.random.randint(column_avg + column_std , column_avg - column_std)
                # #print(random)
                # #print('a')
                # self.data[column][self.data[column].isnull()].replace(random, inplace=True)
                # #self.data[column][self.data[column].isnull()] = column_random_list
                # #self.data[column] = self.data[column].astype(int)
            print('Randomly filled NaNs with values close to the mean value but within one standard deviation')
        
        
        else:
            if outliers or zeros:
                pass
            else:
                print('No valid method was chosen. See valid methods below:')
            
                return self.method_dict
            
        
    def save_cleaned_data(self, by=None):
        """Saves a pkl file containing cleaned data by chosen method"""
        
        self.handle_missing_values(method=by)
        with open('data_by_{}.pkl'.format(by), 'wb') as handle:
                pickle.dump(self.data, handle)
        
            
                
if __name__ == "__main__":
    c = DataCleaner()
    c.data_distribution()
    # c.identify_missing_values()
    # c.handle_missing_values()
    # c.identify_missing_values()
    #c.save_cleaned_data(by='mean')
    
    
    
    
    
