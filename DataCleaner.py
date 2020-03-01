import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import seaborn as sns

class DataCleaner:
    def __init__(self, run_pickle=False):
        self.data = pd.read_pickle('data_raw.pkl')
        self.columns = self.data.iloc[:0]

        if run_pickle:
            tmp = pd.read_csv(r'Assignment2.csv', index_col = 'TIMESTAMP')
            tmp_df = pd.DataFrame(tmp)
            with open('data_raw.pkl', 'wb') as handle:
                pickle.dump(tmp_df, handle)
            with open('data_raw.pkl', 'rb') as handle:
                self.data = pickle.load(handle)
                

                
        
        
    def identify_bad_data(self, column_name=None, _all=False):
        """Distribution-plot to get a feeling of possible outliers"""
        
        if _all:
            for column in self.columns:
                plt.figure(figsize=(15,8))
                sns.distplot(self.data[column], bins=30)
        else:
            try:
                plt.figure(figsize=(15,8))
                sns.distplot(self.data[column_name], bins=30)
            except: 
                print('No valid column name given', self.columns)
                
                
    def identify_missing_values(self, column_name=None, _all=True, plot=True):
        """Identify zero-, outlier- and NaN values"""
        
        self.zeros = pd.Series(dtype='int64')
        self.outliers = pd.Series(dtype='int64')
        self.nans = pd.Series(dtype='int64')
        #self.nans = self.data.isnull().sum()
        
        
        if _all:
            
            for column in self.columns:
                zero_count = 0
                outlier_count = 0
                nan_count = 0
                for val in self.data[column].values:
                    if val == 0:
                        zero_count += 1
                    elif val == -9999:
                        outlier_count += 1
                    elif val == 'NAN':
                        nan_count += 1
               
                self.zeros.loc[str(column)] = zero_count
                self.outliers.loc[str(column)] = outlier_count
                self.nans.loc[str(column)] = nan_count
            
            self.missing_data = pd.concat([self.zeros, self.outliers, self.nans], axis=1, keys=['Zeros', 'Outliers', 'NaNs'])
        else:
            pass
            
        if plot:

            f, ax = plt.subplots(figsize=(15, 6))
            
            ax.barh(self.missing_data.index, self.missing_data['Outliers'], 
                    color='#21725F', label='Outliers', edgecolor='k', lw=0.5)
            
            ax.barh(self.missing_data.index, self.missing_data['NaNs'],
                    color='#509B53', label='NaNs', edgecolor='k', lw=0.5,
                    left=self.missing_data['Outliers'])
            
            ax.barh(self.missing_data.index, self.missing_data['Zeros'],
                    color='#FFF66B', label='Zeros', edgecolor='k', lw=0.5,
                      left=np.array(self.missing_data['NaNs'])+np.array(self.missing_data['Outliers']))
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(loc='upper right', handles = handles, labels=labels, frameon=True,
                      edgecolor='black', facecolor='white', fontsize=20)
            
            plt.tick_params(axis='both', which='major', labelsize=14.5)
            
            ax.set_xscale('log')
            plt.xlabel('Missing data', fontsize=20)
            plt.ylabel('Features', fontsize=20)
            plt.title('Missing values per feature (0, -9999, NaN)', fontsize=20)
            plt.tight_layout()
            plt.show()
        else:
            pass
            
                
    def handle_missing_values(self, method=None, outliers=False, zeros=False):
        """Choose method for handling missing data"""
        
        self.method_dict = {
                        'mean': 'Fill by column mean',           
                        'median': 'Fill by column median',         
                        'remove_row_all': 'Remove row if all NaN', 
                        'remove_row': 'Remove row NaN',     
                        'remove_value': 'Remove data point',
                        'back-fill': 'Back-fill, propagate previous val',     
                        'forward-fill': 'Forward-fill, propagate next val',   
                        'constant': 'Fill with constant -9999',       
                        'random': 'Fill by random within one std',
                        }       
                        
        
        if outliers:
            """Replace -9999 values with NaN, before proceeding"""
            
            for column in self.columns:
                for val in self.data[column]:
                    if val == -9999:
                        val.fillna(inplace=True)
            print("All -9999 values replaced by NaN")
        else:
            print("Proceeding without filling -9999 by NaN")
            
        if zeros:
            """Replace 0 values with NaN, before proceeding"""
            
            for column in self.columns:
                for val in self.data[column]:
                    if val == 0:
                        val.fillna(inplace=True)
            print("All 0 values replaced by NaN")
        else:
            print("Proceeding without filling 0s by NaN")
            
        
        if method == 'mean':
            for column in self.columns:
                self.data.column.fillna(self.data.column.mean(), inplace=True)
            print("Replaced NaN by column mean")
                        
        elif method == 'median':
            for column in self.columns:
                self.data.column.fillna(self.data.column.median(), inplace=True)
            print("Replaced NaN by column median")
                
        elif method == 'remove_row_all':
            self.data.dropna(how='all', inplace=True)
            print('Removed rows containing exclusively NaN')
            
        elif method == 'remove_row':
            self.data.dropna(inplace=True)
            print('Removed rows containing NaN')
            
        elif method == 'back-fill':
            self.data.fillna(method='bfill',inplace=True)
            for column in self.columns:
                for val in self.data[column]:
                    if val.isna:
                        print('There still exist NaNs in dataset')
                        break
                    else:
                        print('All NaNs replaced by back-fill')
        
        elif method == 'forward-fill':
            self.data.fillna(method='ffill', inplace=True)
            for column in self.columns:
                for val in self.data[column]:
                    if val.isna:
                        print('There still exist NaNs in dataset')
                        break
                    else:
                        print('All NaNs replaced by forward-fill')
                        
        elif method == 'constant':
            for column in self.columns:
                self.data.column.fillna(-9999, inplace=True)
            print("Replaced NaNs by constant -9999")
        
        elif method == 'random':
            for column in self.data:
                column_avg = self.data.column.mean()
                column_std = self.data.column.std()
                column_null_count = self.data.column.isnull().sum()
                column_random_list = np.random.randint(column_avg - column_std, column_avg + column_std, size=column_null_count)
                self.data.column[np.isnan(self.data.column)] = column_random_list
                self.data.column = self.data.column.astype(int)
            print('Randomly filled NaNs with values close to the mean value but within one standard deviation')
        
        else:
            print('No valid method was chosen. See valid methods below:')
            
            return self.method_dict
            
        
                
if __name__ == "__main__":
    c = DataCleaner()
    c.identify_missing_values()
    
    
    
    
