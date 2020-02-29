import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import seaborn as sns

class DataCleaner:
    def __init__(self, run_pickle=False):
        
        

        if run_pickle:
            tmp = pd.read_csv(r'Assignment2.csv', index_col = 'TIMESTAMP')
            tmp_df = pd.DataFrame(tmp)
            with open('data_raw.pkl', 'wb') as handle:
                pickle.dump(tmp_df, handle)
            with open('data_raw.pkl', 'rb') as handle:
                self.data = pickle.load(handle)
                
        else:
            with open('data_raw.pkl', 'rb') as handle:
                self.data = pickle.load(handle)
                
    
    def identify_bad_data(self, column_name):
        """Plot distribution-plot to get a feeling of outliers"""
        plt.figure(figsize=(15,8))
        sns.distplot(self.data[column_name], bins =30)
                
if __name__ == "__main__":
    c = DataCleaner(run_pickle=False)
    #print(c.data.head())
    c.identify_bad_data('POWER')
    