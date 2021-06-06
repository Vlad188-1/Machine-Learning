# My impotrs ===============================
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
# ==========================================

def process(data_1):
    
    def engine(x):
        
        all_wells = []
        
        for i in x['well id'].unique():
            all_wells.append(x[x['well id']==i])
        
        for i in tqdm(range(len(all_wells)), desc='Processing data...'):
            
    
            all_wells[i]['diff_depth'] = all_wells[i]['depth, m'].diff(periods=-1)

            all_wells[i] = all_wells[i].loc[~all_wells[i]['diff_depth'].isin([0.0])]
    
            all_wells[i] = all_wells[i].dropna()
    
            all_wells[i] = all_wells[i].drop(['diff_depth'], axis=1)
        
            #feature_engine
            features = ['bk', 'GZ1', 'GZ2', 'GZ3', 'GZ4', 'GZ5', 'GZ7', 'DGK', 'ALPS', 'NKTM', 'NKTD']

            for col in features:
              all_wells[i][f'diff_{col}'] = all_wells[i][col].diff(periods=-1)/all_wells[i]['depth, m'].diff(periods=-1)

              all_wells[i][f'diff_{col}'] = (all_wells[i][f'diff_{col}']-all_wells[i][f'diff_{col}'].min())/\
              (all_wells[i][f'diff_{col}'].max()-all_wells[i][f'diff_{col}'].min())


            for col in features:
              all_wells[i][f'smooth_{col}'] = all_wells[i][col].rolling(window=3, min_periods=1).mean()


            for col in features:
              all_wells[i][f'{col}^2'] = all_wells[i][col]**2
              all_wells[i][f'{col}^3'] = all_wells[i][col]**3

            for col in features[1:7]:
              all_wells[i][f'{col}/DGK'] = all_wells[i][col]/all_wells[i]['DGK']

            all_wells[i] = all_wells[i].dropna()
            
        frame = pd.concat(all_wells, axis=0) 
            
        return frame  
    
    all_wells_1_process = engine(data_1)
    
    test_wells = [all_wells_1_process[all_wells_1_process['well id']==i] for i in [26,14,75, 23, 44]]  ##82, 267, 75
    
    all_wells_1_process = all_wells_1_process.loc[~all_wells_1_process['well id'].isin([26,14,75, 23, 44])]
    
    x_train, x_valid, y_train, y_valid = train_test_split(all_wells_1_process.drop(['well id', 'depth, m', 'lith', 'goal'], axis=1),
                                                      all_wells_1_process['goal'], shuffle=True, test_size=0.3, 
                                                      stratify=all_wells_1_process['goal'], random_state=42)
       
        
    return x_train, x_valid, y_train, y_valid, test_wells