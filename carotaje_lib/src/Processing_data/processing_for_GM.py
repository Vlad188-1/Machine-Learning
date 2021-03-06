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
    
    random.seed(10)
    
    def engine(x):
        
        all_wells = []
        
        for i in x['well id'].unique():
            all_wells.append(x[x['well id']==i])
        
        for i in tqdm(range(len(all_wells)), desc='Preprocessing data...'):

            all_wells[i]['diff_depth'] = all_wells[i]['depth, m'].diff(periods=-1)

            all_wells[i] = all_wells[i].loc[~all_wells[i]['diff_depth'].isin([0.0])]

            all_wells[i] = all_wells[i].dropna()

            # feature enineering
            all_wells[i] = all_wells[i].drop(['NKTM'], axis = 1)
            all_wells[i]['diff_ALPS'] = all_wells[i]['ALPS'].diff(periods = -1)/all_wells[i]['diff_depth']
            all_wells[i]['diff_NKTR'] = all_wells[i]['NKTR'].diff(periods=-1)
            all_wells[i]['diff_NKTD'] = all_wells[i]['NKTD'].diff(periods=-1)/all_wells[i]['diff_depth']
            all_wells[i]['bk**2'] = pow(all_wells[i]['bk'],2)
            all_wells[i]['GZ1**2'] = pow(all_wells[i]['GZ1'],2)
            all_wells[i]['GZ1**3'] = pow(all_wells[i]['GZ1'],3)
            all_wells[i]['GZ7/DGK'] = all_wells[i]['GZ7']/all_wells[i]['DGK']
            all_wells[i]['GZ1/DGK'] = all_wells[i]['GZ1']/all_wells[i]['DGK']
            all_wells[i]['GZ5/DGK'] = all_wells[i]['GZ5']/all_wells[i]['DGK']
    
            all_wells[i] = all_wells[i].drop(['diff_depth'], axis=1)
        
            all_wells[i] = all_wells[i].dropna()
            
        frame = pd.concat(all_wells, axis=0) 
            
        return frame  

    all_wells_1_process = engine(data_1)
    
    #id_test_wells = random.sample(list(all_wells_1_process['well id'].unique()), 5)

    test_wells = [all_wells_1_process[all_wells_1_process['well id']==i] for i in [26,14,75, 23, 44]]#[26,14,75, 23, 44]]

    all_wells_1_process = all_wells_1_process.loc[~all_wells_1_process['well id'].isin([26,14,75, 23, 44])]

    x_train, x_valid, y_train, y_valid = train_test_split(all_wells_1_process.drop(['well id', 'depth, m', 'lith', 'goal'], axis=1),
                                                      all_wells_1_process['goal'], shuffle=True, test_size=0.3, 
                                                      stratify=all_wells_1_process['goal'], random_state=12)
            
    return x_train, x_valid, y_train, y_valid, test_wells, all_wells_1_process