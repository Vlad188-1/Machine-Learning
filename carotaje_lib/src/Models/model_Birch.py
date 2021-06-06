import pandas as pd
import numpy as np

from sklearn.cluster import Birch
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import metrics
from pathlib import Path
from tqdm import tqdm
from joblib import dump, load

class BR():

    def __init__(self, n_clusters=2, treshold=None, branching_factor=None, out_path=None):

        self.n_clusters = n_clusters
        self.treshold = treshold
        self.branching_factor = branching_factor
        self.out_path = out_path

    def training_and_validation(self, X_train, y_train, X_valid, y_valid):

        thresh = np.arange(0.15, 0.46, 0.01)
        branch = np.arange(10, 55, 1)
    
        dict_1 = {}
        
        for i in tqdm(branch, desc='Searching optimal parameters...'):

            F1 = []
            
            print ('\nЗначение branching_factor = ', i)
            
            for j in thresh:
            
                model = Birch(threshold=j, branching_factor=i, n_clusters=self.n_clusters)
                model.fit(X_train)
        
                y_train_predicted = model.predict(X_train)
                y_valid_predicted = model.predict(X_valid)
            
                score_train = roc_auc_score(y_train,  y_train_predicted)
                score_valid = roc_auc_score(y_valid,  y_valid_predicted)

                if score_train < 0.5:
                    y_train_predicted = 1-y_train_predicted 
                    y_valid_predicted = 1-y_valid_predicted
                    score_train = 1-score_train
                    score_valid = 1-score_valid

                F1_score = f1_score(y_valid, y_valid_predicted)
                F1.append([F1_score, j])
 
                print ('treshold = {} \t  Roc_score_valid = '.format(round(j,3)), round(score_valid,3), '\t', 
                                                            f'F1_score_valid = {round(F1_score,3)}')
                
            dict_1['{}'.format(i)] = max(F1)

        list_F1_and_thresh = [value for value in dict_1.values()]

        list_max_F1 = [list_F1_and_thresh[i][0] for i in range(len(list_F1_and_thresh))]
        tresholds_for_max_F1 = [list_F1_and_thresh[i][1] for i in range(len(list_F1_and_thresh))]
        branch = [int(j) for j in [i for i in dict_1.keys()]]

        best_treshold = round(max(list_F1_and_thresh)[1],2)
        best_branch = int(branch[list_F1_and_thresh.index(max(list_F1_and_thresh))])

        print ('Training best model...')
        best_model = Birch(n_clusters=2, threshold=best_treshold, branching_factor=best_branch)
        best_model.fit(X_train, y_train)
        
        y_train_predicted = best_model.predict(X_train)
        y_valid_predicted = best_model.predict(X_valid)
        
        roc_score_train = roc_auc_score(y_train, y_train_predicted)
        roc_score_valid = roc_auc_score(y_valid, y_valid_predicted)
        
        if roc_score_train < 0.5:
            y_train_predicted = 1 - y_train_predicted
            y_valid_predicted = 1 - y_valid_predicted
        
        F1_score_train = f1_score(y_train, y_train_predicted)
        F1_score_valid = f1_score(y_valid, y_valid_predicted)
        
        def save_results(roc_train, roc_valid, f1_train, f1_valid, y_train_predicted, y_valid_predicted):
        
            frame_results = pd.DataFrame({'F1_score_train' : f1_train, 'F1_score_valid': f1_valid,
                                  'Roc_score_train': roc_train, 'Roc_score_valid': roc_valid},
                                 index=[0])   

            if Path(self.out_path).exists():
                frame_results.to_csv(Path(self.out_path)/'Metrics_train_and_validation.csv', index=True)
            else:
                path = Path(self.out_path)
                path.mkdir()
                frame_results.to_csv(path/'Metrics_train_and_validation.csv', index=True)   
                
        save_results(roc_score_train, roc_score_valid, F1_score_train, F1_score_valid, y_train_predicted, y_valid_predicted)

        #print (best_model)

        def _del_cross_links(node):
            for el in node.subclusters_:
                if el.child_ is not None:
                    del el.child_.prev_leaf_
                    del el.child_.next_leaf_
                    _del_cross_links(el.child_)

        _del_cross_links(best_model.root_)

        if Path(self.out_path).exists():
            dump(best_model, Path(self.out_path)/'Best_model_Birch.joblib')

        else:
            path = Path(self.out_path)
            path.mkdir()
            dump(best_model, str(path)+'/Best_model_Birch.joblib')

        return list_max_F1, tresholds_for_max_F1, branch, best_model


    def load_model(self):

        print ('Loading model...')

        best_model = load(Path(self.out_path)/'Best_model_Birch.joblib')    

        print (best_model)

        return best_model 


    def testing(self, model, test_wells):

        print ('Testing_model...')
        
        #Список целевой метрики для каждой тестовой скважины
        y_finals = [well['goal'] for well in test_wells]
        well_id = [int(well['well id'].unique()[0]) for well in test_wells]
        
        #Список всех тестовых скважин, очищенных от ненужных столбцов
        X_finals = [well.drop(['well id', 'depth, m', 'lith', 'goal'], axis = 1) for well in test_wells]
        
        #Предсказанные значения 0,1 для каждой тестовой скважины
        pred_finals = [model.predict(final) for final in X_finals]
        
        #Расчёт метрик roc_auc_score
        roc_score_finals = [round(roc_auc_score(y_true, y_pred),4) for y_true, y_pred in zip(y_finals, pred_finals)]

        if roc_score_finals[0] < 0.5:

            pred_finals = [1-pred for pred in pred_finals]
            roc_score_finals = [1-roc for roc in roc_score_finals]

        f1_score_finals = [round(f1_score(y_true, y_pred),4) for y_true, y_pred in zip(y_finals, pred_finals)]
            
        for idx, Roc_score in zip(well_id, roc_score_finals):
            print (f'ROC Score_final_{idx+1} \t= ', Roc_score)
            
        print ('\n')
        
        for idx, F1_score in zip(well_id, f1_score_finals):
            print (f'F1 Score_final_{idx+1} \t= ', F1_score)
            
        return y_finals, pred_finals, roc_score_finals, f1_score_finals
        
    
    def save_results_test(self, test_wells, roc_score, F1_score, pred_finals):

        well_id = [int(well['well id'].unique()[0]) for well in test_wells]
        well_id = [f'Well_{i}' for i in well_id]

        dict_f1_score = dict(zip(well_id, F1_score))
        dict_pred = dict(zip(well_id, pred_finals))

        frame_well_id = pd.DataFrame(list(dict_f1_score.keys()), index=range(len(dict_f1_score)), columns=['Well_id'])
        frame_f1_score = pd.DataFrame(list(dict_f1_score.values()), index=range(len(dict_f1_score)), columns=['F1_score'])
        frame_roc_score = pd.DataFrame(roc_score, columns=['Roc_score'])
        frame_results = pd.concat([frame_f1_score, frame_well_id, frame_roc_score], axis=1)
        frame_results = frame_results.sort_values(by='F1_score')

        if Path(self.out_path).exists():
            frame_results.to_csv(Path(self.out_path)/'Metrics_teting_wells.csv', index=False)
        else:
            path = Path(self.out_path)
            path.mkdir()
            frame_results.to_csv(path/'Metrics_teting_wells.csv', index=False) 
            
        for idx, final in zip(well_id, pred_finals):
            frame_pred_test = pd.DataFrame({f"{idx}":final}, index=range(len(final)))
            name_file = f"Pred_test_{idx}.csv"
            frame_pred_test.to_csv(Path(self.out_path)/name_file, index=False)
