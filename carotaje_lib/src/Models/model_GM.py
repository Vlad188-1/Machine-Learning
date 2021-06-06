import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import metrics
from pathlib import Path
from tqdm import tqdm
from joblib import dump, load


class GM():
    
    def __init__(self, n_clusters=2, random_state=123, n_init=None, tol=None, init_params=None,
                             max_iter=None, reg_covar=None,out_path=None):
        
        self.n_clusters = n_clusters
        self.random_state=random_state
        self.out_path = out_path
        self.cov_types = ['full', 'diag', 'tied', 'spherical']
        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.init_params = init_params

    def training_and_validation(self, X_train, y_train, X_valid, y_valid):
            
        roc_score_train = []
        roc_score_valid = []
        f1_score_train = []
        f1_score_valid = []
        models = []
    
        for typ in tqdm(self.cov_types, desc='Search the optimal parameter...'):
        
            model = GaussianMixture(n_components=self.n_clusters, covariance_type=typ, n_init=self.n_init, 
                                    tol=self.tol, max_iter=self.max_iter, init_params=self.init_params,
                                    reg_covar=self.reg_covar, random_state=self.random_state)
            model.fit(X_train, y_train)

            pred_train = model.predict(X_train)
            pred_valid = model.predict(X_valid)  

            pred_train_proba = model.predict_proba(X_train)
            pred_valid_proba = model.predict_proba(X_valid)

            ROC_score_train = roc_auc_score(y_train, pred_train)
            ROC_score_valid = roc_auc_score(y_valid, pred_valid)
            models.append(model)
        
            # Преобразование метрик (в случае если классы были присвоены не согласно истинной метрике (инвертация))
            if ROC_score_train < 0.5:
                pred_train = 1 - pred_train
                pred_valid = 1 - pred_valid
                pred_train_proba = 1 - pred_train_proba
                pred_valid_proba = 1 - pred_valid_proba
                ROC_score_train = 1 - ROC_score_train
                ROC_score_valid = 1 - ROC_score_valid

            F1_score_train = round(f1_score(y_train, pred_train),3)
            F1_score_valid = round(f1_score(y_valid, pred_valid),3)
            
            roc_score_train.append(round(ROC_score_train,3))
            roc_score_valid.append(round(ROC_score_valid,3))
            f1_score_train.append(F1_score_train)
            f1_score_valid.append(F1_score_valid)
            
            print ('===Covariance_type=== = ', typ)                
            print ('ROC_Score_train = ', round(ROC_score_train, 3))
            print ('ROC_Score_valid = ', round(ROC_score_valid, 3))
            print ('F1_score_train = ', round(F1_score_train, 3)) 
            print ('F1_score_valid = ', round(F1_score_valid, 3), '\n')
                
        best_model = models[f1_score_valid.index(max(f1_score_valid))] 

        pred_value = best_model.predict(X_train)
        pred_train = best_model.predict_proba(X_train)
        pred_valid = best_model.predict_proba(X_valid)  

        if roc_auc_score(y_train, pred_value) < 0.5:
            pred_train = 1 - pred_train
            pred_valid = 1 - pred_valid   
        
        if Path(self.out_path).exists():
            dump(best_model, Path(self.out_path)/'Best_model_GM.joblib')

        else:
            path = Path(self.out_path)
            path.mkdir()
            dump(best_model, path/'Best_model_GM.joblib')

        return best_model, roc_score_train, roc_score_valid, f1_score_train, f1_score_valid, pred_train, pred_valid

    def load_model(self):

        print ('Loading model...')

        best_model = load(Path(self.out_path)/'Best_model_GM.joblib')    

        print (best_model)

        return best_model  

    #Функция сохранения результатов тренировки и валидации в csv файлы
    def save_results(self, roc_train, roc_valid, f1_train, f1_valid):
        
        frame_results = pd.DataFrame({'F1_score_train' : f1_train, 'F1_score_valid': f1_valid, \
                                  'Roc_score_train': roc_train, 'Roc_score_valid': roc_valid}, \
                                 index=self.cov_types)   

        if Path(self.out_path).exists():
            frame_results.to_csv(Path(self.out_path)/'Metrics_train_and_validation.csv', index=True)
        else:
            path = Path(self.out_path)
            path.mkdir()
            frame_results.to_csv(path/'Metrics_train_and_validation.csv', index=True)            
        return frame_results   


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

    def testing(self, model, test_wells):

        print ('Testing_model...')
        
        #Список целевой метрики для каждой тестовой скважины
        y_finals = [well['goal'] for well in test_wells]
        well_id = [int(well['well id'].unique()[0]) for well in test_wells]
        
        #Список всех тестовых скважин, очищенных от ненужных столбцов
        X_finals = [well.drop(['well id', 'depth, m', 'lith', 'goal'], axis=1) for well in test_wells]
        
        #Предсказанные значения 0,1 для каждой тестовой скважины
        pred_finals = [model.predict(final) for final in X_finals]
        
        #Предсказанные значения вероятности целевой метрики для каждой тестовой скважины
        pred_finals_proba = [model.predict_proba(final) for final in X_finals]
        
        #Расчёт метрик roc_auc_score
        roc_score_finals = [round(roc_auc_score(y_true, y_pred), 4) for y_true, y_pred in zip(y_finals, pred_finals)]

        if roc_score_finals[0] < 0.5:

            pred_finals = [1 - pred for pred in pred_finals]
            pred_finals_proba = [1 - finalProba for finalProba in pred_finals_proba]
            roc_score_finals = [1 - roc for roc in roc_score_finals]

        f1_score_finals = [round(f1_score(y_true, y_pred), 4) for y_true, y_pred in zip(y_finals, pred_finals)]
            
        for idx, Roc_score in zip(well_id, roc_score_finals):
            print (f'ROC Score_final_{idx} \t= ', Roc_score)
            
        print ('\n')
        
        for idx, F1_score in zip(well_id, f1_score_finals):
            print (f'F1 Score_final_{idx} \t= ', F1_score)
            
        return y_finals, pred_finals, pred_finals_proba, roc_score_finals, f1_score_finals
            
            
        
        
