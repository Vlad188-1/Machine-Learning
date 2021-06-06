import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
from tqdm import tqdm


class KM():
    
    def __init__(self, n_clusters=2, random_state=42, max_iter=500, out_path=None):
        
        self.n_clusters = n_clusters
        self.out_path = out_path
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
        
    def training(self, X_train, y_train):
        
        print ('Training model...')
        
        self.model.fit(X_train, y_train)

        pred_train = self.model.labels_
        
        ROC_score_train = round(roc_auc_score(y_train.values, pred_train),3)
        
        # Запись результатов
        
        def save_files_train(roc, F1):
            
            dict_metrics = {'F1_score_train': F1, 'F1_score_valid': [0], 'Roc_score_train': roc, 'Roc_score_valid': [0]}
            #dict_metrics = {'Train': [roc, F1], 'Valid': ['Нет валидационной выборки']*2}
            new_frame = pd.DataFrame(dict_metrics, index=[0])

            if Path(self.out_path).exists():
                new_frame.to_csv(Path(self.out_path)/'Metrics_train_wells.csv', index=True)

            else:
                path = Path(self.out_path)
                path.mkdir()
                new_frame.to_csv(path/'Metrics_train_wells.csv', index=True)
        
        # Преобразование метрик (в случае если классы были присвоены не согласно истинной метрике (инвертация))
        if ROC_score_train < 0.5:
            pred_train = 1 - pred_train
            ROC_score_train - 1 - ROC_score_train
            
        F1_score_train = round(f1_score(y_train, pred_train),3)
        
        print ('\nROC_Score_train = ', round(ROC_score_train, 3))
        print ('F1_score_train = ', round(F1_score_train, 3), '\n') 
        
        save_files_train(ROC_score_train, F1_score_train)
        return self.model, ROC_score_train, F1_score_train
   
    
    def testing(self, test_wells):
        
        #Список целевой метрики для каждой тестовой скважины
        y_finals = [well['goal'] for well in test_wells]
        id_wells = [int(well['well id'].unique()[0]) for well in test_wells]
        
        #Список всех тестовых скважин, очищенных от ненужных столбцов
        X_finals = [well.drop(['well id', 'depth, m', 'lith', 'goal'], axis = 1) for well in test_wells]
        
        #Предсказанные значения 0,1 для каждой тестовой скважины
        pred_finals = [self.model.predict(final) for final in X_finals]
        
        well_id = [str(int(well['well id'].unique()[0])) for well in test_wells]
        well_id = [f'Well_{i}' for i in well_id]
        
        def save_files_test(roc_score_finals, f1_score_finals, pred_finals):
            # Сохранение метрик в датареймы
            dict_f1_score = dict(zip(well_id, f1_score_finals))
            frame_well_id = pd.DataFrame(list(dict_f1_score.keys()), index=range(len(dict_f1_score)), columns=['Well_id'])
            frame_f1_score = pd.DataFrame(list(dict_f1_score.values()), index=range(len(dict_f1_score)), columns=['F1_score'])
            frame_roc_score = pd.DataFrame(roc_score_finals, columns=['Roc_score'])
            new_frame = pd.concat([frame_f1_score, frame_well_id, frame_roc_score], axis=1)
            new_frame = new_frame.sort_values(by='F1_score')

            if Path(self.out_path).exists():
                new_frame.to_csv(Path(self.out_path)/'Metrics_test_wells.csv', index=False)
            else:
                new_frame.to_csv(Path(self.out_path)/'Metrics_test_wells.csv', index=False)
                
            for idx, final in zip(well_id, pred_finals):
                frame_pred_test = pd.DataFrame({f"{idx}":final}, index=range(len(final)))
                name_file = f"Pred_test_{idx}.csv"
                frame_pred_test.to_csv(Path(self.out_path)/name_file, index=False)
                
        roc_score_finals = [round(roc_auc_score(y_true, y_pred),4) for y_true, y_pred in zip(y_finals, pred_finals)]
        if roc_score_finals[0] < 0.5:
            pred_finals = [1 - pred for pred in pred_finals]
            pred_finals_proba = [1 - finalProba for finalProba in pred_finals_proba]
            roc_score_finals = [1 - roc for roc in roc_score_finals]

        f1_score_finals = [round(f1_score(y_true, y_pred),4) for y_true, y_pred in zip(y_finals, pred_finals)]
        
        for Roc_score, id_well in tqdm(zip(roc_score_finals, id_wells), desc='Testing model...'):
                print (f'ROC Score_final_{id_well} \t = ', Roc_score)
        for F1_score, id_well in zip(f1_score_finals, id_wells):
                print (f'F1 Score_final_{id_well} \t = ', F1_score)
                
        save_files_test(roc_score_finals, f1_score_finals, pred_finals)
                
        return pred_finals
