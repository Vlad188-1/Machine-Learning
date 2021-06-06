import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
from tqdm import tqdm

from keras.models import Sequential, Model, load_model
from keras.layers import Dense,BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


class AE_NN():

    def __init__(self, lr=None, epochs_NN=None, epochs_AE=None, batch_size_NN=None, batch_size_AE=None, out_path=None, callback_epochs=None):

        self.lr = lr
        self.epochs_NN = epochs_NN
        self.epochs_AE = epochs_AE
        self.batch_size_NN = batch_size_NN
        self.batch_size_AE = batch_size_AE
        self.out_path = out_path
        self.callback_epochs = callback_epochs

    def training_and_validation(self, X_train, y_train, X_valid, y_valid):
        
        np.random.seed(0)
        
        print ('Training Autoencoder...')
        
        model = Sequential()
        model.add(Dense(45, activation='relu', input_shape=(len(X_train.columns),)))
        model.add(BatchNormalization())
        model.add(Dense(30, activation='relu'))
        model.add(BatchNormalization())
        
        model.add(Dense(20, activation='linear', name="bottleneck"))
        
        model.add(BatchNormalization())
        model.add(Dense(30, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(45, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(len(X_train.columns), activation='linear'))
        model.compile(loss='mean_squared_error', optimizer = Adam())

        history = model.fit(X_train, X_train, batch_size=self.batch_size_AE, epochs=self.epochs_AE, verbose=1, validation_data=(X_valid, X_valid))
        
        
        encoder = Model(model.input, model.get_layer('bottleneck').output)
        nc_train = encoder.predict(X_train)
        nc_valid = encoder.predict(X_valid)
        
        encoder.save(Path(self.out_path)/'autoencoder.h5')
        
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=self.callback_epochs), ModelCheckpoint(filepath=Path(self.out_path)/'NN_model.h5', 
                                                                                        monitor='val_loss', save_best_only=True)]
        print ('\nTraining NN...')
        
        model_NN = Sequential()
        model_NN.add(Dense(256, activation='relu',  input_shape=(nc_train.shape[1],)))
        model_NN.add(BatchNormalization())
        model_NN.add(Dropout(0.35))
        model_NN.add(Dense(128, activation='relu'))
        model_NN.add(BatchNormalization())
        model_NN.add(Dropout(0.35))
        model_NN.add(Dense(64, activation='relu'))
        model_NN.add(Dense(1, activation='sigmoid'))
        
        model_NN.compile(optimizer=Adam(learning_rate=self.lr), loss='binary_crossentropy', metrics=['binary_accuracy'])
        
        
        history = model_NN.fit(nc_train, y_train, batch_size=self.batch_size_NN, epochs=self.epochs_NN, callbacks=callbacks_list, validation_data=(nc_valid, y_valid))
        
        pred_train = model_NN.predict_classes(nc_train)
        pred_valid = model_NN.predict_classes(nc_valid)
        
        pred_train_proba = model_NN.predict_proba(nc_train)
        pred_valid_proba = model_NN.predict_proba(nc_valid)
        
        Roc_score_train = round(roc_auc_score(y_train, pred_train),3)
        Roc_score_valid = round(roc_auc_score(y_valid, pred_valid),3)
        F1_score_train = round(f1_score(y_train, pred_train),3)
        F1_score_valid = round(f1_score(y_valid, pred_valid),3)
        
        print ('Roc_score_train = ', Roc_score_train)
        print ('Roc_score_valid = ', Roc_score_valid)
        print ('F1_score_train = ', F1_score_train)
        print ('F1_score_valid = ', F1_score_valid)
        
        
        def save_results(roc_train, roc_valid, f1_train, f1_valid):
            
            frame_results = pd.DataFrame({'F1_score_train' : f1_train, 'F1_score_valid': f1_valid, 
                                  'Roc_score_train': roc_train, 'Roc_score_valid': roc_valid}, index=[1])   
            if Path(self.out_path).exists():
                frame_results.to_csv(Path(self.out_path)/'Metrics_train_and_validation.csv')
            else:
                path = Path(self.out_path)
                path.mkdir()
                frame_results.to_csv(path/'Metrics_train_and_validation.csv')            
            
        save_results(Roc_score_train, Roc_score_valid, F1_score_train, F1_score_valid)
        
        return history, pred_train_proba, pred_valid_proba
        
        
    def testing(self, encoder, NN_model, test_wells):

        print ('Testing_model...')
        
        #Список целевой метрики для каждой тестовой скважины
        y_finals = [well['goal'] for well in test_wells]
        well_id = [int(well['well id'].unique()[0]) for well in test_wells]
        
        #Список всех тестовых скважин, очищенных от ненужных столбцов
        X_finals = [well.drop(['well id', 'depth, m', 'lith', 'goal'], axis = 1) for well in test_wells]
        
        #Данные пропускаются через автоэнкодер
        nc_finals = [encoder.predict(final) for final in X_finals]
        
        #Предсказанные значения 0,1 для каждой тестовой скважины
        pred_finals = [NN_model.predict_classes(final) for final in nc_finals]
        
        #Предсказанные значения вероятности целевой метрики для каждой тестовой скважины
        pred_finals_proba = [NN_model.predict_proba(final) for final in nc_finals]
        
        #Расчёт метрик roc_auc_score
        roc_score_finals = [round(roc_auc_score(y_true, y_pred),4) for y_true, y_pred in zip(y_finals, pred_finals)]

        f1_score_finals = [round(f1_score(y_true, y_pred),4) for y_true, y_pred in zip(y_finals, pred_finals)]
            
        for idx, Roc_score in zip(well_id, roc_score_finals):
            print (f'ROC Score_final_{idx} \t= ', Roc_score)
            
        print ('\n')
        
        for idx, F1_score in zip(well_id, f1_score_finals):
            print (f'F1 Score_final_{idx} \t= ', F1_score)
            
        return y_finals, pred_finals, pred_finals_proba, roc_score_finals, f1_score_finals
        
        
    def save_results_test(self, test_wells, roc_score, F1_score, pred_finals):

        well_id = [int(well['well id'].unique()[0]) for well in test_wells]
        well_id = [f'Well_{i}' for i in well_id]
    
        dict_f1_score = dict(zip(well_id, F1_score))
    
        frame_well_id = pd.DataFrame(list(dict_f1_score.keys()), index=range(len(dict_f1_score)), columns=['Well_id'])
        frame_f1_score = pd.DataFrame(list(dict_f1_score.values()), index=range(len(dict_f1_score)), columns=['F1_score'])
        frame_roc_score = pd.DataFrame(roc_score, columns=['Roc_score'])
        frame_results = pd.concat([frame_f1_score, frame_well_id, frame_roc_score], axis=1)
        frame_results = frame_results.sort_values(by='F1_score')
    
        if Path(self.out_path).exists():
            frame_results.to_csv(Path(self.out_path)/'Metrics_testing_wells.csv', index=False)
        else:
            path = Path(self.out_path)
            path.mkdir()
            frame_results.to_csv(path/'Metrics_testing_wells.csv', index=False)
            
        for idx, final in zip(well_id, pred_finals):
            frame_pred_test = pd.DataFrame({f"{idx}":np.array(final).ravel()}, index=range(len(np.array(final).ravel())))
            name_file = f"Pred_test_{idx}.csv"
            frame_pred_test.to_csv(Path(self.out_path)/name_file, index=False)