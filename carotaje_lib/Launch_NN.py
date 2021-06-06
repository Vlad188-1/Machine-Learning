#===Сторонние бибилотеки===
import configargparse
import pandas as pd
from pathlib import Path
from keras.models import load_model
from tqdm import tqdm
#===Мои библиотеки===
from src.Processing_data.processing_for_NN import process
from src.Plot.plot_markup import plot_test_wells
from src.Plot.Plot_NN import curve_train_and_valid, roc_pr_curve_train_and_valid, plot_curve_testing
from src.Models.model_NN import NN
#=======================

def train(args):

    data = pd.read_csv(args.data_csv, sep = ';')
    
    # Replace decimal comma to dot and convert it
    for i in tqdm(data.columns, desc='Read files...'):
        if i == 'lith':
            continue
        else:
            data[i] = data[i].replace(',','.', regex=True).astype(float)
 
    x_1 = data.sort_values(by=['well id', 'depth, m']).dropna()
    
    # Cleaning data and feature engineering
    x_train, x_valid, y_train, y_valid, test_wells = process(x_1)

    Model = NN(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,out_path=args.output, callback_epochs=args.callback_epochs)

    if args.status == 'train':
        history, pred_train_proba, pred_valid_proba = Model.training_and_validation(x_train, y_train, x_valid, y_valid)
        roc_pr_curve_train_and_valid(y_train, y_valid, pred_train_proba, pred_valid_proba, args)
        curve_train_and_valid(history, args)
        print ('Success!')

    elif args.status == 'test':
        best_model = load_model(Path(args.output)/'best_model.h5')
        y_finals, pred_finals, pred_finals_proba, roc_score_finals, f1_score_finals = Model.testing(best_model, test_wells)
        Model.save_results_test(test_wells, roc_score_finals, f1_score_finals, pred_finals)
        plot_curve_testing(test_wells, pred_finals_proba, y_finals, args)
        plot_test_wells(test_wells, pred_finals, args, method_name='NN')
        print ('Success!')
    else:
        print ('Status is not assigned. Model is not tested!')

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--data_csv', required=True, type=str, help='Path to data')
    parser.add_argument('--status', required=True, type=str, default='load', help='Указать предпочтение (тренировать новую модель или загрузить уже лучшую натренированную')
    parser.add_argument('--output', default='./', type=str, help='Path to save results (По дефолту файлы сохраняются просто в папку Results, рекомендуется создавать внутри папки Results новую папку - для этого достаточно в аргумент output подать название новой создаваемой папки)')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=10, type=int, help='The number of epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='The size of batch')
    parser.add_argument('--callback_epochs', default=10, type=int, help='Если в течении этого количества эпох значение Loss функции существенно не меняется, то прекращается обучение и сохраняется лучшая модель.')

    args = parser.parse_args()
    
    print (args)
    
    train(args)