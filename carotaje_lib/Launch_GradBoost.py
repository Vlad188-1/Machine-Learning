#===Сторонние бибилотеки===
import configargparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
#===Мои библиотеки===
from src.Models.model_GradBoost import GradBoost
from src.Processing_data.processing_for_GradBoost import process
from src.Plot.plot_markup import plot_test_wells
from src.Plot.Plot_GradBoost import curve_train_and_valid, roc_pr_curve_train_and_valid, plot_curve_testing
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
    x_train, x_valid, y_train, y_valid, test_wells, for_final_model = process(x_1)

    Model = GradBoost(learning_rate=args.learning_rate, max_depth=args.max_depth, random_state=123,
                            out_path=args.output)

    if args.status == 'train':
        best_model, roc_score_train, roc_score_valid, f1_score_train, f1_score_valid, pred_train, pred_valid, loss_train, loss_valid = Model.training_and_validation(x_train, y_train, x_valid, y_valid)
        frame_results = Model.save_results(roc_score_train, roc_score_valid, f1_score_train, f1_score_valid)
        print (best_model)
        print ('Plotting of results training and validation...')
        #plot_bar_results_train(frame_results, args)
        curve_train_and_valid(Model.n_estimators, f1_score_train, f1_score_valid, loss_train, loss_valid, args)
        roc_pr_curve_train_and_valid(y_train, y_valid, pred_train, pred_valid, args)
        print ('Success!')

    elif args.status == 'test':
        best_model = Model.load_model()
        y_finals, pred_finals, pred_finals_proba, roc_score_finals, f1_score_finals = Model.testing(best_model, test_wells)
        Model.save_results_test(test_wells, roc_score_finals, f1_score_finals, pred_finals)
        plot_curve_testing(test_wells, pred_finals_proba, y_finals, args)
        plot_test_wells(test_wells, pred_finals, args, method_name='GradBoost')
        print ('Success!')
    else:
        print ('Status is not assigned. Model is not tested!')

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--data_csv', required=True, type=str, help='Path to data')
    parser.add_argument('--status', required=True, type=str, default='load', help='Указать предпочтение (тренировать новую модель или загрузить уже лучшую натренированную')
    parser.add_argument('--output', default='./', type=str, help='Path to save results (По дефолту файлы сохраняются просто в папку Results, рекомендуется создавать внутри папки Results новую папку - для этого достаточно в аргумент output подать название новой создаваемой папки)')
    parser.add_argument('--n_estimators', default=100, type=int, help='Число базовых алгоритмов')
    parser.add_argument('--max_depth', default=2, type=int, help='Максимальная глубина базового алгоритма')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Коэффициент обучения')
    # parser.add_argument('--reg_covar',  default=0.000001, type=float, help='')
    # parser.add_argument('--n_init', default=1, type=int, help='The number of initilizations')
    # parser.add_argument('--init_params', default='kmeans', type=str, help='The approach of initilization parameters')

    args = parser.parse_args()
    
    print (args)
    
    train(args)