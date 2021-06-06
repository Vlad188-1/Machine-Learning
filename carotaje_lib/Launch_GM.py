#===Сторонние бибилотеки===
import configargparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
#===Мои библиотеки===
from src.Models.model_GM import GM
from src.Processing_data.processing_for_GM import process
from src.Plot.plot_markup import plot_test_wells
from src.Plot.Plot_GM import curve_train_and_valid, plot_bar_results_train, plot_curve_testing
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

    Model = GM(n_clusters=args.n_clusters, random_state=123, n_init=args.n_init, tol=args.tol, 
                            init_params=args.init_params, max_iter=args.max_iter, reg_covar=args.reg_covar,
                            out_path=args.output)

    if args.status == 'train':
        best_model, roc_score_train, roc_score_valid, f1_score_train, f1_score_valid, pred_train, pred_valid = \
                                                                Model.training_and_validation(x_train, y_train,
                                                                                         x_valid, y_valid)
        frame_results = Model.save_results(roc_score_train, roc_score_valid, f1_score_train, f1_score_valid)
        print (best_model)
        print ('Plotting of results training and validation...')
        plot_bar_results_train(frame_results, args)
        curve_train_and_valid(y_train, y_valid, pred_train, pred_valid, args)
        print ('Success!')

    elif args.status == 'test':
        best_model = Model.load_model()
        y_finals, pred_finals, pred_finals_proba, roc_score_finals, f1_score_finals = Model.testing(best_model, test_wells)
        Model.save_results_test(test_wells, roc_score_finals, f1_score_finals, pred_finals)
        plot_curve_testing(test_wells, pred_finals_proba, y_finals, args)
        plot_test_wells(test_wells, pred_finals, args, method_name='GM')
        print (best_model)
        print ('Success!')
    else:
        print ('Status is not assigned. Model is not tested!')

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--data_csv', required=True, type=str, help='Path to data')
    parser.add_argument('--status', required=True, type=str, default='load', help='Указать предпочтение (тренировать новую модель или загрузить уже лучшую натренированную')
    parser.add_argument('--output', default='./', type=str, help='Path to save results (По дефолту файлы сохраняются просто в папку Results, рекомендуется создавать внутри папки Results новую папку - для этого достаточно в аргумент output подать название новой создаваемой папки)')
    parser.add_argument('--n-clusters', default=2, type=int, help='Count of clusters')
    parser.add_argument('--max_iter', default=100, type=int, help='The number of iterations')
    parser.add_argument('--tol', default=0.001, type=float, help='')
    parser.add_argument('--reg_covar',  default=0.000001, type=float, help='')
    parser.add_argument('--n_init', default=1, type=int, help='The number of initilizations')
    parser.add_argument('--init_params', default='kmeans', type=str, help='The approach of initilization parameters')

    args = parser.parse_args()
    
    print (args)
    
    train(args)