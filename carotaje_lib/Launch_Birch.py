#===Сторонние бибилотеки===
import configargparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
#===Мои библиотеки===
from src.Models.model_Birch import BR
from src.Processing_data.processing_for_Birch import process
from src.Plot.plot_markup import plot_test_wells
from src.Plot.Plot_Birch import plot_space_parameters
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

    Model = BR(n_clusters=args.n_clusters, treshold=args.treshold, branching_factor=args.branching_factor,
                                                                        out_path=args.output)
    if args.status == 'train':
        list_max_F1, tresholds_for_max_F1, branch, best_model = Model.training_and_validation(x_train, y_train, x_valid, y_valid)
        plot_space_parameters(branch, tresholds_for_max_F1, list_max_F1, args)
        print ('Success!')
    
    elif args.status == 'test':
        best_model = Model.load_model()
        y_finals, pred_finals, roc_score_finals, f1_score_finals = Model.testing(best_model, test_wells)
        Model.save_results_test(test_wells, roc_score_finals, f1_score_finals, pred_finals)
        plot_test_wells(test_wells, pred_finals, args, method_name='Birch')
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
    parser.add_argument('--treshold', default=0.35, type=float, help='Max diameter of cluster')
    parser.add_argument('--branching_factor', default=23, type=int, help='The number of CF-vectors in non-leaf node')

    args = parser.parse_args()
    
    print (args)
    
    train(args)