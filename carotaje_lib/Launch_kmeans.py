#===Сторонние бибилотеки===
import configargparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
#===Мои библиотеки===
from src.Models.model_kmeans import KM
from src.Processing_data.processing_for_kmeans import process
from src.Plot.plot_markup import plot_test_wells
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
    x_train, y_train, test_wells, for_final_model = process(x_1)
    # Вызывем класс KM чтобы обучить алгорит kmeans
    Model = KM(n_clusters=args.n_clusters, random_state=42, max_iter=args.max_iter, out_path=args.output)
    Model.training(x_train, y_train)
    pred_finals = Model.testing(test_wells)
    plot_test_wells(test_wells, pred_finals, args, method_name='Kmeans')


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--data_csv', type=str, help='Path to data')
    parser.add_argument('--output', default='Results\Exp_kmeans\Exp_1', type=str, help='Path to save results')
    parser.add_argument('--n-clusters', default=2, type=int, help='Count of clusters')
    parser.add_argument('--max-iter', default=500, type=int, help='Max iter for train')

    args = parser.parse_args()
    
    print (args)
    
    train(args)

    