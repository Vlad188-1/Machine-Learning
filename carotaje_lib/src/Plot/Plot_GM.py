import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score
from sklearn import metrics
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pathlib import Path
from tqdm import tqdm


def curve_train_and_valid(y_train, y_valid, pred_train_proba, pred_valid_proba, args): 

    """
    #Построение roc_auc и precision_recall кривых на тренировочных и валидационных скважинах
    """

    fpr_train, tpr_train, _ = metrics.roc_curve(y_train.values, pred_train_proba[:,1])
    fpr_valid, tpr_valid, _ = metrics.roc_curve(y_valid.values, pred_valid_proba[:,1])

    precision_train, recall_train, _ = metrics.precision_recall_curve(y_train.values, pred_train_proba[:,1])
    precision_valid, recall_valid, _ = metrics.precision_recall_curve(y_valid.values, pred_valid_proba[:,1])

    fig = make_subplots(rows=1, cols=2, subplot_titles=['$\Large\\textbf{Accuracy}$', '$\Large\\textbf{Loss}$'], 
                                                                vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=fpr_train, y=tpr_train, marker_color='red',
                                    name='Train_roc, \t S_train={}'.format(round(metrics.auc(fpr_train, tpr_train),3))),1,1)
    fig.add_trace(go.Scatter(x=fpr_valid, y=tpr_valid, marker_color='blue',
                                    name='Valid_roc, \t S_valid={}'.format(round(metrics.auc(fpr_valid, tpr_valid),3))),1,1)

    fig.add_trace(go.Scatter(x=recall_train, y=precision_train, marker_color='red',
                            name='Train_pr, \t S_train={}'.format(round(metrics.auc(recall_train, precision_train),3))),1,2)
    fig.add_trace(go.Scatter(x=recall_valid, y=precision_valid, marker_color='blue',
                            name='Valid_pr, \t S_valid={}'.format(round(metrics.auc(recall_valid, precision_valid),3))),1,2)                                
    
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='black'), showlegend=False),1,1)
    fig.add_trace(go.Scatter(x=[0,1], y=[1,0], line=dict(dash='dash', color='black'), showlegend=False),1,2)
    
    fig.update_layout(plot_bgcolor='white', height=700, width=1450,
                                xaxis=dict(title=dict(text = 'False Positive Rate', font=dict(size=20)), tickfont=dict(size=20), gridcolor = 'lightgrey', \
                                            zerolinecolor = 'lightgrey', constrain='domain'), \
                                yaxis=dict(title=dict(text='True Positive Rate', font=dict(size=20)),tickfont=dict(size=20), gridcolor = 'lightgrey', \
                                            zerolinecolor = 'lightgrey'),  
                                xaxis2 = dict(title=dict(text='Recall', font=dict(size=20)), tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey'),
                                yaxis2 = dict(title=dict(text='Precision', font=dict(size=20)), tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey'))

    if Path(args.output).exists():
        url_new = str(args.output)
        fig.write_image(url_new + '/Roc_curve_train_and_validation.png')

    else:
        path = Path(args.output)
        path.mkdir()
        fig.write_image(str(path)+'/Roc_curve_train_and_validation.png')


def plot_curve_testing(test_wells, pred_finals_proba, y_finals, args):

    """
    #Построение roc_auc и precision_recall кривых на тестовых скважинах
    """

    well_id = [int(well['well id'].unique()[0]) for well in test_wells]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    fig = make_subplots(rows=1, cols=2, subplot_titles=['$\Large\\textbf{Accuracy}$', '$\Large\\textbf{Loss}$'], 
                                                                vertical_spacing=0.1)

    for y_final, final_proba, well, col in tqdm(zip(y_finals, pred_finals_proba, well_id, colors), desc='Plotting_test_curve...', total=len(test_wells)):
            
        fpr, tpr, _ = metrics.roc_curve(y_final.values, final_proba[:,1])
        prec, rec, _ = metrics.precision_recall_curve(y_final.values, final_proba[:,1])

        fig.add_trace(go.Scatter(x=fpr, y=tpr, marker_color=col, name=f'Well_{well}, \t S_roc={round(metrics.auc(fpr, tpr),3)}'),1,1) 
        fig.add_trace(go.Scatter(x=rec, y=prec, marker_color=col, name=f'Well_{well}, \t S_pr={round(metrics.auc(rec, prec),3)}'),1,2)

    fig.update_layout(plot_bgcolor='white', height=700, width=1450,
                                xaxis=dict(title=dict(text = 'False Positive Rate', font=dict(size=20)), tickfont=dict(size=20), gridcolor = 'lightgrey', \
                                            zerolinecolor = 'lightgrey', constrain='domain'), \
                                yaxis=dict(title=dict(text='True Positive Rate', font=dict(size=20)),tickfont=dict(size=20), gridcolor = 'lightgrey', \
                                            zerolinecolor = 'lightgrey'),  
                                xaxis2 = dict(title=dict(text='Recall', font=dict(size=20)), tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey'),
                                yaxis2 = dict(title=dict(text='Precision', font=dict(size=20)), tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey'))

    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='black'), showlegend=False),1,1)
    fig.add_trace(go.Scatter(x=[0,1], y=[1,0], line=dict(dash='dash', color='black'), showlegend=False),1,2)   
            
    if Path(args.output).exists():
        url_new = Path(args.output)
        fig.write_image(str(url_new)+'/Roc_curve_testing.png')

    else:
        path = Path(args.output)
        path.mkdir()
        fig.write_image(str(path)+'/Roc_curve_testing.png')


def plot_bar_results_train(new_frame, args):

    """
    Построение гистограммы качества roc_score и f1_score на тренировочных и валидационных скважинах
    """


    fig = go.Figure()
    fig.add_trace(go.Bar(x=new_frame.index, y=new_frame['Roc_score_train'], \
                            text=[f'{round(i*100,3)}%' for i in  new_frame['Roc_score_train']], \
                            textfont=dict(color='white', size=20), 
                            marker_color='darkblue',width=[0.2], textposition='auto', name='Roc_score_train'))

    fig.add_trace(go.Bar(x=new_frame.index, y=new_frame['Roc_score_valid'], \
                            text=[f'{round(i*100,3)}%' for i in  new_frame['Roc_score_valid']], \
                            textfont=dict(color='white',size=20), 
                            marker_color='skyblue', width=[0.2], textposition='auto', name='Roc_score_valid'))

    fig.add_trace(go.Bar(x=new_frame.index, y=new_frame['F1_score_train'], \
                            text=[f'{round(i*100,3)}%' for i in  new_frame['F1_score_train']],
                            marker_color='green', textfont=dict(color='white', size=20), \
                            textposition='auto', name='F1_score_train'))

    fig.add_trace(go.Bar(x=new_frame.index, y=new_frame['F1_score_valid'], \
                            text=[f'{round(i*100,3)}%' for i in  new_frame['F1_score_valid']], 
                            marker_color='olive', textfont=dict(color='white', size=20), \
                            textposition='auto', name='F1_score_valid'))

    fig.update_layout(plot_bgcolor='white', \
                        title=dict(text='<b>Поиск оптимального параметра covariance_type по качеству F1_score и Roc_score', \
                                    x=0.5, y=0.95, font=dict(size=20)), 
                        legend=dict(font=dict(size=16)), width=1050, height=800)
    
    fig.update_xaxes(gridcolor='lightgrey', tickfont=dict(size=20), zerolinecolor='lightgrey', \
                        title=dict(text='<b>covariance_type', font=dict(size=20)))
    fig.update_yaxes(gridcolor='lightgrey', tickfont=dict(size=25), zerolinecolor='lightgrey', \
                        range=[0,1], title=dict(text='<b>F1_score, Roc_score', font=dict(size=16)))

    if Path(args.output).exists():
        url_new = str(args.output)
        fig.write_image(str(url_new)+'/Bar_results_params.png')

    else:
        path = Path(args.output)
        path.mkdir()
        fig.write_image(str(path)+'/Bar_results_params.png')