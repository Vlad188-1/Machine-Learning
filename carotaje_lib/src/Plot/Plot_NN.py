import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score
from sklearn import metrics
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pathlib import Path
from tqdm import tqdm


def curve_train_and_valid(history, args):

    print ('Plotting training and validation curve...')

    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = np.arange(1, len(acc)+1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=['$\Large\\textbf{Accuracy}$', '$\Large\\textbf{Loss}$'], 
                                                                vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=epochs, y=acc, mode='lines', name='Training', marker_color='blue'),1,1)
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Validation', marker_color='red'),1,1)
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', showlegend=False, marker_color='blue'),1,2)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', showlegend=False, marker_color='red'),1,2)

    fig.add_trace(go.Scatter(x=[epochs[val_loss.index(min(val_loss))]], 
                        y=[val_acc[epochs[val_loss.index(min(val_loss))-1]]], marker_color='darkgreen', 
                        name='Accuracy для<br> минимального loss'),1,1)

    fig.add_trace(go.Scatter(x=[epochs[val_loss.index(min(val_loss))]], 
                        y=[val_loss[epochs[val_loss.index(min(val_loss))-1]]], marker_color='black', 
                        name='Минимальный loss'),1,2)

    fig.add_shape(type="line", row=1, col=2,
            x0=epochs[val_loss.index(min(val_loss))], y0=0.3, x1=epochs[val_loss.index(min(val_loss))], y1=0.1,
            line=dict(color="black",width=2,dash="dashdot"))

    fig.add_shape(type="line", row=1, col=1,
            x0=epochs[val_loss.index(min(val_loss))], y0=0.96, x1=epochs[val_loss.index(min(val_loss))], y1=0.86,
            line=dict(color="black",width=2,dash="dashdot"))

    fig.update_layout(plot_bgcolor='white', title=dict(text='<b>Обучение и валидация нейронной сети', x=0.45, font=dict(size=20)), 
                  legend=dict(font=dict(size=14)), height=700, width=1450, showlegend=False)
    fig.update_xaxes(row=1, col=1, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey', 
                                        title=dict(text='Epochs', font=dict(size=20)))
    fig.update_xaxes(row=1, col=2, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey', 
                                                                    title=dict(text='Epochs', font=dict(size=20)))

    fig.update_yaxes(row=1, col=1, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')
    fig.update_yaxes(row=1, col=2, tickfont=dict(size=20), gridcolor='lightgrey', zerolinecolor='lightgrey')

    if Path(args.output).exists():
        url_new = str(args.output)
        fig.write_image(url_new + '/Train-valid-curve.png')

    else:
        path = Path(args.output)
        path.mkdir()
        fig.write_image(str(path)+'/Train-valid-curve.png')
        
        
        
def roc_pr_curve_train_and_valid(y_train, y_valid, pred_train_proba, pred_valid_proba, args): 

    """
    #Построение roc_auc и precision_recall кривых на тренировочных и валидационных скважинах
    """
    print ('Plotting roc-curve, precision-recall-curve...')

    fpr_train, tpr_train, _ = metrics.roc_curve(y_train.values, pred_train_proba[:,0])
    fpr_valid, tpr_valid, _ = metrics.roc_curve(y_valid.values, pred_valid_proba[:,0])

    precision_train, recall_train, _ = metrics.precision_recall_curve(y_train.values, pred_train_proba[:,0])
    precision_valid, recall_valid, _ = metrics.precision_recall_curve(y_valid.values, pred_valid_proba[:,0])

    fig = make_subplots(rows=1, cols=2, subplot_titles=['$\Large\\textbf{ROC-curve}$', '$\Large\\textbf{Precision-recall-curve}$'])

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
        fig.write_image(url_new + '/ROC-PR-curve_train-valid.png')

    else:
        path = Path(args.output)
        path.mkdir()
        fig.write_image(str(path)+'/ROC-PR-curve_train-valid.png')    

        
def plot_curve_testing(test_wells, pred_finals_proba, y_finals, args):

    """
    #Построение roc_auc и precision_recall кривых на тестовых скважинах
    """

    well_id = [int(well['well id'].unique()[0]) for well in test_wells]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    fig = make_subplots(rows=1, cols=2, subplot_titles=['$\Large\\textbf{ROC-curve}$', '$\Large\\textbf{Precision-recall-curve}$'])

    for y_final, final_proba, well, col in tqdm(zip(y_finals, pred_finals_proba, well_id, colors), desc='Plotting_test_curve...', total=len(test_wells)):
            
        fpr, tpr, _ = metrics.roc_curve(y_final.values, final_proba)
        prec, rec, _ = metrics.precision_recall_curve(y_final.values, final_proba)

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
        fig.write_image(str(url_new)+'/ROC-PR-curve_test.png')
        
