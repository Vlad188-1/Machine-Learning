from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px


def plot_test_wells(test_wells, pred_finals, args, method_name='Kmeans'):
    
    for x_test, y_test_pred in tqdm(zip(test_wells, pred_finals), desc='Plotting well logs...', total=len(pred_finals)):
        
        x_test.index = range(len(x_test))
        x_test = pd.concat([x_test, pd.DataFrame(y_test_pred, columns=['goal_predict'])], axis=1)

        name_columns = ['ALPS']

        frame = x_test   
        rows, cols= 1,7

        Fig = make_subplots(rows = rows, cols = cols, subplot_titles = [k for k in ['Разметка', method_name]],
                            horizontal_spacing= 0.015, shared_yaxes=True)

        fig_1 = px.area(frame, x = frame[name_columns[0]], y = (-1)*frame["depth, m"], color = frame['goal'],
                                    orientation = 'h', width = 700, height = 900, 
                                    color_discrete_map = {0:'peru', 1:'grey'})        
        for j in range(len(frame['goal'].unique())):
            Fig.add_trace(fig_1['data'][j],1,1)

        fig_2 = px.area(frame, x = frame[name_columns[0]], y = (-1)*frame["depth, m"], color = frame['goal_predict'], 
                                    orientation = 'h', width = 700, height = 900,
                                    color_discrete_map = {0:'peru', 1:'grey'})            
        for j in range(2):
            Fig.add_trace(fig_2['data'][j],1,2)

        for i in range(1,cols+1):
            Fig.update_xaxes(range=[0, 1], showgrid=True, row=1, col=i, color = 'black', nticks = 10, gridcolor = 'lightgrey')
            Fig.update_yaxes(showgrid=True, row=1, col=i, color = 'black', nticks = 50, gridcolor = 'lightgrey')        
            Fig.update_layout(legend_orientation = 'h', width = 1100, height = 1400, margin = dict(l=0, r=0, t=50, b=0))  
            
        Fig.update_layout(title=dict(text=f'<b>Well_{int(frame["well id"].unique()[0])}', x=0.3), plot_bgcolor='white',
                         showlegend=False)

        if Path(args.output).exists():
            url_new = str(args.output)
            Fig.write_image(url_new+f'//Test_well_{int(frame["well id"].unique()[0])}.png')

        else:
            path = Path(args.output)
            path.mkdir()
            Fig.write_image(str(path)+f'//Test_well_{int(frame["well id"].unique()[0])}.png')