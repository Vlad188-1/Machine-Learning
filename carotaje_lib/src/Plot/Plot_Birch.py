import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pathlib import Path

def plot_space_parameters(all_branch, all_tresholds, all_scores, args):

    colorscales = [[0, 'red'], [0.5, 'orange'], [1, 'lime']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_branch, y=all_tresholds, mode='markers', 
                            marker = dict(color = all_scores, size=10, colorscale=colorscales, colorbar=dict(title="<b>Оценка <br> F1_score"), 
                line=dict(color='black', width=1.5)), showlegend=False, opacity=1))

    fig.update_layout(title=dict(text='<b>Поиск оптимальных параметров', x=0.5, font=dict(size=20)), colorscale=dict(sequential=colorscales), width = 950, height = 750, 
                                plot_bgcolor='white', coloraxis=dict(showscale=True),
                                legend=dict(font=dict(size=20)))

    fig.update_xaxes(title = dict(text='<b>branching_factor', font=dict(size=20)), tickfont=dict(size=20), gridcolor ='lightgrey', zerolinecolor = 'grey')
    fig.update_yaxes(title=dict(text='<b>threshold', font=dict(size=20)), tickfont=dict(size=20), gridcolor ='lightgrey', zerolinecolor = 'grey')
    
    if Path(args.output).exists():
        url_new = str(args.output)
        fig.write_image(url_new + '/Search_best_parameters.png')

    else:
        path = Path(args.output)
        path.mkdir()
        fig.write_image(str(path) + '/Search_best_parameters.png')