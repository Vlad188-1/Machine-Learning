#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots


# ### 1. Построение двух графиков в разных слайдах, но в одинаковых осях

# In[2]:


x = np.arange(-10, 10, 0.1)
y_1 = np.sin(x)
y_2 = np.tanh(x)
y_3 = np.cos(2*x)


# In[3]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_1, marker_color='red', name="$ \large sin(x)$"))
fig.add_trace(go.Scatter(x=x, y=y_2, marker_color='blue', name="$ \large tg(x)$"))
fig.update_layout(plot_bgcolor='white', legend=dict(orientation='h', font=dict(size=14)),
                 xaxis=dict(zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="x")),
                 yaxis=dict(zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="y")),
                 
                 xaxis2 = dict(zerolinecolor='red', gridcolor='red'))

fig.layout.update(
                 updatemenus=[go.layout.Updatemenu(
                 type = "buttons", direction = "right",  active = 0, x = 0.57, y = 1.2,
        buttons = list(
            [
               dict(
                    label='Все графики', method="update", 
                    args=[{"title": "Все графики", "visible": [True, True]}]),
               dict(
                  label = "sin(x)", method = "update",
                  args = [{"title": "График 1", "visible": [True, False]}]),
               dict(
                  label = "tg(x)", method = "update", 
                  args = [{"title": "График 2", "visible": [False, True]}])
            ]
                     )
                          )]
                )
fig.show()


# ### 2. Построение трёх графиков на разных слайдах и в разных осях и с изменением названия осей

# *К аргументу args прописывается параметр "x" или "y"*

# In[4]:


x_2 = np.arange(-5, 5, 0.1)
x_3 = np.arange(-100, 100, 1)


# In[5]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_1, marker_color='red', name="$ \large sin(x)$"))
fig.add_trace(go.Scatter(x=x, y=y_2, marker_color='blue', name="$ \large tg(x)$"))
fig.add_trace(go.Scatter(x=x, y=y_3, marker_color='green', name="$ \large cos(2x)$"))
fig.update_layout(title="<b>Все графики", plot_bgcolor='white', legend=dict(orientation='h', font=dict(size=14)),
                 xaxis=dict(zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="x")),
                 yaxis=dict(zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="y")))

args_1 = [{"visible": [True, True, True], "x":[x], "showlegend":[True, True, True]},
          {"title": "<b>Все графики", "xaxis": {"title": "x", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}, 
           "yaxis": {"title": "y", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}
          }]

args_2 = [{"visible": [True, False, False], "x":[x_2], "showlegend": [True, False, False]},
          {"title": "<b>График синуса", "xaxis": {"title": "new_x", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}, 
           "yaxis": {"title": "new_y", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}
          }]

args_3 = [{"visible": [False, True, False], "x":[x_3], "showlegend": [False, True, False]},
          {"title": "<b>График тангенса", "xaxis": {"title": "new_xx", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}, 
           "yaxis": {"title": "new_yy", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}
          }]

args_4 = [{"visible": [False, False, True], "x":[x_3], "showlegend": [False, False, True]},
          {"title": "<b>График косинуса", "xaxis": {"title": "new_xxx", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}, 
           "yaxis": {"title": "new_yyy", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}
          }]

fig.layout.update(
                 updatemenus=[go.layout.Updatemenu(
                 type="buttons", direction="right",  active=0, x=0.57, y=1.2,
                 buttons = list([
                       dict(label='Все графики', method="update", args=args_1),
                       dict(label="sin(x)", method="update",args=args_2),
                       dict(label="tg(x)", method="update", args=args_3),
                       dict(label="cos(2x)", method="update", args=args_4)]))])
fig.show()


# ### 3. Построение двух графиков со scroll bar

# In[6]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_1, marker_color='red', name="$ \large sin(x)$"))
fig.add_trace(go.Scatter(x=x, y=y_2, marker_color='blue', name="$ \large tg(x)$"))
fig.update_layout(title="<b>Все графики", plot_bgcolor='white', legend=dict(orientation='h', font=dict(size=14)),
                 xaxis=dict(zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="x"),
                                   rangeslider=dict(visible=True), type="linear"),
                 yaxis=dict(zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="y")),
                 )

args_1 = [{"visible": [True, True, True], "x":[x], "showlegend":[True, True, True]},
          {"title": "<b>Все графики", "xaxis": {"title": "x", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey",
                                               "rangeslider":{"visible":True}, "type": "linear"}, 
           "yaxis": {"title": "y", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}
          }]

args_2 = [{"visible": [True, False, False], "x":[x_2], "showlegend": [True, False, False]},
          {"title": "<b>График синуса", "xaxis": {"title": "new_x", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey",
                                                 "rangeslider":{"visible":True}, "type": "linear"}, 
           "yaxis": {"title": "new_y", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}
          }]

fig.layout.update(
                 updatemenus=[go.layout.Updatemenu(
                 type="buttons", direction="right",  active=0, x=0.57, y=1.2,
                 buttons = list([
                       dict(label='Все графики', method="update", args=args_1),
                       dict(label="sin(x)", method="update",args=args_2)]))])
fig.show()


# In[9]:




fig = make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=x, y=y_1, marker_color='red', name="$ \large sin(x)$"), 1, 1)
fig.add_trace(go.Scatter(x=x, y=y_2, marker_color='blue', name="$ \large tg(x)$"), 2, 1)

fig.add_trace(go.Scatter(x=x, y=2*y_1, marker_color='darkred', name="$ \large 2*sin(x)$"), 1, 1)
fig.add_trace(go.Scatter(x=x, y=2*y_2, marker_color='darkblue', name="$ \large 2*tg(x)$"), 2, 1)


fig.update_xaxes(row=1, col=1, zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="x"))
fig.update_xaxes(row=2, col=1, zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="x"))
fig.update_yaxes(row=1, col=1, zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="y"))
fig.update_yaxes(row=2, col=1, zerolinecolor='lightgrey', gridcolor='lightgrey', title=dict(text="y"))
fig.update_layout(title="<b>Все графики", plot_bgcolor='white', legend=dict(orientation='h', font=dict(size=14)))

# args_1 = [{"visible": [[True, False], [False, True]], "x":[x], "showlegend":[True, True, False, False]},
#           {"title": "<b>Все графики", "xaxis": {"title": "x", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}, 
#            "yaxis": {"title": "y", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}
#           }]

# args_2 = [{"visible": [[False, True], [True, False]], "x":[x], "showlegend": [False, False, True, True]},
#           {"title": "<b> Умноженные на два", "xaxis": {"title": "new_x", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}, 
#            "yaxis": {"title": "new_y", "gridcolor":"lightgrey", "zerolinecolor":"lightgrey"}
#           }]

buttons = []
for i, label in enumerate(['Все графики', "sin(x)", "sin(2x)"]):
    visibility = [i==j for j in range(len(['Все графики', "sin(x)", "sin(2x)"]))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)
    
    
updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

# updatemenus = list([dict(type="buttons", direction="right",  active=0, x=0.57, y=1.2, buttons=list([
#                        dict(label='Все графики', method="update", args=args_1),
#                        dict(label="sin(x)", method="update",args=args_2)]))])

fig['layout']['updatemenus'] = updatemenus

# fig.layout.update(
#                  updatemenus=[go.layout.Updatemenu(
#                  type="buttons", direction="right",  active=0, x=0.57, y=1.2,
#                  buttons = list([
#                        dict(label='Все графики', method="update", args=args_1),
#                        dict(label="sin(x)", method="update",args=args_2)]))])
fig.show()


# In[19]:


labels = ['A', "B"]
new = []
for i, label in enumerate(labels):
    visibility = [i==j for j in range(len(labels))]
    new.append(visibility)


# In[21]:


new


# In[30]:



x = [i for i in range(100)]
df_1 = pd.DataFrame([(i, 1+i) for i in range(100)], columns=["X", "Y"])
df_2 = pd.DataFrame([(i, i*i) for i in range(100)], columns=["X", "Y"])
labels = ["Plus one", "Square"]

### Create individual figures
# START
fig = make_subplots(rows=1, cols=2)

trace1 = go.Scatter(x=x, y=y_1)
trace2 = go.Scatter(x=x, y=y_2)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)

trace1 = go.Scatter(x=x, y=2*y_1)
trace2 = go.Scatter(x=x, y=2*y_2)

fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 2)
# END

### Create buttons for drop down menu
buttons = []
for i, label in enumerate(labels):
    visibility = [i==j for j in range(len(labels))]
    button = dict(
                 label =  label,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': label}])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

fig['layout']['title'] = 'Title'
#fig['layout']['showlegend'] = False
fig['layout']['updatemenus'] = updatemenus

fig.show()


# In[ ]:




