import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

import dotenv
import os
import sys

from maldi_learn.driams import load_driams_dataset

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.manifold import TSNE

import numpy as np

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

site = 'DRIAMS-A'
years = ['2015','2016', '2017', '2018']
species = 'Escherichia coli'
antibiotics = 'Ceftriaxone'

if False:
    dataset = load_driams_dataset(
        DRIAMS_ROOT,
        site,
        years,
        species,
        antibiotics=antibiotics,
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type='binned_6000',
        on_error='warn',
        id_suffix='strat'
        )

    print(dataset.y.head())
    X_ = dataset.X
    X = np.array([x.intensities for x in X_])
    y = dataset.to_numpy(antibiotics)
    years = [caseno[:4] for caseno in dataset.y['acquisition_date'].values]
    ws = list(dataset.y['workstation'].values)

    X_ = TSNE(n_components=2).fit_transform(X)

    plot_data = {
        'tSNE 1': X_[:,0],
        'tSNE 2': X_[:,1],
        'workstation': ws,
        'amr': ['0' if y_==0 else '1' for y_ in y],
        'years': years,
        'acquisition_date': dataset.y['acquisition_date'].values,
        'code': dataset.y['code'].values,
    }

    df = pd.DataFrame(plot_data)
    df.to_csv('E-CEF_plotly.csv', index=False)
else:
    df = pd.read_csv('E-CEF_plotly.csv')
    df['amr'] = df['amr'].astype(str)

df['years'] = df['years'].astype(str)
df['machine_type'] = df['code'].str[-6:] 
df.loc[df['years'] == '2015', 'machine_type'] = 'MALDI1'
df.loc[df['years'] == '2016', 'machine_type'] = 'MALDI1'
print(df.head())

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.P("E-CEF"),
    html.P("Color:"),
    dcc.Dropdown(
        id="dropdown",
        options=[
            {'label': x, 'value': x}
            for x in ['workstation','amr', 'years', 'machine_type']
        ],
        value='workstation',
        clearable=False,
    ),
    dcc.Graph(
        id='e_cef_tSNE',
        #figure=fig
    )
])

@app.callback(
    Output("e_cef_tSNE", "figure"), 
    [Input("dropdown", "value")])
def display_color(coloring):
    fig = px.scatter(df, x='tSNE 1', y='tSNE 2',
                # size="years", 
                 color=coloring, hover_name="years",
                 hover_data=['code', 'amr', 'acquisition_date'],
                 log_x=False, size_max=60,
                 width=900, height=700)
    return fig

if __name__ == '__main__':
    app.run_server()
