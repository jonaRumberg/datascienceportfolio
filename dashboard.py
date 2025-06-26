import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn import tree
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Dashboard im Browser anzeigen
# http://127.0.0.1:8050/

df = pd.read_csv("data/smoker_train.csv")

# load images
def encode_image(image_file):
    with open(image_file, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

tree = encode_image("images/dashboard_tree.jpeg")
weight_height = encode_image("images/dashboard_weight_height.png")
age = encode_image("images/dashboard_age.png")
smoker = encode_image("images/dashboard_smoker.png")


# Dash-Anwendung initialisieren
app = dash.Dash(__name__)

# Layout des Dash-Dashboards definieren
app.layout = html.Div([
    html.H1("Smoker prediction", style={'textAlign': 'center'}),
    html.Details([
        html.Summary("Show/Hide Participant Data"),
        html.Div([
            html.Img(src=weight_height, style={'width': '400px', 'height': 'auto', 'marginRight': '20px'}),
            html.Img(src=smoker, style={'width': '400px', 'height': 'auto', 'marginRight': '20px'}),
            html.Img(src=age, style={'width': '400px', 'height': 'auto'})
        ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginBottom': '30px'})
    ]),
    dcc.Tabs([
        dcc.Tab(label='Blood Values', children=[
            html.Div([
            ])
        ]),
        dcc.Tab(label='Modeling', children=[
            html.Div([
                html.H3("Modeling"),
                html.Img(src=tree, style={'width': '600px', 'height': 'auto'})
            ])
        ])
    ])
])

# Dash-Anwendung ausführen
if __name__ == '__main__':
    app.run(debug=True) # Debug Modus für Ausgabe von Fehlermeldungen im Dashboard uvm
