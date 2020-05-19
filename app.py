# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import plotly.express as px
from dash.dependencies import Input, Output
import os
from zipfile import ZipFile
import urllib.parse
from flask import Flask
import json
import pandas as pd
import requests
import numpy as np

import sys
sys.path.insert(0, "Classifier")
import fingerprint_handler

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


NAVBAR = dbc.Navbar(
    children=[
        dbc.NavbarBrand(
            html.Img(src="https://gnps-cytoscape.ucsd.edu/static/img/GNPS_logo.png", width="120px"),
            href="https://gnps.ucsd.edu"
        ),
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("NP Classifier", href="#")),
            ],
        navbar=True)
    ],
    color="light",
    dark=False,
    sticky="top",
)

DASHBOARD = [
    dbc.CardHeader(html.H5("NP Classifier")),
    dbc.CardBody(
        [
            html.Div(id='version', children="Version - Release_1"),
            dbc.Textarea(className="mb-3", id='smiles_string', placeholder="Smiles Structure"),
            
            dcc.Loading(
                id="files_text_area",
                children=[html.Div([html.Div(id="loading-output-3")])],
                type="default",
            )
        ]
    )
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(DASHBOARD)),], style={"marginTop": 30}),
    ],
    className="mt-12",
)

app.layout = html.Div(children=[NAVBAR, BODY])

# This function will rerun at any 
@app.callback(
    [Output('files_text_area', 'children')],
    [Input('smiles_string', 'value')],
)
def handle_smiles(smiles_string):
    all_classifications = classify_structure(smiles_string)

    return [smiles_string + ":" + str(all_classifications)]


def classify_structure(smiles):
    fp = fingerprint_handler.calculate_fingerprint(smiles, 2)

    query_dict = {}
    query_dict["input_1"] = fp[0].tolist()[0]
    query_dict["input_2"] = fp[1].tolist()[0]

    fp_pred_url = "http://npclassifier-tf-server:8501/v1/models/superclassifier:predict"
    payload = json.dumps({"instances": [ query_dict ]})

    
    headers = {"content-type": "application/json"}
    json_response = requests.post(fp_pred_url, data=payload, headers=headers)
    print(json_response.text)

    classified_prediction = np.asarray(json.loads(json_response.text)['predictions'])
    classification_indices = np.where(classified_prediction[0] >= 0.5)[0]

    import pickle
    with open('Classifier/dict/0512_char2idx_super.pkl','rb') as char2idx_super:
        super_class = pickle.load(char2idx_super)
    
    all_classifications = []
    for XXX in classification_indices:
        classification = [name for name, age in super_class.items() if age == XXX][0]
        all_classifications.append(classification)

    return all_classifications

if __name__ == "__main__":
    app.run_server(debug=True, port=5000, host="0.0.0.0")
