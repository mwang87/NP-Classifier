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
from flask import Flask, request
import json
import pandas as pd
import requests
import numpy as np
import urllib.parse

import sys
sys.path.insert(0, "Classifier")
import fingerprint_handler

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

from peewee import SqliteDatabase
db = SqliteDatabase("/data/database.db", pragmas=[('journal_mode', 'wal')])

ontology_dictionary = json.loads(open("Classifier/dict/DNP_final_ontology_mapping.json").read())

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
    dcc.Location(id='url', refresh=False),
    dbc.CardHeader(html.H5("NP Classifier")),
    dbc.CardBody(
        [
            html.Div(id='version', children="Version - Release_1"),
            dbc.Textarea(className="mb-3", id='smiles_string', placeholder="Smiles Structure"),
            dcc.Loading(
                id="structure",
                children=[html.Div([html.Div(id="loading-output-5")])],
                type="default",
            ),
            dcc.Loading(
                id="classification_table",
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


# This enables parsing the URL to shove a task into the qemistree id
@app.callback(Output('smiles_string', 'value'),
              [Input('url', 'pathname')])
def display_page(pathname):
    # Otherwise, lets use the url
    if len(pathname) > 1:
        return pathname[1:]
    else:
        return "CC1C(O)CC2C1C(OC1OC(COC(C)=O)C(O)C(O)C1O)OC=C2C(O)=O"

# This function will rerun at any 
@app.callback(
    [Output('classification_table', 'children'), Output('structure', 'children')],
    [Input('smiles_string', 'value')],
)
def handle_smiles(smiles_string):
    all_classifications, classified_prediction, fp1, fp2  = classify_structure(smiles_string)

    # Creating Table
    white_list_columns = ["Super_class", "Class", "Sub_class"]
    table_fig = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in white_list_columns
        ],
        data=all_classifications,
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
    )

    # Creating Structure Image
    img_obj = html.Img(id='image', src="https://gnps-structure.ucsd.edu/structureimg?smiles={}".format(urllib.parse.quote(smiles_string)))

    return [table_fig, img_obj]


def classify_structure(smiles):
    fp = fingerprint_handler.calculate_fingerprint(smiles, 2)

    fp1 = fp[0].tolist()[0]
    fp2 = fp[1].tolist()[0]

    query_dict = {}
    query_dict["input_5"] = fp1
    query_dict["input_6"] = fp2

    fp_pred_url = "http://npclassifier-tf-server:8501/v1/models/DNP_final:predict"
    payload = json.dumps({"instances": [ query_dict ]})

    headers = {"content-type": "application/json"}
    json_response = requests.post(fp_pred_url, data=payload, headers=headers)

    classified_prediction = np.asarray(json.loads(json_response.text)['predictions'])
    classification_indices = np.where(classified_prediction[0] >= 0.5)[0]

    output_classification_list = []

    for index in classification_indices:
        output_classification_list.append(ontology_dictionary[str(index)])
    
    if output_classification_list == []:
        index = np.argmax(classified_prediction[0])
        output_classification_list.append(ontology_dictionary[str(index)])
    
    return output_classification_list, classified_prediction.tolist()[0], fp1, fp2


from models import ClassifyEntity

@server.route("/classify")
def classify():
    smiles_string = request.values.get("smiles")

    if "cached" in request.values:
        try:
            db_record = ClassifyEntity.get(ClassifyEntity.smiles == smiles_string)
            return db_record.classification_json
        except:
            pass

    all_classifications, classified_prediction, fp1, fp2 = classify_structure(smiles_string)

    respond_dict = {}
    respond_dict["classifications"] = all_classifications
    respond_dict["classified_prediction"] = classified_prediction
    respond_dict["fp1"] = fp1
    respond_dict["fp2"] = fp2

    # Lets save the result here, we should also check if its changed, and if so, we overwrite
    try:
        # Save it out
        ClassifyEntity.create(
                smiles=smiles_string,
                classification_json=json.dumps(respond_dict)
            )
    except:
        pass

    return json.dumps(respond_dict)

@server.route("/model/metadata")
def metadata():
    """Serve a file from the upload directory."""
    return requests.get("http://npclassifier-tf-server:8501/v1/models/DNP_final/metadata").text

if __name__ == "__main__":
    app.run_server(debug=True, port=5000, host="0.0.0.0")
