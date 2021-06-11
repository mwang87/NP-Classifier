# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import urllib.parse
from flask import Flask, request
import json
import requests
import urllib.parse

from classification import classify_structure
from models import ClassifyEntity

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'NP Classifier'

server = app.server

ontology_dictionary = json.loads(open("Classifier/dict/index_v1.json").read())

NAVBAR = dbc.Navbar(
    children=[
        dbc.NavbarBrand(
            html.Img(src="https://gnps-cytoscape.ucsd.edu/static/img/GNPS_logo.png", width="120px"),
            href="https://gnps.ucsd.edu"
        ),
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("NP Classifier", href="#")),
                dbc.NavItem(dbc.NavLink("Report Feedback",
                                        href="https://docs.google.com/forms/d/e/1FAIpQLSf1-sw-P0SQGokyeaOpEmLda0UPJW93qkrI8rfp7D46fHVi6g/viewform?usp=sf_link")),
                dbc.NavItem(dbc.NavLink("Preprint Publication",
                                        href="https://chemrxiv.org/articles/preprint/NPClassifier_A_Deep_Neural_Network-Based_Structural_Classification_Tool_for_Natural_Products/12885494/1")),
                dbc.NavItem(dbc.NavLink("API",
                                        href="https://ccms-ucsd.github.io/GNPSDocumentation/api/#structure-np-classifier")),
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
            html.Div(id='version', children="Version - 1.5"),
            html.Br(),
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
            ),
            html.Hr(),
            dcc.Loading(
                id="usage_summary",
                children=[html.Div([html.Div(id="loading-output-323")])],
                type="default",
            )
        ]
    )
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(DASHBOARD)), ], style={"marginTop": 30}),
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
    classification_dict = _process_full_classification(smiles_string)

    output_list = []

    for result in classification_dict["pathway_results"]:
        output_dict = {"type": "pathway", "entry": result}
        output_list.append(output_dict)

    for result in classification_dict["superclass_results"]:
        output_dict = {"type": "superclass", "entry": result}
        output_list.append(output_dict)

    for result in classification_dict["class_results"]:
        output_dict = {"type": "class", "entry": result}
        output_list.append(output_dict)

    if classification_dict["isglycoside"]:
        output_dict = {"type": "is glycoside", "entry": "glycoside"}
        output_list.append(output_dict)

    # Creating Table
    white_list_columns = ["type", "entry"]
    table_fig = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in white_list_columns
        ],
        data=output_list,
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current=0,
        page_size=10,
    )

    # Creating Structure Image
    img_obj = html.Img(id='image', src="https://gnps-structure.ucsd.edu/structureimg?smiles={}".format(
        urllib.parse.quote(smiles_string)))

    return [table_fig, img_obj]


# This function will rerun at any
@app.callback(
    [Output('usage_summary', 'children')],
    [Input('url', 'pathname')],
)
def usage_summary(pathname):
    number_entries = ClassifyEntity.select().count()
    return ["Total Unique SMILES Classified - {}".format(number_entries)]


def _process_full_classification(smiles_string):
    try:
        db_record = ClassifyEntity.get(ClassifyEntity.smiles == smiles_string)
        return json.loads(db_record.classification_json)
    except:
        pass

    class_result, superclass_result, pathway_result, fp1, fp2, isglycoside = classify_structure(smiles_string, ontology_dictionary)

    # Next version of the API response could just output *_result directly

    respond_dict = {"class_results": list(class_result.values()),
                    "superclass_results": list(superclass_result.values()),
                    "pathway_results": list(pathway_result.values()),
                    "class_results_ids": [int(i) for i in class_result.keys()],
                    "superclass_results_ids": [int(i) for i in superclass_result.keys()],
                    "pathway_results_ids": [int(i) for i in pathway_result.keys()],
                    "isglycoside": isglycoside,
                    "fp1": fp1, "fp2": fp2}

    # Lets save the result here, we should also check if its changed, and if so, we overwrite
    try:
        # Save it out
        ClassifyEntity.create(
            smiles=smiles_string,
            classification_json=json.dumps(respond_dict)
        )
    except:
        pass

    return respond_dict


@server.route("/classify")
def classify():
    smiles_string = request.values.get("smiles")
    respond_dict = _process_full_classification(smiles_string)

    return json.dumps(respond_dict)


# This gets you the model metadata
@server.route("/model/metadata")
def metadata():
    """Serve a file from the upload directory."""
    all_metadata = {}
    pathway_metadata = json.loads(requests.get("http://npclassifier-tf-server:8501/v1/models/PATHWAY/metadata").text)
    class_metadata = json.loads(requests.get("http://npclassifier-tf-server:8501/v1/models/CLASS/metadata").text)
    superclass_metadata = json.loads(
        requests.get("http://npclassifier-tf-server:8501/v1/models/SUPERCLASS/metadata").text)

    all_metadata["pathway"] = pathway_metadata
    all_metadata["class"] = class_metadata
    all_metadata["superclass"] = superclass_metadata

    return json.dumps(all_metadata)


if __name__ == "__main__":
    app.run_server(debug=True, port=5000, host="0.0.0.0")
