# -*- coding: utf-8 -*-

import json

import requests
import numpy as np
from Classifier import fingerprint_handler
from Classifier import prediction_voting


def classify_structure(smiles, ontology_dictionary):
    isglycoside = fingerprint_handler._isglycoside(smiles)

    fp = fingerprint_handler.calculate_fingerprint(smiles, 2)

    fp1 = fp[0].tolist()[0]
    fp2 = fp[1].tolist()[0]

    query_dict = {"input_3": fp1, "input_4": fp2}

    # Handling SUPERCLASS
    fp_pred_url = "http://npclassifier-tf-server:8501/v1/models/SUPERCLASS:predict"
    payload = json.dumps({"instances": [query_dict]})

    headers = {"content-type": "application/json"}
    json_response = requests.post(fp_pred_url, data=payload, headers=headers)

    pred_super = np.asarray(json.loads(json_response.text)['predictions'])[0]
    n_super = set(np.where(pred_super >= 0.3)[0])

    path_from_superclass = {j for i in n_super for j in ontology_dictionary['Super_hierarchy'][str(i)]['Pathway']}

    query_dict = {"input_3": fp1, "input_4": fp2}

    # Handling CLASS
    fp_pred_url = "http://npclassifier-tf-server:8501/v1/models/CLASS:predict"
    payload = json.dumps({"instances": [query_dict]})

    headers = {"content-type": "application/json"}
    json_response = requests.post(fp_pred_url, data=payload, headers=headers)

    pred_class = np.asarray(json.loads(json_response.text)['predictions'])[0]
    n_class = set(np.where(pred_class >= 0.1)[0])

    path_from_class = {j for i in n_class for j in ontology_dictionary['Class_hierarchy'][str(i)]['Pathway']}

    query_dict = {"input_1": fp1, "input_2": fp2}

    # Handling PATHWAY
    fp_pred_url = "http://npclassifier-tf-server:8501/v1/models/PATHWAY:predict"
    payload = json.dumps({"instances": [query_dict]})

    headers = {"content-type": "application/json"}
    json_response = requests.post(fp_pred_url, data=payload, headers=headers)

    pred_path = np.asarray(json.loads(json_response.text)['predictions'])[0]
    n_path = set(np.where(pred_path >= 0.5)[0])

    # Voting on Answer
    #
    # pred_path, pred_super, pred_class are the nparrays of the totality of the predicted
    # n_path, n_class, n_super are sets of the predicted pathways, classes and superclasses that are above noise
    # path_from_class, path_from_superclass are sets of the pathways extracted from the superclasses/classes
    n_pathway_result, n_superclass_result, n_class_result = prediction_voting.vote_classification(n_path,
                                                                                                  n_class,
                                                                                                  n_super,
                                                                                                  pred_class,
                                                                                                  pred_super,
                                                                                                  path_from_class,
                                                                                                  path_from_superclass,
                                                                                                  ontology_dictionary)

    # Get all the existing things of the ontology
    index_class = list(ontology_dictionary['Class'].keys())
    index_superclass = list(ontology_dictionary['Superclass'].keys())
    index_pathway = list(ontology_dictionary['Pathway'].keys())

    class_result = {i: index_class[i] for i in n_class_result}
    superclass_result = {i: index_superclass[i] for i in n_superclass_result}
    pathway_result = {i: index_pathway[i] for i in n_pathway_result}

    return class_result, superclass_result, pathway_result, fp1, fp2, isglycoside
