import pandas as pd
import urllib.parse
import requests


SERVER_URL = "https://npclassifier.gnps2.org/"
#SERVER_URL = "http://mingwangbeta.ucsd.edu:6541"

def test_heartbeat():
    request_url = "{}/model/metadata".format(SERVER_URL)
    r = requests.get(request_url)
    r.raise_for_status()
    

def test():
    df = pd.read_csv("test.tsv", sep=",")

    for entry in df.to_dict(orient="records"):
        smiles = str(entry["smiles"])
        if len(smiles) > 5:
            request_url = "{}/classify?smiles={}".format(SERVER_URL, urllib.parse.quote(smiles))
            r = requests.get(request_url)
            r.raise_for_status()
            classification = r.json()
