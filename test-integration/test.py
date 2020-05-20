import pandas as pd
import urllib.parse
import requests

SERVER_URL = "http://dorresteintesthub.ucsd.edu:6541"
#SERVER_URL = "http://mingwangbeta.ucsd.edu:6541"

def test():
    df = pd.read_csv("test.tsv", sep="\t")

    for entry in df.to_dict(orient="records"):
        smiles = str(entry["smiles"])
        if len(smiles) > 5:
            request_url = "{}/classify?smiles={}".format(SERVER_URL, urllib.parse.quote(smiles))
            print(request_url)
            r = requests.get(request_url)
            classification = r.json()
            print(classification)
