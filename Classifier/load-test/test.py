
from tqdm import tqdm
import requests
import urllib.parse

def test():
    request_url = "http://dorresteintesthub.ucsd.edu:6541/classify?smiles={}".format("CC1C(O)CC2C1C(OC1OC(COC(C)=O)C(O)C(O)C1O)OC=C2C(O)=O")

    for i in tqdm(range(10)):
        r = requests.get(request_url)

def test_gnps():
    request_url = "https://gnps-external.ucsd.edu/gnpslibraryfornpatlasjson"
    r = requests.get(request_url)

    all_library = r.json()

    for entry in tqdm(all_library):
        smiles = str(entry["COMPOUND_SMILES"])
        if len(smiles) > 5:
            request_url = "http://dorresteintesthub.ucsd.edu:6541/classify?smiles={}".format(urllib.parse.quote(smiles))
            r = requests.get(request_url)
