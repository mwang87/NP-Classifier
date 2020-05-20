
from tqdm import tqdm
import requests
import grequests
import urllib.parse
import datetime

SERVER_URL = "http://dorresteintesthub.ucsd.edu:6541"
#SERVER_URL = "http://mingwangbeta.ucsd.edu:6541"

def test():
    request_url = "{}/classify?smiles={}".format(SERVER_URL, "CC1C(O)CC2C1C(OC1OC(COC(C)=O)C(O)C(O)C1O)OC=C2C(O)=O")

    for i in tqdm(range(10)):
        r = requests.get(request_url)

def test_gnps():
    request_url = "https://gnps-external.ucsd.edu/gnpslibraryfornpatlasjson"
    r = requests.get(request_url)

    all_library = r.json()

    all_urls = []
    for entry in all_library:
        smiles = str(entry["COMPOUND_SMILES"])
        if len(smiles) > 5:
            request_url = "{}/classify?smiles={}".format(SERVER_URL, urllib.parse.quote(smiles))
            all_urls.append(request_url)

    # Lets get them in parallel now
    a = datetime.datetime.now()

    #all_urls = all_urls[:2000]
    rs = (grequests.get(u) for u in all_urls)
    grequests.map(rs, size=50)

    b = datetime.datetime.now()
    seconds_elapsed = (b - a).total_seconds()
    print("seconds_elapsed", seconds_elapsed)