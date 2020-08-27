import pandas as pd
import urllib.parse
import requests
from tqdm import tqdm
import grequests
import time

#SERVER_URL = "http://localhost:6541"
SERVER_URL = "http://dorresteintesthub.ucsd.edu:6541"
#SERVER_URL = "http://npclassifier.ucsd.edu:6541"

def test_speed():
    df = pd.read_csv("test.tsv", sep=",")

    iterations = 100

    all_urls = []
    for i in range(iterations):
        for entry in df.to_dict(orient="records"):
            smiles = str(entry["smiles"])
            if len(smiles) > 5:
                request_url = "{}/classify?smiles={}".format(SERVER_URL, urllib.parse.quote(smiles))
                all_urls.append(request_url)

    # Lets actually do the query and measure the speed
    rs = (grequests.get(u) for u in all_urls)
    start_time = time.time()
    responses = grequests.map(rs, size=20)
    print("--- {} seconds for {} Requests---".format(time.time() - start_time, len(all_urls)))
