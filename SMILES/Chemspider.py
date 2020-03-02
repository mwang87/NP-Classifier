from bs4 import BeautifulSoup
from urllib.request import urlopen
import time
from tqdm import tqdm
import numpy as np
import pandas as pd

def Find_ChemSP(Name):
    delay = np.random.randint(1,4)
    url = f"https://www.chemspider.com/Search.aspx?q={Name}"
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    smiles = soup.select('span[id=ctl00_ctl00_ContentSection_ContentPlaceHolder1_RecordViewDetails_rptDetailsView_ctl00_moreDetails_WrapControl2]')
    return smiles
