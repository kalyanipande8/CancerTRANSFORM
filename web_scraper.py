"""Simple web-scraping helpers to fetch tabular clinical data.

This module provides conservative utilities to:
- find and download CSV links from given pages
- parse the first HTML table on a page into a pandas.DataFrame

Note: Use only on sites where scraping is permitted. The functions are
designed to be run interactively by the researcher to gather public CSVs
or table-formatted datasets. They do not crawl sites or bypass robots.txt.
"""

from typing import List, Optional
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup


def find_csv_links(url: str) -> List[str]:
    """Return absolute URLs for anchors that end with .csv on the page."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    dom = BeautifulSoup(resp.text, "lxml")
    links = []
    for a in dom.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith('.csv'):
            links.append(requests.compat.urljoin(url, href))
    return links


def download_csv(url: str, dest_dir: str = "data/raw") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    local_name = os.path.join(dest_dir, os.path.basename(url))
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(local_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_name


def parse_first_table(url: str) -> Optional[pd.DataFrame]:
    """Parse the first HTML table on a page into a DataFrame (if present)."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    dom = BeautifulSoup(resp.text, "lxml")
    table = dom.find("table")
    if table is None:
        return None
    df = pd.read_html(str(table))[0]
    return df
