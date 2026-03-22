# fetcher/wfs_client.py
import requests
from owslib.wfs import WebFeatureService

BC_WFS_BASE = "https://openmaps.gov.bc.ca/geo/pub/wfs"

def get_capabilities():
    wfs = WebFeatureService(BC_WFS_BASE, version="2.0.0")
    return list(wfs.contents.keys())  # all available layer names

def fetch_layer(typename: str, bbox=None, max_features=5000):
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": typename,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
    }
    if bbox:
        params["bbox"] = f"{bbox},EPSG:4326"
    if max_features:
        params["count"] = max_features

    r = requests.get(BC_WFS_BASE, params=params, timeout=60)
    r.raise_for_status()
    return r.json()