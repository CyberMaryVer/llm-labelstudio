import requests
from typing import Union, List, Dict, Optional
from uuid import uuid4

URL = 'http://inference.staging.lisuto.io:8080/predictions/'

EXAMPLE = {
    "lang": "ja",
    "site_id": 35,
    "q": " 特価】 【最短即日出荷】 light green 安全靴 アシックス ウィンジョブ CP112 1273A056 yellow イエロー asics",
    "category_id": 401589,
    "test": 1
}


def request_staging(q: str, lang: str = 'ja', site_id: int = 35, category_id: int = 401589, test: int = 1):
    data = {
        "lang": lang,
        "site_id": site_id,
        "q": q,
        "category_id": category_id,
        "test": test
    }
    response = requests.post(URL, json=data)
    return response


def extract_ner_results(response: requests.Response):
    ner_strings = {'MPN': 'mpn',
                   'MAKER': 'maker_string',
                   'COUNTRY': 'country_origin_string',
                   'COLORS': 'color_string'
                   }
    if response.status_code != 200:
        return None
    response = response.json()
    results = {}
    for k, v in ner_strings.items():
        results[k] = response.get(v, [])
    return results


if __name__ == '__main__':
    response = request_staging(**EXAMPLE)
    print(response.json())
    print(extract_ner_results(response))
