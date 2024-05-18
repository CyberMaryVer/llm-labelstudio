import requests
from loguru import logger
from typing import Union, List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from datetime import datetime


class StagingModel(LabelStudioMLBase):
    URL = 'http://inference.staging.lisuto.io:8080/predictions/'
    MODEL = f'staging_{datetime.now().strftime("%Y%m%d")}'
    IGNORED_ENTITIES = ['PRODUCT', 'URL']
    NER_MAPPING = {'MPN': 'mpn',
                   'MAKER': 'maker_string',
                   'COUNTRY': 'country_origin_string',
                   'COLORS': 'color_string'
                   }

    def check_content_annotations(self, data, label=None):
        """['0'] or ['0', '1']"""
        try:
            # Define label: if there are only '0' - maker, [0, 1] - mpn, [0, 1, 2] - country, more - color
            if label is None:
                label = 'LABEL'
            # Add suffix
            label = f"{label}_FROM_FILE"

            # Get provided annotations
            annotations_data = []
            for lab in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                annotations_data.append(data.get(lab, ''))
            print(f"\033[090mData: {annotations_data}\033[0m")

            # Filter out empty annotations
            annotations = [d for d in annotations_data if len(d) > 0]
            content_annotations = {label: annotations}
            logger.info(f"\033[093m{label} Annotations: {content_annotations}\033[0m")

            # Check if annotations are valid
            q = data.get('query', '')
            valid_annotations = self.check_ner_results(q, content_annotations)
            logger.info(f"\033[092mValid Annotations: {valid_annotations}\033[0m")
            return valid_annotations

        except Exception as e:
            logger.error(f"\033[091mError: {e}\033[0m")
            return []

    def check_ner_results(self, q, results):
        def _find_ent_in_text(ent, text):
            start = 0
            found_ents = []
            while start < len(text):
                start = text.find(ent, start)
                if start == -1:  # No more occurrences found
                    break
                end = start + len(ent)
                found_ents.append({"start": start, "end": end, "labels": ent_name, "text": text[start:end]})
                start += len(ent)  # Move past this occurrence
            return found_ents

        llm_ents = []
        for ent_name, ents in results.items():
            if ent_name in self.IGNORED_ENTITIES:
                continue
            if not isinstance(ents, list):
                ents = [ents]  # Ensure ents is always a list

            for e in ents:
                found_ents = _find_ent_in_text(e, q)
                if not found_ents:  # Search for the uppercase version
                    found_ents = _find_ent_in_text(e.upper(), q)
                llm_ents.extend(found_ents)

        return llm_ents

    def request_staging(self, q: str, lang: str = 'ja', site_id: int = 35, category_id: int = 401589, test: int = 1):
        data = {
            "lang": lang,
            "site_id": site_id,
            "q": q,
            "category_id": category_id,
            "test": test
        }
        response = requests.post(self.URL, json=data)
        return response

    def extract_ner_results(self, response: requests.Response):
        if response.status_code != 200:
            return None
        response = response.json()
        results = {}
        for k, v in self.NER_MAPPING.items():
            results[k] = response.get(v, [])
        return results

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        from_name, to_name, value = self.get_first_tag_occurence('Labels', 'Text')
        predictions = []
        for task in tasks:
            text = task['data']['query']
            response = self.request_staging(text)
            ner_result = self.extract_ner_results(response)
            logger.debug(f"\033[095mNER Result: {ner_result}\033[0m [{self.MODEL}]")
            list_of_ents = self.check_ner_results(text, ner_result)
            content_annotations = self.check_content_annotations(task['data'])
            logger.info(f"\033[092mExisting Annotations: {content_annotations}\033[0m")
            list_of_ents.extend(content_annotations)

            entities = []
            for ent in list_of_ents:
                entities.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'labels',
                    'value': {
                        'start': ent['start'],
                        'end': ent['end'],
                        'text': ent['text'],
                        'labels': [ent['labels']]
                    }
                })
            predictions.append({
                'result': entities,
                'model_version': self.MODEL,
            })
        return predictions

    def fit(self, event, data, **additional_params):
        logger.info(f"Received fit event: {event}")
