from loguru import logger
from label_studio_ml.model import LabelStudioMLBase
from typing import List, Dict, Optional

from utils.llm_utils import *
from utils.json_processing import *
from utils.prompts import *


class OpenAIInteractive(LabelStudioMLBase):
    OVERLAPPING_STRATEGY = os.getenv('OVERLAPPING_STRATEGY', 'longest')  # 'remove', 'longest'
    PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE', PROMPT_NER)
    SUPPORTED_INPUTS = ('Text', 'HyperText', 'Paragraphs')
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    IGNORED_ENTITIES = [
        'PRODUCT',
        'SINGLE_ITEM_QUANTITY',
        'TOTAL_QUANTITY',
        'PRODUCT_PROPERTY'
    ]

    def __init__(self):
        super().__init__()
        self.model_name = "openai"
        self.model_agent = openai_llm
        self.prompt_template = PROMPT_NER

    def check_content_annotations(self, data, label=None):
        """['0'] or ['0', '1']"""
        try:
            # Define label: if there are [0, 1] - mpn, [0, 1, 2] - country or maker, more - color
            if label is None:
                # label = (
                #     'COLORS' if data.get('4', 'n/a') != 'n/a' else
                #     'COUNTRY' if data.get('2', 'n/a') != 'n/a' else
                #     'MAKER' if not data.get('2', 'n/a') != 'n/a' else  # implement one more condition
                #     'MPN' if data.get('1', 'n/a') != 'n/a' else
                #     'UNKNOWN'
                # )
                label = 'LABEL'
            # Add suffix
            label = f"{label}_FROM_FILE"

            # Get provided annotations
            annotations_data = []
            for lab in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                annotations_data.append(data.get(lab, ''))

            # Filter out empty annotations
            annotations = [d for d in annotations_data if len(d) > 0]
            mpn_annotations = {label: annotations}
            logger.info(f"\033[090m{label} Annotations: {mpn_annotations}\033[0m")

            # Check if annotations are valid
            q = data.get('query', '')
            valid_annotations = self.check_ner_results(q, mpn_annotations)
            logger.info(f"\033[092mValid Annotations: {valid_annotations}\033[0m")
            return valid_annotations
        except Exception as e:
            logger.error(f"\033[091mError: {e}\033[0m")
            return []

    def check_ner_results(self, q, results):
        def _find_ent_in_text(ent, text):
            start = 0
            while start < len(text):
                start = text.find(ent, start)
                if start == -1:  # No more occurrences found
                    break
                end = start + len(ent)

                # Log LLM error if entity not found
                if start == -1:
                    logger.error(f"\033[091mEntity {ent} not found in the query text.\033[0m")

                llm_ents.append({"start": start,
                                 "end": end,
                                 "labels": ent_name,
                                 "text": text[start:end]})
                start += len(ent)  # Move past this occurrence

        llm_ents = []
        for ent_name in results.keys():
            if ent_name in self.IGNORED_ENTITIES:
                continue
            ents = results[ent_name]
            if type(ents) is list:
                for e in ents:
                    _find_ent_in_text(e, q)
            elif type(ents) is str:
                _find_ent_in_text(ents, q)

        return llm_ents

    def remove_overlapping_entities(self, entities):
        """
        entities: list of entities with start, end, labels, and text
        """
        entities = sorted(entities, key=lambda x: x['start'])
        non_overlapping = []
        for i, ent in enumerate(entities):
            if not non_overlapping:
                non_overlapping.append(ent)
                continue

            prev_ent = non_overlapping[-1]
            if ent['start'] < prev_ent['end']:  # There is an overlap
                if 'FROM_FILE' in ent['labels'] or 'FROM_FILE' in prev_ent['labels']:
                    continue
                if self.OVERLAPPING_STRATEGY == 'remove':
                    continue
                elif self.OVERLAPPING_STRATEGY == 'longest':
                    if (ent['end'] - ent['start']) > (prev_ent['end'] - prev_ent['start']):
                        non_overlapping[-1] = ent
            else:
                non_overlapping.append(ent)
        return non_overlapping

    def request_llm(self, query):
        def ask_api(api, query, prompt):
            chain = prompt | api
            message = chain.invoke({"item_description": query})
            return message

        logger.info(f"\033[095mRequesting...\033[0m")

        prompt = create_langchain_prompt(user_prompt=self.PROMPT_TEMPLATE, mode="general")
        message = ask_api(api=openai_llm, query=str(query), prompt=prompt)
        return message

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        from_name, to_name, value = self.get_first_tag_occurence('Labels', 'Text')
        predictions = []
        for task in tasks:
            text = task['data']['query']
            message = self.request_llm(text)
            ner_result = extract_and_validate_json(message.content)
            logger.debug(f"\033[095mNER Result: {ner_result}\033[0m [{self.OPENAI_MODEL}]")
            list_of_ents = self.check_ner_results(text, ner_result)

            # Check for existing annotations
            content_annotations = self.check_content_annotations(task['data'])
            logger.info(f"\033[092mExisting Annotations: {content_annotations}\033[0m")
            list_of_ents += content_annotations
            list_of_ents = self.remove_overlapping_entities(list_of_ents)
            # logger.debug(f"\033[092mList of Entities: {list_of_ents}\033[0m")

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
                'model_version': self.OPENAI_MODEL,
            })
        return predictions

    def fit(self, event, data, **additional_params):
        logger.info(f"Received fit event: {event}")
