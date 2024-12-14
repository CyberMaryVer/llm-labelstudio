import os
import requests
import json
import spacy
from loguru import logger
from typing import Union, List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_env

from preprocess import process_query_text

HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = get_env('API_KEY')
SPACY_MODEL = os.path.join(os.getenv('MODEL_DIR', './models'), os.getenv('SPACY_MODEL', 'checkpoint'))
print(f"\033[090mSPACY_MODEL ENV VAR: {os.getenv('SPACY_MODEL')}\033[0m")
print(f"\033[093mSpacy Model: {SPACY_MODEL}\033[0m")
nlp = spacy.load(SPACY_MODEL)


class SpacyMLBackend(LabelStudioMLBase):
    IGNORED_ENTITIES = ['PRODUCT', 'URL']
    OVERLAPPING_STRATEGY = 'longest'  # 'remove', 'longest'

    def __init__(self, **kwargs):
        # Extract and remove label_config from kwargs if it exists
        label_config = kwargs.pop('label_config', None)

        # If label_config is not provided, define a default one
        if label_config is None:
            label_config = '''
                    <View>
                      <Labels name="label" toName="text">
                        <Label value="PER" background="red"/>
                        <Label value="ORG" background="darkorange"/>
                        <Label value="LOC" background="orange"/>
                        <Label value="MISC" background="green"/>
                      </Labels>
                      <Text name="text" value="$text"/>
                    </View>
                    '''

        # Initialize the base class with label_config
        super().__init__(label_config=label_config, **kwargs)
        logger.info(f"label_interface initialized: {hasattr(self, 'label_interface')}")

    def spacy_ents_to_results(self, ents):
        results = {}
        for ent in ents:
            if ent.label_ in results:
                results[ent.label_].append(ent.text)
            else:
                results[ent.label_] = [ent.text]
        return results

    def check_ner_results(self, q, results):
        """
        q: text query (e.g. "Apple iPhone 12, made in USA, available in black and white")
        results: dictionary with entity names as keys and entity values as values
        (e.g. {'MPN': 'iPhone 12', 'MAKER': 'Apple', 'COUNTRY': 'USA', 'COLORS': ['black', 'white']})
        """
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

                entities.append({"start": start,
                                 "end": end,
                                 "labels": ent_name,
                                 "text": text[start:end]})
                start += len(ent)  # Move past this occurrence

        entities = []
        for ent_name in results.keys():
            if ent_name in self.IGNORED_ENTITIES:
                continue
            ents = results[ent_name]
            if type(ents) is list:
                for e in ents:
                    _find_ent_in_text(e, q)
            elif type(ents) is str:
                _find_ent_in_text(ents, q)

        return entities

    def remove_overlapping_entities(self, entities):
        """
        entities: list of entities with start, end, labels, and text
        return: list of non-overlapping entities
        """
        # Sort entities by their start position
        entities = sorted(entities, key=lambda x: x['start'])
        non_overlapping = []

        for ent in entities:
            # If no entities in the list, add the first one
            if not non_overlapping:
                non_overlapping.append(ent)
                continue

            prev_ent = non_overlapping[-1]
            if ent['start'] < prev_ent['end']:
                # If strategy is to remove overlapping entities, skip the current entity
                if self.OVERLAPPING_STRATEGY == 'remove':
                    continue
                # If strategy is to keep the longest entity
                elif self.OVERLAPPING_STRATEGY == 'longest':
                    if (ent['end'] - ent['start']) > (prev_ent['end'] - prev_ent['start']):
                        non_overlapping[-1] = ent
            else:
                non_overlapping.append(ent)

        return non_overlapping

    # def predict(self, tasks, context, **kwargs):
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        # Initialize label_interface if label_config is provided in context
        if context and 'label_config' in context:
            self.use_label_config(context['label_config'])

        from_name, to_name, value = self.get_first_tag_occurence('Labels', 'Text')
        predictions = []
        for task in tasks:
            # Preprocess the text and extract entities using Spacy
            text = task['data'][value]
            text = process_query_text(text)
            doc = nlp(text)
            logger.info(f"\033[095mNER Result: {doc.ents}\033[0m")

            # Convert Spacy entities to list of entities
            ents = self.spacy_ents_to_results(doc.ents)
            list_of_ents = self.check_ner_results(text, ents)
            list_of_ents = self.remove_overlapping_entities(list_of_ents)

            # Convert entities to Label Studio format
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
                'model_version': SPACY_MODEL,
            })
        return predictions

    def fit(self, event, data, **additional_params):
        print(f"\033[095mReceived fit event: {event}\033[0m")

        # return {
        #     'model_path': workdir,
        #     'batch_size': batch_size,
        #     'pretrained_model': pretrained_model,
        #     'label_map': label_map
        # }

    def _get_annotated_dataset(self, project_id):
        """Just for demo purposes: retrieve annotated data from Label Studio API"""
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        if response.status_code != 200:
            raise Exception(f"Can't load task data using {download_url}, "
                            f"response status_code = {response.status_code}")
        return json.loads(response.content)

    def _train_spacy(self, data, config: Dict):
        import random
        from spacy.training import Example
        from datetime import datetime

        # Training data
        train = data['train']
        examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in train]

        # Training loop
        n_iter = config.get('n_iter', 10)
        disabled_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        optimizer = nlp.resume_training()
        optimizer.learn_rate = config.get('learn_rate', 1e-4)
        decay_factor = config.get('decay_factor', 0.9)
        decay_after = config.get('decay_after', 10)
        decay_every = config.get('decay_every', 4)
        batch_size = config.get('batch_size', 16)
        all_losses = []

        with nlp.disable_pipes(*disabled_pipes):
            for itn in range(n_iter):
                random.shuffle(examples)
                losses = {}
                for batch in spacy.util.minibatch(examples, size=batch_size):
                    nlp.update(batch, drop=0.4, losses=losses)
                print(f"Iteration: {itn} | Loss: {losses['ner']}")
                all_losses.append(losses.get('ner', 0))
                # Adjust learning rate
                if itn > decay_after and itn % decay_every == 0:
                    optimizer.learn_rate *= decay_factor

        # Save model
        output_dir = config.get('MODEL_DIR', './models')
        model_name = f"ner_v{datetime.now().strftime('%Y%m%d')}"
        nlp.to_disk(os.path.join(output_dir, model_name))

        return {
            'model_path': os.path.join(output_dir, model_name),
            'losses': all_losses
        }
