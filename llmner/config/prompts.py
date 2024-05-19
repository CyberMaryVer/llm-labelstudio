PROMPT_NER = """
Extract from the text below these entities if any: ['PRODUCT', 'MAKER', 'MPN', 'COUNTRY', 'COLORS', 'PRODUCT_PROPERTY']

'PRODUCT' - Product Name
'MAKER' - Manufacturer Name
'MPN' - Manufacturer Part Number, if there are multiple MPNs, return list of all found
'COUNTRY' - made in ... | country of origin (e.g. ['Japan', '日本'])
'COLORS' - list of colors found in the text (e.g. ['green', 'lime', 'lightgreen'])
'PRODUCT_PROPERTY' - list of characteristics of the product (e.g. ['strengthened', 'leather'], ['covered by gold'], ['boxed', 'heavy'] etc.). Each property up to 4 words

Don't add additional text, return only valid JSON string, don't add any other text besides this.

TEXT:
{query}
"""

PROMPT_NER_PREFIX = """
Extract from the text below these entities if any:
"""

PROMPT_NER_SUFFIX = """
Don't add additional text, return only valid JSON string, don't add any other text besides this.
"""


class PromptNER:
    def __init__(self, entities):
        self.entities = entities
        self.prompt = self.build_prompt()

    def build_prompt(self):
        entity_list = ', '.join([f"'{ent}'" for ent in self.entities.keys()])
        prompt = f"{PROMPT_NER_PREFIX} [{entity_list}]\n"
        for ent, ent_data in self.entities.items():
            prompt += f"\n'{ent}' - {ent_data['description']}"
            if ent_data['example']:
                prompt += f"\n\t- Example: {ent_data['example']}"
        prompt += "\n\n" + PROMPT_NER_SUFFIX
        prompt += "\nTEXT:\n{query}"
        return prompt


if __name__ == '__main__':
    from llmner.config.entities import NER_ENTITIES
    prompt = PromptNER(NER_ENTITIES).prompt
    print(prompt)