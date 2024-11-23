PROMPT_NER = """
Extract from the text in Russian below these entities if any: 
    [
    'NAME', 
    'NICKNAME', 
    'EMAIL', 
    'ADDRESS',
    'PHONE'
    'SOCIAL_MEDIA',
    'OTHER_PERSONAL_INFO'
,,

'NAME' - list of names, first names, and patronymics, e.g. ['Иван', 'Иванович', 'Иванов', 'Ивановна']
'NICKNAME' - list of nicknames, e.g. ['Вася', 'Кот Василий']
'EMAIL' - list of email addresses
'ADDRESS' - list of addresses, e.g. ['г. Москва, ул. Ленина, д. 1', 'пр. Невский, д. 2']
'PHONE' - list of phone numbers, e.g. ['+7 (999) 123-45-67', '9901234']
'SOCIAL_MEDIA' - list of social media accounts, e.g. ['@username', 'https://vk.com/username']
'OTHER_PERSONAL_INFO' - other personal information, e.g. ['СНИЛС 123-456-789 00', 'ИНН 1234567890']

Don't add additional text, return only valid JSON string, don't add any other text besides this.
e.g. {"NAME": ["Иван", "Иванович"], "PHONE": ["+7 (999) 123-45-67", "9901234"]}

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