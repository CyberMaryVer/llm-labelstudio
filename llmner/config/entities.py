NER_ENTITIES = {
    'PRODUCT': {
        'description': 'Product Name',
        'example': 'Product Name'
    },
    'MAKER': {
        'description': 'Manufacturer Name',
        'example': None
    },
    'MPN': {
        'description': 'Manufacturer Part Number, if there are multiple MPNs, return list of all found',
        'example': None
    },
    'COUNTRY': {
        'description': 'made in ... | country of origin',
        'example': '[\'Japan\', \'日本\']'
    },
    'COLORS': {
        'description': 'list of colors found in the text',
        'example': '[\'green\', \'lime\', \'lightgreen\']'
    },
    'PRODUCT_PROPERTY': {
        'description': 'list of characteristics of the product',
        'example': '[\'strengthened\', \'leather\'], [\'covered by gold\'], [\'boxed\', \'heavy\'] etc.'
    }
}