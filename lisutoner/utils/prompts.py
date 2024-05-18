PROMPT_NER = """
Extract from the text below these entities if any: ['PRODUCT', 'MAKER', 'MPN', 'SINGLE_ITEM_QUANTITY', 'TOTAL_QUANTITY', 'COUNTRY', 'COLORS', 'PRODUCT_PROPERTY']

'PRODUCT' - Product Name
'MAKER' - Manufacturer Name
'MPN' - Manufacturer Part Number, if there are multiple MPNs, return list of all found
'SINGLE_ITEM_QUANTITY' - list of all entities that relevant to how much one item contains (e.g. ['1', '1ml'], ['100m'] etc.)
'TOTAL_QUANTITY' - list of all entities that relevant to how much one listing contains (e.g. ['1品づつ'])
'COUNTRY' - made in ... | country of origin (e.g. ['Japan', '日本'])
'COLORS' - list of colors found in the text (e.g. ['green', 'lime', 'lightgreen'])
'PRODUCT_PROPERTY' - list of characteristics of the product (e.g. ['strengthened', 'leather'], ['covered by gold'], ['boxed', 'heavy'] etc.). Each property up to 4 words

Don't add additional text, return only valid JSON string, don't add any other text besides this.

TEXT:
{item_description}
"""

PROMPT_ITEM_AMOUNT = """
TASK:
Extract from the text below the only part that related to a quantity of items if any.
Do not change the text itself.
Cut out only the part that contains the quantity of items.
If there are multiple parts, return all of them as a list.
Include the context of the quantity, if it is important to understand the quantity.
Return answer in the JSON format: {{ "text": ["extracted text_1", "extracted text_2", ...] }}

TEXT:
{item_description}
"""

PROMPT_SIZE_RELATED = """
TASK:
You are labeling the size-related information in the text.
Extract from the text below the only part that related to a size if any.
Do not change the text itself.
Cut out only the part that contains the size.
If there are multiple parts, return all of them as a list.
Include the context of the size, if it is important to understand the size.

Return answer in the JSON format. Example: 
{{ 
 "contains_size_related_info": true,
 "attributes": [
    { "attribute_name": "size", "attribute_category": "Body Height", "text": "extracted text_1" },
    { "attribute_name": "size", "attribute_category": "Shoe Size", "text": "extracted text_2" },
    ...
 ]
}}

POSSIBLE ATTRIBUTES:
- Body dimensions:
    * Width
    * Height
    * Depth
    * Gusset depth (if applicable)
- Total height
- Shoe sizes:
    * JP
    * EUR
    * US
- Heel size
- Waist circumference
- Hip circumference
- Shoulder width
- Neck circumference
- Chest circumference
- Sleeve length
- Seam length
- Dress length
- Total length (if applicable)
- Human height
- Watch dimensions:
    * Horizontal size of the watch face
    * Vertical size of the watch face
    * Watch strap width
    * Watch length
- Head circumference

TEXT:
{item_description}
"""