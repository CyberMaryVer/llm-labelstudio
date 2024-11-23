import json
import re


def extract_and_validate_json(text):
    # Trying return the JSON string as is
    try:
        return json.loads(text)
    except Exception as e:
        print(f"\033[090m[Error] {e}")

    # Extracting the JSON string using regex
    match = re.search(r'```json\n(.+?)\n```', text, re.DOTALL)
    if not match:
        return "No JSON string found in the text."
    json_str = match.group(1)

    # Validating the JSON string
    try:
        json_data = json.loads(json_str)
        return json_data  # It's a valid JSON, return the parsed data
    except json.JSONDecodeError as e:
        return f"Invalid JSON string: {e}"
