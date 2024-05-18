import re
import unicodedata


def extract_text_from_html(s):
    # Regular expression for removing HTML tags and scripts
    cleanr = re.compile('<.*?>|<script.*?>.*?</script>')
    cleantext = re.sub(cleanr, '', s)
    return cleantext


def adjust_wide_text(text):
    return unicodedata.normalize('NFKC', text)


def adjust_brackets(text):
    return text \
        .replace("(", " (") \
        .replace(")", ") ") \
        .replace("{", " (") \
        .replace("}", ") ")


def clean_html_tags(s):
    # Regular expression for removing HTML tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', s)
    return cleantext


def get_japanese():
    # Unicode ranges for Japanese scripts
    hiragana = (chr(i) for i in range(0x3040, 0x309F))
    katakana = (chr(i) for i in range(0x30A0, 0x30FF))
    kanji = (chr(i) for i in range(0x4E00, 0x9FAF))
    special_symbols = '、。"\\'

    # Combine all Japanese characters into a single set
    japanese_chars = set(hiragana) | set(katakana) | set(kanji) | \
                     set(special_symbols)

    return japanese_chars


def extract_non_japanese_text(s, exceptions=None):
    exceptions = ['・', ] or exceptions
    japanese = get_japanese() - set(exceptions)
    return "".join([c if c not in japanese else " " for c in s])


def process_query_text(s, del_japanese=False, html_clean_mode="all"):
    if html_clean_mode == "standard":
        s = clean_html_tags(s)
    elif html_clean_mode == "all":
        s = extract_text_from_html(s)
    s = adjust_wide_text(s)
    s = adjust_brackets(s)
    s = extract_non_japanese_text(s) if del_japanese else s
    return s
