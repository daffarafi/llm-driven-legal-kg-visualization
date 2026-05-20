import json, re

with open('data/extracted/UU_27_2022.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

bab_re = re.compile(r'BAB', re.IGNORECASE)
for p in d['pages']:
    text = p.get('clean_text', '') or p.get('selectable_text', '')
    for line in text.split('\n'):
        if bab_re.search(line) and any(x in line.upper() for x in ['VIII', 'IX', 'XII']):
            print(f"p{p['page_number']:>2}: [{line.strip()}]")
