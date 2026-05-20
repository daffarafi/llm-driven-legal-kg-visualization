import json, sys
sys.path.insert(0, '.')
from pipeline.extract.structure_parser import merge_pages_to_text

with open('data/extracted/UU_27_2022.json', 'r', encoding='utf-8') as f:
    doc = json.load(f)

text, line_to_page = merge_pages_to_text(doc['pages'])

# Check if BAB VIII, IX, XII now appear as clean lines
import re
for i, line in enumerate(text.split('\n')):
    if re.match(r'^\s*BAB\s+[IVXLCDM]+\s*$', line.strip(), re.IGNORECASE):
        print(f"line {i:>4} (p{line_to_page.get(i,'?'):>2}): [{line.strip()}]")
