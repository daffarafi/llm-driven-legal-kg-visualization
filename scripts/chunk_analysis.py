import json

data = json.load(open('data/chunks/UU_27_2022_chunks.json', 'r', encoding='utf-8'))
print(f"{'chunk_id':<35} {'parent_bab':<25} {'tokens':>6}  pasal_in_text")
print("-" * 100)
for c in data['chunks']:
    # Count pasal mentions
    import re
    pasal_count = len(re.findall(r'\nPasal \d+', c['text']))
    parent = c.get('parent_component_id', '').split('__')[-1] if '__' in c.get('parent_component_id', '') else c.get('parent_component_id', '-')
    print(f"{c['chunk_id']:<35} {parent:<25} {c['token_count']:>6}  {pasal_count} pasal")
