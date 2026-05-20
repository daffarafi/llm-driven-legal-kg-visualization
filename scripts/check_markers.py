import json
data = json.load(open('data/chunks/UU_27_2022_chunks.json', 'r', encoding='utf-8'))
for c in data['chunks'][4:10]:
    text_preview = c['text'][:120].replace('\n', ' ')
    print(f"{c['chunk_id']} | {c['parent_component_id']} | {text_preview}...")
