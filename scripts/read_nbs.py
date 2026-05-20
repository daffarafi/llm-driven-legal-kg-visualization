import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nbs = ['01_pdf_extraction.ipynb','02_llm_extraction.ipynb','03_neo4j_ingestion.ipynb','04_evaluation.ipynb']
for nb in nbs:
    print(f'=== {nb} ===')
    with open(f'notebooks/kg_extraction/{nb}', 'r', encoding='utf-8') as f:
        d = json.load(f)
    for c in d['cells']:
        if c['cell_type'] == 'markdown':
            src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            print(src[:300])
            print('---')
    print()
