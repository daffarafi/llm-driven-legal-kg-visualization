import json, os
from pathlib import Path

parsed_dir = Path('data/parsed')
for f in sorted(parsed_dir.glob('*.json')):
    with open(f, 'r', encoding='utf-8') as fp:
        d = json.load(fp)
    
    doc_id = d['document_id']
    components = d['components']
    
    babs = [c for c in components if c['component_type'] == 'BAB']
    pasals = [c for c in components if c['component_type'] == 'PASAL']
    bagians = [c for c in components if c['component_type'] == 'BAGIAN']
    ayats = [c for c in components if c['component_type'] == 'AYAT']
    total = d['total_components']
    
    # Count pages
    pages = set()
    for c in components:
        pr = c.get('page_range', [])
        if pr:
            for p in range(pr[0], pr[-1]+1):
                pages.add(p)
    
    print(f"{doc_id} | BAB:{len(babs)} | Bagian:{len(bagians)} | Pasal:{len(pasals)} | Ayat:{len(ayats)} | Total:{total} | Pages:{len(pages) if pages else '?'}")
