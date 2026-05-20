import json, os

for stage in ['triples', 'validated', 'deduped']:
    d = f'data/{stage}'
    for f in os.listdir(d):
        if f.startswith('UU_27_2022') and f.endswith('.json'):
            data = json.load(open(os.path.join(d, f), encoding='utf-8'))
            edges = data.get('edges', [])
            memuat = [e for e in edges if e.get('relation_type', e.get('type')) == 'MEMUAT']
            # Check sources
            bab_vi_memuat = [e for e in memuat if 'VI' in str(e.get('source_id', e.get('source', '')))]
            print(f"{stage}/{f}: {len(memuat)} MEMUAT total, {len(bab_vi_memuat)} from BAB_VI")
