import json

FILE = 'data/deduped/UU_27_2022_PROMPT_3_triples.json'
data = json.load(open(FILE, encoding='utf-8'))

nodes_by_id = {n['id']: n for n in data['nodes']}

# 1. All Bab nodes
print("=== ALL Bab nodes ===")
bab_nodes = [n for n in data['nodes'] if n['type'] == 'Bab']
for n in bab_nodes:
    print(f"  ID: {n['id']}")
    print(f"  Label: {n['label']}")
    print()

# 2. All Regulasi nodes
print("=== ALL Regulasi nodes ===")
reg_nodes = [n for n in data['nodes'] if n['type'] == 'Regulasi']
for n in reg_nodes:
    print(f"  ID: {n['id']}")
    print(f"  Label: {n['label']}")
    print()

# 3. Edge key names
if data['edges']:
    print(f"=== Edge keys: {list(data['edges'][0].keys())} ===\n")

# 4. Regulasi -> Bab MEMUAT edges
print("=== Regulasi -> Bab MEMUAT edges ===")
for e in data['edges']:
    etype = e.get('relation_type', e.get('type', ''))
    src = e.get('source_id', e.get('source', ''))
    tgt = e.get('target_id', e.get('target', ''))
    if etype == 'MEMUAT' and 'Regulasi' in src:
        tgt_label = nodes_by_id.get(tgt, {}).get('label', '???')
        print(f"  {src} -> {tgt} ({tgt_label})")

# 5. BAB VI -> Pasal MEMUAT edges  
print("\n=== BAB VI -> Pasal MEMUAT edges ===")
for e in data['edges']:
    etype = e.get('relation_type', e.get('type', ''))
    src = e.get('source_id', e.get('source', ''))
    tgt = e.get('target_id', e.get('target', ''))
    if etype == 'MEMUAT' and 'VI' in src:
        src_label = nodes_by_id.get(src, {}).get('label', '???')
        tgt_label = nodes_by_id.get(tgt, {}).get('label', '???')
        print(f"  {src} -> {tgt} ({tgt_label})")

# 6. BAB VII -> Pasal MEMUAT edges
print("\n=== BAB VII -> Pasal MEMUAT edges ===")
for e in data['edges']:
    etype = e.get('relation_type', e.get('type', ''))
    src = e.get('source_id', e.get('source', ''))
    tgt = e.get('target_id', e.get('target', ''))
    if etype == 'MEMUAT' and 'VII' in src:
        src_label = nodes_by_id.get(src, {}).get('label', '???')
        tgt_label = nodes_by_id.get(tgt, {}).get('label', '???')
        print(f"  {src} -> {tgt} ({tgt_label})")

# 7. Orphan Pasal (no incoming MEMUAT)
print("\n=== Pasal tanpa MEMUAT dari Bab ===")
pasal_nodes = {n['id'] for n in data['nodes'] if n['type'] == 'Pasal'}
pasal_with_memuat = set()
for e in data['edges']:
    etype = e.get('relation_type', e.get('type', ''))
    tgt = e.get('target_id', e.get('target', ''))
    if etype == 'MEMUAT' and tgt in pasal_nodes:
        pasal_with_memuat.add(tgt)
orphans = pasal_nodes - pasal_with_memuat
for p in sorted(orphans):
    label = nodes_by_id[p].get('label', '???')
    print(f"  {p} ({label})")
print(f"\nTotal orphan Pasal: {len(orphans)}/{len(pasal_nodes)}")
