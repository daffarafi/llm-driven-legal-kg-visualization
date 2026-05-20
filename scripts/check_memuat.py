import json

data = json.load(open('data/deduped/UU_27_2022_PROMPT_3_triples.json', 'r', encoding='utf-8'))

# Get MEMUAT edges from Bab to Pasal
memuat_edges = [e for e in data['edges'] if e['type'] == 'MEMUAT']

# Map: which Bab connects to which Pasal?
bab_to_pasal = {}
for e in memuat_edges:
    src = e.get('source_id') or e.get('source', '')
    tgt = e.get('target_id') or e.get('target', '')
    if src.startswith('Bab_') and tgt.startswith('Pasal_'):
        bab_to_pasal.setdefault(src, []).append(tgt)

print("=== MEMUAT edges: Bab -> Pasal ===")
for bab, pasals in sorted(bab_to_pasal.items()):
    print(f"  {bab}: {sorted(pasals)}")

# Find all Pasal nodes
all_pasal_ids = sorted([n['id'] for n in data['nodes'] if n['type'] == 'Pasal'])
connected_pasal = set()
for pasals in bab_to_pasal.values():
    connected_pasal.update(pasals)

unconnected = [p for p in all_pasal_ids if p not in connected_pasal]
print(f"\n=== Pasal tanpa MEMUAT dari Bab ({len(unconnected)}) ===")
for p in unconnected:
    print(f"  {p}")
