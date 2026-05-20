import json

# Load triples
with open('data/triples/UU_27_2022_PROMPT_3_triples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check for BAB VI related nodes and MEMUAT edges
print("=== BAB nodes ===")
for node in data.get('nodes', []):
    if node.get('type') == 'Bab':
        print(f"  {node['id']} | {node['label']}")

print("\n=== MEMUAT edges ===")
memuat_count = 0
for edge in data.get('edges', []):
    if edge.get('type') == 'MEMUAT':
        memuat_count += 1
        print(f"  {edge['source']} → {edge['target']}")

print(f"\nTotal MEMUAT edges: {memuat_count}")

print("\n=== Pasal nodes from BAB VI chunks (007-014) ===")
for node in data.get('nodes', []):
    prov = node.get('provenance', {})
    chunk_id = prov.get('source_chunk_id', '')
    if any(f'struct_chunk_0{i:02d}' in chunk_id for i in range(7, 15)):
        if node.get('type') in ('Pasal', 'Bab', 'Bagian'):
            print(f"  {node['id']} | {node['type']} | chunk: {chunk_id}")
