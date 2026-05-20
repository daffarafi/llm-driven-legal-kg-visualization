import json

with open('data/parsed/UU_19_2016.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Total components: {d['total_components']}")
print(f"Types: {d['component_types']}")

babs = [c for c in d['components'] if c['component_type'] == 'BAB']
pasals = [c for c in d['components'] if c['component_type'] == 'PASAL']
print(f"BABs: {len(babs)}")
print(f"Pasals: {len(pasals)}")

for c in babs:
    print(f"  BAB {c['number']} | {c.get('title','')}")
for c in pasals[:10]:
    print(f"  Pasal {c['number']} | parent: {c.get('parent_id','')}")
