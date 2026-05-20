import json

with open('data/parsed/UU_27_2022.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Show all BAB components
print("=== All BAB components ===")
for c in data['components']:
    if c['component_type'] == 'BAB':
        print(f"  ID: {c['component_id']}")
        print(f"  Number: {c.get('number', '?')}")
        print(f"  Title: {c.get('title', '?')}")
        print(f"  Parent: {c.get('parent_id', 'NONE')}")
        print()

# Show Pasal 55-62 and their parents
print("=== Pasal 55-62 parent assignments ===")
for c in data['components']:
    if c['component_type'] == 'PASAL':
        num = c.get('number', '')
        if num in ['55', '56', '57', '58', '59', '60', '61', '62']:
            print(f"  Pasal {num}: parent_id = {c.get('parent_id', 'NONE')}")
