import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/deduped/UU_11_2008_triples.json"
data = json.load(open(path, "r", encoding="utf-8"))
regs = [n for n in data["nodes"] if n["type"] == "Regulasi"]
print(f"Total Regulasi nodes: {len(regs)}")
for r in regs:
    print(f"  {r['id']}: {r['label']}")
print(f"\nTotal nodes: {data['total_nodes']} (before: {data.get('nodes_before','?')})")
print(f"Total edges: {data['total_edges']} (before: {data.get('edges_before','?')})")
print(f"Nodes merged: {data.get('nodes_merged','?')}")
