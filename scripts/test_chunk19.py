import json, sys
sys.path.insert(0, '.')
from pipeline.extract.chunker import create_structure_aware_chunks

with open('data/parsed/UU_19_2016.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

chunks = create_structure_aware_chunks(
    d['components'], d['document_id'], d.get('title', ''),
    max_tokens=800
)

print(f"Total chunks: {len(chunks)}\n")
for i, c in enumerate(chunks):
    # Show first 120 and last 80 chars
    text = c.text
    preview = text[:120].replace('\n', '\\n')
    ending = text[-80:].replace('\n', '\\n')
    print(f"--- Chunk {i} ({c.token_count} tok) ---")
    print(f"  START: {preview}...")
    print(f"  END:   ...{ending}")
    print()
