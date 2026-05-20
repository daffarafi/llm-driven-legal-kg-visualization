"""Check which chunks contain Bab context and which Pasal they cover."""
import json

data = json.load(open('data/chunks/UU_27_2022_chunks.json', 'r', encoding='utf-8'))
chunks = data.get('chunks', [])

print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    chunk_id = chunk.get('chunk_id', f'chunk_{i}')
    text = chunk.get('text', '')
    
    # Find Bab mentions
    bab_mentions = []
    for line in text.split('\n'):
        stripped = line.strip().upper()
        if stripped.startswith('BAB '):
            bab_mentions.append(line.strip())
    
    # Find Pasal mentions
    import re
    pasal_nums = sorted(set(int(m) for m in re.findall(r'Pasal\s+(\d+)', text)))
    
    # Check structural context
    struct_context = chunk.get('structural_context', {})
    context_bab = struct_context.get('bab', '')
    
    pasal_range = f"{min(pasal_nums)}-{max(pasal_nums)}" if pasal_nums else "none"
    
    print(f"[{chunk_id}]")
    print(f"  Context Bab: {context_bab or '(none)'}")
    print(f"  Bab in text: {bab_mentions or '(none)'}")
    print(f"  Pasal range: {pasal_range} ({len(pasal_nums)} pasal)")
    print(f"  Text length: {len(text)} chars")
    print()
