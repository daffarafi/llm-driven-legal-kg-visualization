import json, re, os

# Check all extracted documents for BAB-related noise
for f in os.listdir('data/extracted'):
    if not f.endswith('.json'):
        continue
    with open(f'data/extracted/{f}', 'r', encoding='utf-8') as fh:
        d = json.load(fh)
    
    doc_id = d.get('document_id', f)
    bab_lines = []
    for p in d['pages']:
        text = p.get('clean_text', '') or p.get('selectable_text', '')
        for line in text.split('\n'):
            stripped = line.strip()
            # Find lines containing "BAB" that DON'T match the clean pattern
            if re.search(r'BAB', stripped, re.IGNORECASE):
                clean_match = re.match(r'^\s*BAB\s+[IVXLCDM]+\s*$', stripped, re.IGNORECASE)
                if not clean_match:
                    bab_lines.append((doc_id, p['page_number'], stripped))
    
    if bab_lines:
        print(f"\n=== {doc_id} ({len(bab_lines)} noisy BAB lines) ===")
        for doc, page, line in bab_lines:
            print(f"  p{page:>2}: [{line[:80]}]")
