import fitz

doc = fitz.open(r'd:\TA\research\Towards Building a Legal Virtual Assistant Based on Knowledge Graphs\Towards Building a Legal Virtual Assistant Based on Knowledge Graphs.pdf')
for i in range(min(12, len(doc))):
    print(f'--- PAGE {i+1} ---')
    print(doc[i].get_text()[:3000])
