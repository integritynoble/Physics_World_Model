import re

with open(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\papers\universal_simulation\arxiv\main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Extract all cite keys
cites = set()
pat_cite = re.compile(r'\\cite\{([^}]+)\}')
for m in pat_cite.finditer(text):
    for key in m.group(1).split(','):
        cites.add(key.strip())

# Extract all bibitem keys
bibitems = set()
pat_bib = re.compile(r'\\bibitem\{([^}]+)\}')
for m in pat_bib.finditer(text):
    bibitems.add(m.group(1).strip())

print("Cited but no bibitem:", cites - bibitems)
print("Bibitem but never cited:", bibitems - cites)

# Extract all label keys
labels = set()
pat_label = re.compile(r'\\label\{([^}]+)\}')
for m in pat_label.finditer(text):
    labels.add(m.group(1).strip())

# Extract all ref keys
refs = set()
pat_ref = re.compile(r'\\ref\{([^}]+)\}')
for m in pat_ref.finditer(text):
    refs.add(m.group(1).strip())

print("\nReferenced but no label:", refs - labels)
print("Labeled but never referenced:", labels - refs)
