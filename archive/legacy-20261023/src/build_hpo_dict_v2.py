"""
HPO词典增强：
1. 解析hp.obo
2. 只保留HP:0000118（Phenotypic abnormality）分支
3. 提取name + synonym(exact) + 括号缩写
4. 生成变体（单复数、连字符、英式美式）
5. 输出增强词典json
"""
import re, json
from collections import defaultdict

# 解析obo
terms = {}
current_id = None
current_data = {}

with open("hp.obo", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == "[Term]":
            if current_id:
                terms[current_id] = current_data
            current_id = None
            current_data = {}
        elif line.startswith("id: "):
            current_id = line[4:]
        elif line.startswith("name: "):
            current_data["name"] = line[6:]
        elif line.startswith("synonym: "):
            m = re.match(r'synonym: "(.+?)" (EXACT|BROAD|NARROW|RELATED)', line)
            if m:
                syn = m.group(1)
                scope = m.group(2)
                if "synonyms" not in current_data:
                    current_data["synonyms"] = []
                current_data["synonyms"].append((syn, scope))
        elif line.startswith("is_a: "):
            if "is_a" not in current_data:
                current_data["is_a"] = []
            current_data["is_a"].append(line[6:].split(" ! ")[0])

if current_id:
    terms[current_id] = current_data

print(f"total terms: {len(terms)}")

# 找所有HP:0000118后代
def get_descendants(root_id):
    descendants = set()
    queue = [root_id]
    while queue:
        tid = queue.pop(0)
        if tid in descendants:
            continue
        descendants.add(tid)
        for t_id, t_data in terms.items():
            if tid in t_data.get("is_a", []):
                queue.append(t_id)
    return descendants

pheno_ids = get_descendants("HP:0000118")
print(f"Phenotypic abnormality descendants: {len(pheno_ids)}")

# 提取所有surface form
dict_map = defaultdict(set)

def add_surface(surface, hpo_id):
    surface = surface.strip().lower()
    if len(surface) >= 3:
        dict_map[surface].add(hpo_id)

for hpo_id in pheno_ids:
    if hpo_id not in terms:
        continue
    data = terms[hpo_id]
    
    if "name" in data:
        add_surface(data["name"], hpo_id)
    
    for syn, scope in data.get("synonyms", []):
        if scope == "EXACT":
            add_surface(syn, hpo_id)
    
    name = data.get("name", "")
    m = re.search(r'\(([A-Z]{2,})\)$', name)
    if m:
        add_surface(m.group(1), hpo_id)

print(f"total surface forms: {len(dict_map)}")

# 生成变体
def generate_variants(surface):
    variants = set()
    variants.add(surface)
    if surface.endswith("s"):
        variants.add(surface[:-1])
    else:
        variants.add(surface + "s")
    if "-" in surface:
        variants.add(surface.replace("-", " "))
    else:
        variants.add(surface.replace(" ", "-"))
    if "isation" in surface:
        variants.add(surface.replace("isation", "ization"))
    if "ization" in surface:
        variants.add(surface.replace("ization", "isation"))
    return variants

extended_map = defaultdict(set)
for surface, hpo_ids in dict_map.items():
    for v in generate_variants(surface):
        v = v.strip().lower()
        if len(v) >= 3:
            extended_map[v].update(hpo_ids)

print(f"after variants: {len(extended_map)}")

out = {}
for surface, hpo_ids in extended_map.items():
    out[surface] = list(hpo_ids)

with open("hpo_dict_v2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"saved hpo_dict_v2.json: {len(out)} entries")
