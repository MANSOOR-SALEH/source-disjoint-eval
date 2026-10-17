import json, numpy as np

for c in ["aydid","sada"]:
    print(f"=== {c} ===")
    folds = json.load(open(f"folds/{c}_seed0.json"))
    probe = json.load(open(f"results_fix/{c}.wavlm.json"))
    sd_folds = folds["protocols"]["source_disjoint"]["folds"]
    sd_scores = probe["protocols"]["source_disjoint"]["fixed"]["per_fold"]
    # rho_min per fold: smallest class fraction retained in training after matching
    # try common field names
    for i,(f,sc) in enumerate(zip(sd_folds, sd_scores)):
        rho = f.get("rho_min") or f.get("min_class_retention") or f.get("class_retention_min")
        print(f"fold {i}: rho_min={rho}  score={sc:.3f}")