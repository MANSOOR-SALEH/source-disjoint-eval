import json, os, subprocess, sys

ENCODERS = ["wavlm","xlsr","mhubert","mms","whisper","w2vbert","artst"]
CORPORA  = {"aydid":"manifests/aydid_clean.csv", "sada":"manifests/sada.csv"}
SEEDS    = [1,2,3,4]   # seed 0 = existing results_fix
py = sys.executable

def flags(ref):
    if not os.path.exists(ref): return ["--no-linear"]
    r = json.load(open(ref)); out=[]
    fl = r.get("fixed_layer") or (r.get("convention_layer") or {}).get("layer")
    if fl is not None: out += ["--fixed-layer", str(int(fl))]
    ly = r.get("layers")
    if ly and len(ly)>1: out += ["--layer-stride", str(int(ly[1]-ly[0]))]
    out += ["--no-linear"]
    return out

for s in SEEDS:
    rdir = f"results_fix_seed{s}"; os.makedirs(rdir, exist_ok=True)
    for c,man in CORPORA.items():
        folds = f"folds/{c}_seed{s}.json"
        for e in ENCODERS:
            feat = f"features/{c}.{e}.npz"
            out  = f"{rdir}/{c}.{e}.json"
            ref  = f"results_fix/{c}.{e}.json"
            if os.path.exists(out): print("have", out); continue
            if not os.path.exists(feat): print("skip missing", feat); continue
            cmd = [py,"src/probe.py","--features",feat,"--manifest",man,
                   "--folds",folds,"--out",out,"--epochs","30","--seed",str(s)] + flags(ref)
            print("RUN:", c, e, "seed", s); subprocess.run(cmd)
print("DONE")