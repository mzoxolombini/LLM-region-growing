"""Compact view of the pending prompts of one generation (operator, parents, feedback)."""
import glob, json, os, re, sys
run, g = sys.argv[1], int(sys.argv[2])
D = {}
for f in glob.glob(f"{run}/designs/*.json"):
    e = json.load(open(f)); D[e["id"]] = e
def fb(e):
    out = [f"  {e['id']} F={e['fitness']:.2f} :: {e.get('description','')[:150]}"]
    for k, r in e["per_scene"].items():
        out.append(f"     {k[:6]} " + (f"ERR {r['error'][-200:]}" if r.get("error") else
                   f"F={r['fitness']:.1f} val={r['oa_val']:.1f} grp={r['oa_grp']:.1f} f1={r['mf1_val']:.3f} "
                   f"n={r['n_regions']} sz={r['mean_region_size']:.1f} pur={r['purity']:.3f} t={r['seconds']:.0f}"))
    return "\n".join(out)
pop = json.load(open(f"{run}/population_gen{g-1}.json"))["ids"]
print("POPULATION:"); [print(fb(D[i])) for i in pop]
for f in sorted(glob.glob(f"{run}/manual/g{g:02d}_k*.prompt.md")):
    h = open(f).readline()
    print(os.path.basename(f), h.strip())
