import copy
import json

fin = open("./eval.json", "r")
raw_results = json.load(fin)

wd_objs = {
    0.01: [], 
    0.03: [], 
    0.05: []
    }

results = {
    1e-5: copy.deepcopy(wd_objs), 
    3e-5: copy.deepcopy(wd_objs), 
    5e-5: copy.deepcopy(wd_objs)
    }

for rr in raw_results:
    results[rr["lr"]][rr["wd"]].append(rr)

def avg_results(lr, wd):
    results_sublist = results[lr][wd]
    ps, rs, f1s = 0, 0, 0

    for instance in results_sublist:
        ps += instance["ps"]
        rs += instance["rs"]
        f1s += instance["f1s"]

    print(f"Averages with lr:{lr} and wd:{wd} :", 
        round(ps/10, 3), round(rs/10, 3), round(f1s/10, 3))

lr = [1e-5, 3e-5, 5e-5]
wd = [0.01, 0.03, 0.05]

print(" " * 36, "ps   ", "rs   ", "f1s")

for lrx in lr:
    for wdx in wd:
        avg_results(lrx, wdx)
