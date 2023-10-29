import json
import glob

fs = [json.load(open(x)) for x in glob.glob('output/minif2f_test/pythia/*.json')]

n = 0
ns = 0
for f in fs:
    for result in f['results']:
        # Skip Lean Dojo init failures (only occurred on LeanDojo Mathlib)
        if result['attempt_results'][0]['failure_reason'].startswith("DojoInitError"):
            continue
        n += 1
        if result['success']:
            ns += 1

print(ns/n, ns, n)
