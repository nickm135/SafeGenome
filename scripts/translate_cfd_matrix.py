"""
translate_cfd_matrix.py - translate Doench 2016 CFD mismatch scores
into SafeGenome internal format and print new _CFD_MM_SCORES.

Conventions confirmed by user test case (rG:dA pos6 = 0.67):
  - dX in reference = TEMPLATE strand base (Watson-Crick pairing with guide RNA)
  - Position 1 = PAM-proximal  =>  idx = ref_pos - 1
  - KEY_MAP: template-strand keys -> protospacer-strand keys (our internal format)
"""

# Reference data: Doench 2016 / GuideMaker (template-strand / rU notation, PAM-proximal pos 1)
REF = {
    "rU:dT,1": 1.0, "rU:dT,2": 0.846153846, "rU:dT,3": 0.714285714,
    "rU:dT,4": 0.476190476, "rU:dT,5": 0.5, "rU:dT,6": 0.866666667,
    "rU:dT,7": 0.875, "rU:dT,8": 0.8, "rU:dT,9": 0.928571429,
    "rU:dT,10": 0.857142857, "rU:dT,11": 0.75, "rU:dT,12": 0.8,
    "rU:dT,13": 0.692307692, "rU:dT,14": 0.619047619, "rU:dT,15": 0.578947368,
    "rU:dT,16": 0.909090909, "rU:dT,17": 0.533333333, "rU:dT,18": 0.666666667,
    "rU:dT,20": 0.5625,
    "rU:dC,1": 0.956521739, "rU:dC,2": 0.84, "rU:dC,3": 0.5,
    "rU:dC,4": 0.625, "rU:dC,5": 0.64, "rU:dC,6": 0.571428571,
    "rU:dC,7": 0.588235294, "rU:dC,8": 0.733333333, "rU:dC,9": 0.619047619,
    "rU:dC,13": 0.260869565, "rU:dC,18": 0.333333333, "rU:dC,19": 0.25,
    "rU:dC,20": 0.176470588,
    "rU:dG,1": 0.857142857, "rU:dG,2": 0.857142857, "rU:dG,3": 0.428571429,
    "rU:dG,4": 0.647058824, "rU:dG,5": 1.0, "rU:dG,6": 0.909090909,
    "rU:dG,7": 0.6875, "rU:dG,8": 1.0, "rU:dG,9": 0.923076923,
    "rU:dG,10": 0.533333333, "rU:dG,11": 0.666666667, "rU:dG,12": 0.947368421,
    "rU:dG,13": 0.789473684, "rU:dG,14": 0.285714286, "rU:dG,15": 0.272727273,
    "rU:dG,16": 0.666666667, "rU:dG,17": 0.705882353, "rU:dG,18": 0.428571429,
    "rU:dG,19": 0.275862069, "rU:dG,20": 0.090909091,
    "rA:dA,1": 1.0, "rA:dA,2": 0.727272727, "rA:dA,3": 0.705882353,
    "rA:dA,4": 0.636363636, "rA:dA,5": 0.363636364, "rA:dA,6": 0.714285714,
    "rA:dA,7": 0.4375, "rA:dA,8": 0.428571429, "rA:dA,9": 0.6,
    "rA:dA,10": 0.882352941, "rA:dA,11": 0.307692308, "rA:dA,12": 0.333333333,
    "rA:dA,13": 0.3, "rA:dA,14": 0.533333333, "rA:dA,15": 0.2,
    "rA:dA,16": 0.0, "rA:dA,17": 0.133333333, "rA:dA,18": 0.5,
    "rA:dA,19": 0.538461538, "rA:dA,20": 0.6,
    "rA:dC,3": 0.611111111, "rA:dC,8": 0.733333333, "rA:dC,9": 0.666666667,
    "rA:dC,10": 0.555555556, "rA:dC,11": 0.65, "rA:dC,12": 0.722222222,
    "rA:dC,13": 0.652173913, "rA:dC,14": 0.466666667, "rA:dC,15": 0.65,
    "rA:dC,16": 0.192307692, "rA:dC,17": 0.176470588, "rA:dC,18": 0.4,
    "rA:dC,19": 0.375, "rA:dC,20": 0.764705882,
    "rA:dG,1": 0.857142857, "rA:dG,2": 0.785714286, "rA:dG,3": 0.428571429,
    "rA:dG,4": 0.352941176, "rA:dG,5": 0.5, "rA:dG,6": 0.454545455,
    "rA:dG,7": 0.4375, "rA:dG,8": 0.428571429, "rA:dG,9": 0.571428571,
    "rA:dG,10": 0.333333333, "rA:dG,11": 0.4, "rA:dG,12": 0.263157895,
    "rA:dG,13": 0.210526316, "rA:dG,14": 0.214285714, "rA:dG,15": 0.272727273,
    "rA:dG,16": 0.0, "rA:dG,17": 0.176470588, "rA:dG,18": 0.19047619,
    "rA:dG,19": 0.206896552, "rA:dG,20": 0.227272727,
    "rC:dA,1": 1.0, "rC:dA,2": 0.909090909, "rC:dA,3": 0.6875,
    "rC:dA,4": 0.8, "rC:dA,5": 0.636363636, "rC:dA,6": 0.928571429,
    "rC:dA,7": 0.8125, "rC:dA,8": 0.875, "rC:dA,9": 0.875,
    "rC:dA,10": 0.941176471, "rC:dA,11": 0.307692308, "rC:dA,12": 0.538461538,
    "rC:dA,13": 0.7, "rC:dA,14": 0.733333333, "rC:dA,15": 0.066666667,
    "rC:dA,16": 0.307692308, "rC:dA,17": 0.466666667, "rC:dA,18": 0.642857143,
    "rC:dA,19": 0.461538462, "rC:dA,20": 0.3,
    "rC:dC,1": 0.913043478, "rC:dC,2": 0.695652174, "rC:dC,3": 0.5,
    "rC:dC,4": 0.5, "rC:dC,5": 0.6, "rC:dC,6": 0.5,
    "rC:dC,7": 0.470588235, "rC:dC,8": 0.642857143, "rC:dC,9": 0.619047619,
    "rC:dC,10": 0.388888889, "rC:dC,11": 0.25, "rC:dC,12": 0.444444444,
    "rC:dC,13": 0.136363636, "rC:dC,14": 0.0, "rC:dC,15": 0.05,
    "rC:dC,16": 0.153846154, "rC:dC,17": 0.058823529, "rC:dC,18": 0.133333333,
    "rC:dC,19": 0.125, "rC:dC,20": 0.058823529,
    "rC:dT,1": 1.0, "rC:dT,2": 0.727272727, "rC:dT,3": 0.866666667,
    "rC:dT,4": 0.842105263, "rC:dT,5": 0.571428571, "rC:dT,6": 0.928571429,
    "rC:dT,7": 0.75, "rC:dT,8": 0.65, "rC:dT,9": 0.857142857,
    "rC:dT,19": 0.428571429, "rC:dT,20": 0.5,
    "rG:dA,1": 1.0, "rG:dA,2": 0.636363636, "rG:dA,3": 0.5,
    "rG:dA,4": 0.363636364, "rG:dA,5": 0.3, "rG:dA,6": 0.666666667,
    "rG:dA,7": 0.571428571, "rG:dA,8": 0.625, "rG:dA,9": 0.533333333,
    "rG:dA,10": 0.8125, "rG:dA,11": 0.384615385, "rG:dA,12": 0.384615385,
    "rG:dA,13": 0.3, "rG:dA,14": 0.266666667, "rG:dA,15": 0.142857143,
    "rG:dA,16": 0.0, "rG:dA,17": 0.25, "rG:dA,18": 0.666666667,
    "rG:dA,19": 0.666666667, "rG:dA,20": 0.7,
    "rG:dG,1": 0.714285714, "rG:dG,2": 0.692307692, "rG:dG,3": 0.384615385,
    "rG:dG,4": 0.529411765, "rG:dG,5": 0.785714286, "rG:dG,6": 0.681818182,
    "rG:dG,7": 0.6875, "rG:dG,8": 0.615384615, "rG:dG,9": 0.538461538,
    "rG:dG,10": 0.4, "rG:dG,11": 0.428571429, "rG:dG,12": 0.529411765,
    "rG:dG,13": 0.421052632, "rG:dG,14": 0.428571429, "rG:dG,15": 0.272727273,
    "rG:dG,16": 0.0, "rG:dG,17": 0.235294118, "rG:dG,18": 0.476190476,
    "rG:dG,19": 0.448275862, "rG:dG,20": 0.428571429,
    "rG:dT,1": 0.9, "rG:dT,2": 0.846153846, "rG:dT,3": 0.75,
    "rG:dT,4": 0.9, "rG:dT,5": 0.866666667, "rG:dT,6": 1.0,
    "rG:dT,7": 1.0, "rG:dT,8": 1.0, "rG:dT,9": 0.642857143,
    "rG:dT,10": 0.933333333, "rG:dT,11": 1.0, "rG:dT,12": 0.933333333,
    "rG:dT,13": 0.923076923, "rG:dT,14": 0.75, "rG:dT,15": 0.941176471,
    "rG:dT,16": 1.0, "rG:dT,17": 0.933333333, "rG:dT,18": 0.692307692,
    "rG:dT,19": 0.714285714, "rG:dT,20": 0.9375,
}

# KEY_MAP: reference (template-strand dX) -> internal (protospacer-strand dX)
# template A <-> protospacer T, template C <-> protospacer G, etc.
KEY_MAP = {
    "rA:dA": "rA:dT",
    "rA:dC": "rA:dG",
    "rA:dG": "rA:dC",
    "rC:dA": "rC:dT",
    "rC:dC": "rC:dG",
    "rC:dT": "rC:dA",
    "rG:dA": "rG:dT",
    "rG:dG": "rG:dC",
    "rG:dT": "rG:dA",
    "rU:dT": "rT:dA",
    "rU:dG": "rT:dC",
    "rU:dC": "rT:dG",
}

INTERNAL_KEYS = [
    "rA:dC", "rA:dG", "rA:dT",
    "rC:dA", "rC:dG", "rC:dT",
    "rG:dA", "rG:dC", "rG:dT",
    "rT:dA", "rT:dC", "rT:dG",
]

new_matrix = {k: [0.0] * 20 for k in INTERNAL_KEYS}
populated = set()

for entry, value in REF.items():
    ref_key, ref_pos_str = entry.rsplit(",", 1)
    ref_pos = int(ref_pos_str)
    if ref_key not in KEY_MAP:
        print(f"WARNING: unknown ref key {ref_key!r}")
        continue
    internal_key = KEY_MAP[ref_key]
    idx = ref_pos - 1  # PAM-proximal pos 1 = index 0
    new_matrix[internal_key][idx] = value
    populated.add((internal_key, idx))

print("=== MISSING POSITIONS (will remain 0.0) ===")
total_missing = 0
for k in INTERNAL_KEYS:
    missing = [i + 1 for i in range(20) if (k, i) not in populated]
    if missing:
        print(f"  {k}: positions {missing}")
        total_missing += len(missing)
print(f"  Total missing: {total_missing}/240\n")

print("=== VALIDATION TEST CASES ===")
# Test 1: user says "rG:dA at pos6, NGG -> 0.67"
# ref rG:dA,6 (template A = protospacer T) -> internal rG:dT[idx5]
v1 = new_matrix["rG:dT"][5]
ok1 = abs(v1 - 0.666666667) < 0.001
print(f"Test 1: rG:dT[5] (ref rG:dA pos6) = {v1:.9f}  expected ~0.6667  {'PASS' if ok1 else 'FAIL'}")

# Test 2: ref rG:dA pos7 -> rG:dT[6]; ref rC:dT pos10 -> rC:dA[9]
v2a = new_matrix["rG:dT"][6]
v2b = new_matrix["rC:dA"][9]
combined = v2a * v2b  # NGG PAM factor = 1.0
ok2a = abs(v2a - 0.571428571) < 0.001
ok2b = abs(v2b - 0.875) < 0.01
print(f"Test 2a: rG:dT[6] (ref rG:dA pos7) = {v2a:.9f}  expected ~0.5714  {'PASS' if ok2a else 'FAIL'}")
print(f"Test 2b: rC:dA[9] (ref rC:dT pos10) = {v2b:.9f}  expected ~0.8750  {'PASS' if ok2b else 'FAIL - DATA MISSING'}")
print(f"  Combined = {combined:.4f}  expected 0.50\n")

print("=== NEW _CFD_MM_SCORES (Python source) ===")
print("_CFD_MM_SCORES: dict[str, list[float]] = {")
for k in INTERNAL_KEYS:
    vals = new_matrix[k]
    parts = []
    for v in vals:
        if v == 0.0:
            parts.append("0.0")
        else:
            s = f"{v:.9f}".rstrip("0")
            if s.endswith("."):
                s += "0"
            parts.append(s)
    row1 = ", ".join(parts[:10])
    row2 = ", ".join(parts[10:])
    print(f'    "{k}": [{row1},')
    print(f'             {row2}],')
print("}")
