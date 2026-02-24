# CLAUDE.md — SafeGenome

**Read PROJECT.md for full specifications, biological edge cases, and module details.**

## What Is This Project?
SafeGenome predicts CRISPR off-target risk per cell type using AlphaGenome-predicted
chromatin instead of experimental data. First tool bridging AlphaGenome with CRISPR safety.

## Pipeline
1. Cas-OFFinder → candidate off-target sites
2. Sequence features (CFD score, mismatch encoding)
3. AlphaGenome API → predicted chromatin at each site per cell type
4. ENCODE BigWig → experimental chromatin (gold standard comparison)
5. XGBoost → three models: sequence-only (A), sequence+experimental (B), sequence+predicted (C)
6. Cell-type transfer demo across cell types without experimental data

## Key Result
PR curve comparison: Model C (ours) should approach Model B (gold standard).

---

## How To Work On This Project

### Workflow
When the user says "build Module X" or "work on [component]":
1. Read the relevant Module section in PROJECT.md thoroughly
2. Read the Biological Edge Cases section in PROJECT.md for that phase
3. Check Current Status to see what exists already
4. Implement with tests
5. Run tests, fix failures
6. Update Current Status in PROJECT.md when done

### Decision Authority
You have full authority to make decisions about:
- Code structure, function signatures, variable names
- Error handling strategy, retry logic, caching
- Test structure and assertions
- Which Python libraries to use for standard tasks
- Refactoring for clarity

**STOP and tell the user to consult Opus for:**
- Biological interpretation of unexpected results
- Whether a data processing choice introduces bias or leakage
- Design decisions that affect the three-model comparison validity
- Any result where you're not sure if it's a bug or real biology
- Paper-related decisions

Say: *"This could affect the analysis validity — take this to Opus with PROJECT.md before I proceed."*

### Development Environment
- OS: Windows 11, PowerShell
- Python environment: venv (activate with `venv\Scripts\activate`)
- GPU: NVIDIA GeForce RTX 3070 (Cas-OFFinder should use GPU mode: `G` not `C`)
- Genome: hg38.fa located at `data/raw/genome/hg38.fa`
- All file paths use forward slashes in Python code (pathlib handles this)
- Cas-OFFinder binary located at `C:\cas-offinder\cas-offinder.exe`

### Coding Standards
- Python 3.10+, type hints on all functions
- NumPy-style docstrings
- 0-based half-open coordinates internally (BED convention). Convert on input/output.
- `logging` module, never `print()` for status messages
- `pathlib.Path`, never string concatenation for paths
- All configuration in `config.py` (gitignored)
- `random_state=42` everywhere for reproducibility
- Every module gets a corresponding `tests/test_*.py`

### File Structure
```
safegenome/
├── CLAUDE.md              ← YOU ARE HERE
├── PROJECT.md             ← Full spec, edge cases, module details
├── config.py              ← API keys, paths (gitignored)
├── requirements.txt
├── data/raw/              ← Downloaded files (gitignored)
├── data/processed/        ← Parsed standardized tables
├── data/features/         ← Feature matrices
├── data/experimental_chromatin/  ← ENCODE BigWig files
├── src/
│   ├── offtarget_finder.py
│   ├── sequence_features.py   ← CFD scoring + mismatch features
│   ├── alphagenome_client.py
│   ├── encode_client.py
│   ├── feature_matrix.py
│   ├── model.py
│   ├── cell_type_transfer.py
│   └── utils.py
├── tests/
├── scripts/               ← Numbered pipeline scripts (01_ through 06_)
├── notebooks/             ← Exploration and figures
└── figures/
```

### Critical Rules (Non-Negotiable)
These come from biological edge cases. Violating any of these corrupts the analysis:

1. **NEVER mix genome builds.** Everything must be hg38. If you encounter hg19 coordinates,
   convert with pyliftover before doing anything else.

2. **REMOVE on-target sites** from training/evaluation. The on-target has 0 mismatches and
   very high GUIDE-seq reads. If included, the model learns "0 mismatches = positive" which
   is trivial and useless. Filter: remove any site with 0 mismatches to the guide.

3. **Leave-one-GUIDE-out cross-validation.** Off-targets from the same guide are correlated.
   Standard k-fold will leak. Group splits by guide_id, always.

4. **Class imbalance: use precision-recall, not ROC.** Positive rate is 1-5%.
   ROC-AUC is misleadingly optimistic. Primary metric = Average Precision (AP).
   Set `scale_pos_weight` in XGBoost.

5. **CHANGE-seq is NAKED DNA.** It has no chromatin context. Cannot be used as positive
   labels for in-cell cleavage. Only GUIDE-seq (performed in cells) provides valid positives.
   CHANGE-seq is useful for expanding the candidate off-target set.

6. **Spearman, not Pearson** for AlphaGenome vs ENCODE correlation checks. Different scales.

7. **Log all data drops.** Every time you filter a row, log why and how many. The user needs
   this for the paper's methods section.

### When You Hit External Dependencies
Some things require the user to act. When you reach these points, give clear instructions:

- **Data downloads:** Tell user exactly which URL to visit, which file to download, where to
  save it (`data/raw/filename.csv`). Be specific: "Go to [URL], click Supplementary Table S2,
  save as `data/raw/tsai2015_guide_seq.xlsx`"
- **API keys:** Tell user to add to `config.py`, never hardcode
- **Cas-OFFinder installation:** Provide exact install commands for their OS
- **Large file downloads (ENCODE BigWig):** Provide wget/curl commands

### Module Build Order
This is the dependency chain. Build in this order:
1. `utils.py` — no dependencies, needed by everything
2. `sequence_features.py` — needs only guide+off-target sequences
3. `offtarget_finder.py` — needs Cas-OFFinder installed
4. `scripts/02_parse_datasets.py` — needs raw data downloaded
5. `alphagenome_client.py` — needs API key and SDK installed
6. `encode_client.py` — needs BigWig files downloaded
7. `feature_matrix.py` — needs modules 2-6 outputs
8. `model.py` — needs feature matrix
9. `cell_type_transfer.py` — needs trained model

### Current Status
Check PROJECT.md "Current Status" table — it is the source of truth, updated after every
work session.
