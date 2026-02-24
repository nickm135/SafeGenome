# SafeGenome: AlphaGenome-Powered Cell-Type-Specific CRISPR Off-Target Prediction

## FOR CLAUDE: Model Routing Instructions

This project uses two Claude models in a division-of-labor workflow:

- **Sonnet (you, most likely):** Handles all code implementation, debugging, test
  writing, data parsing, plotting, and straightforward technical questions.
- **Opus:** Handles architectural decisions, biological interpretation of unexpected
  results, paper writing, strategic planning, and any situation requiring deep
  reasoning across the full project context.

**If you are Sonnet and you encounter any of the following, STOP and tell the user to
take this question to Opus instead:**
- You are not confident in your answer and it involves biological interpretation
- The user's results look unexpected and you cannot clearly explain why
- A design decision could affect the validity of downstream analysis or the paper
- You need to reason about multiple interacting edge cases simultaneously
- The user asks you to interpret model results in a publication-ready way
- You are unsure whether a data processing choice introduces bias or leakage
- Anything involving paper framing, narrative, or journal strategy

Say something like: *"This is a decision that could affect the project's validity —
I'd recommend taking this to an Opus chat with PROJECT.md attached for deeper analysis
before proceeding."*

Do NOT guess on high-stakes biological or methodological questions. It is always
better to flag uncertainty and defer than to give a confident but wrong answer that
corrupts downstream work.

---

## Project Summary

SafeGenome is a computational tool that predicts CRISPR-Cas9 off-target cleavage risk in a **cell-type-specific** manner by integrating sequence-based off-target identification with AlphaGenome's predicted chromatin accessibility. Current off-target scoring tools (CFD, MIT, Elevation) use sequence alone or require experimental epigenomic data (ENCODE DNase-seq, ATAC-seq), which exists for only ~125-200 cell lines. CRISPR therapies are moving into primary patient cells, iPSC-derived tissues, and rare populations where no chromatin maps exist. AlphaGenome (Avsec et al., Nature 2026) predicts chromatin accessibility, histone modifications, and TF binding from sequence alone for any cell type. **Nobody has connected AlphaGenome's chromatin predictions to off-target risk scoring.** This is the gap SafeGenome fills.

**Target output:** A preprint-ready manuscript demonstrating that AlphaGenome-predicted chromatin features improve CRISPR off-target prediction, approaching performance of models using experimental chromatin data.

**Target journals:** Genome Biology, Genome Research, NAR Genomics and Bioinformatics. If results are striking: Nature Methods brief communication.

---

## Architecture Overview

```
safegenome/
├── CLAUDE.md                  ← Context file for Claude Code sessions
├── PROJECT.md                 ← THIS FILE — living project document
├── README.md                  ← Public-facing repo description
├── requirements.txt           ← Python dependencies
├── config.py                  ← API keys (GITIGNORED), cell type ontology mappings
├── .gitignore
├── data/
│   ├── raw/                   ← Downloaded GUIDE-seq, CHANGE-seq supplementary tables
│   ├── processed/             ← Parsed, standardized off-target tables
│   ├── features/              ← Extracted feature matrices (sequence + chromatin)
│   └── experimental_chromatin/ ← ENCODE DNase-seq BigWig files for gold-standard comparison
├── src/
│   ├── __init__.py
│   ├── offtarget_finder.py    ← Cas-OFFinder wrapper: guide → candidate off-target sites
│   ├── sequence_features.py   ← Sequence-based feature extraction (CFD, mismatch encoding)
│   ├── alphagenome_client.py  ← AlphaGenome API wrapper: locus + cell type → chromatin vector
│   ├── encode_client.py       ← ENCODE experimental chromatin data retrieval (gold standard)
│   ├── feature_matrix.py      ← Merges sequence + chromatin features into training matrix
│   ├── model.py               ← Model training, cross-validation, evaluation
│   ├── cell_type_transfer.py  ← Cross-cell-type re-ranking demonstration
│   └── utils.py               ← Shared helpers (coordinate parsing, caching, logging)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_figures.ipynb
├── tests/
│   ├── test_offtarget_finder.py
│   ├── test_sequence_features.py
│   ├── test_alphagenome_client.py
│   └── test_feature_matrix.py
├── scripts/
│   ├── 01_download_data.sh        ← Downloads all raw data
│   ├── 02_parse_datasets.py       ← Parses raw supplementary tables into standard format
│   ├── 03_generate_candidates.py  ← Runs Cas-OFFinder for all guides
│   ├── 04_extract_features.py     ← Batch AlphaGenome + sequence feature extraction
│   ├── 05_train_models.py         ← Trains and evaluates all three models
│   └── 06_cell_type_transfer.py   ← Runs cross-cell-type demonstration
└── figures/
    ├── fig1_pipeline_schematic.svg
    ├── fig2_pr_curves.svg
    ├── fig3_cell_type_heatmap.svg
    └── fig4_feature_importance.svg
```

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Project planning | ✅ Complete | Architecture, validation strategy, timeline defined |
| AlphaGenome API key | ✅ Obtained | REGENERATE KEY — old one was exposed |
| Data download | ❌ Not started | GUIDE-seq + CHANGE-seq supplementary tables |
| Cas-OFFinder setup | ❌ Not started | Need to install and test |
| AlphaGenome SDK setup | ❌ Not started | `pip install alphagenome` from GitHub |
| Sequence feature module | ❌ Not started | |
| AlphaGenome client module | ❌ Not started | |
| Feature matrix assembly | ❌ Not started | |
| Model training | ❌ Not started | |
| Cell-type transfer demo | ❌ Not started | |
| Paper draft | ❌ Not started | |

---

## Biological Background (For Claude Context)

### Why chromatin matters for off-target prediction

CRISPR-Cas9 creates double-strand breaks at sites where the guide RNA (gRNA) base-pairs with genomic DNA. The gRNA tolerates mismatches — a guide targeting gene X can also cut "off-target" sites elsewhere in the genome that have partial sequence similarity. This is the #1 safety concern for clinical CRISPR.

**The critical insight:** Cas9 can only cut DNA it can physically access. In cells, DNA is wrapped around histones forming chromatin. Regions of "open" chromatin (euchromatin) are accessible to Cas9. Regions of "closed" chromatin (heterochromatin) are physically blocked. This means the same off-target site can be high-risk in one cell type (where chromatin is open at that locus) and zero-risk in another cell type (where it's buried in heterochromatin).

**Experimental evidence:**
- Kuscu et al. (2014): CRISPR off-target cleavage correlates with DNase I hypersensitivity
- Wu et al. (2014): dCas9 binding is enriched in open chromatin
- Kim and Kim (2018, DIG-seq): Demonstrated that native chromatin structure strongly affects cleavage
- Lazzarotto et al. (2020, CHANGE-seq): Showed cellular off-target activity is 2-4x more likely near active promoters, enhancers, and transcribed regions

### Why AlphaGenome changes the game

Previous chromatin-informed tools (CROP-IT, Elevation, DeepCRISPR) require ENCODE/Roadmap experimental data (DNase-seq, ATAC-seq, histone ChIP-seq), which exists for ~125 cell lines. Clinical CRISPR is moving into patient-specific primary cells where no such data exists.

AlphaGenome (Avsec et al., Nature 2026) predicts from DNA sequence alone:
- Chromatin accessibility (DNase-seq, ATAC-seq profiles)
- Histone modifications (H3K4me3, H3K27me3, H3K9me3, H3K27ac, etc.)
- Transcription factor binding (CTCF, POLR2A, and hundreds more)
- Gene expression, splicing, 3D contact maps

It takes as input: a genomic interval (up to 1 Mb) + cell type ontology terms (UBERON/CL/EFO).

**SafeGenome's thesis:** We can replace experimental chromatin data with AlphaGenome predictions and achieve nearly equivalent off-target prediction performance, while enabling predictions in ANY cell type.

---

## Data Sources

### Primary validation datasets

**1. CHANGE-seq (Lazzarotto et al., 2020, Nat Biotechnol)**
- 110 sgRNA targets across 13 therapeutically relevant loci
- Human primary T cells
- 201,934 off-target sites detected
- **Key data:** Supplementary Table 3 — complete list of CHANGE-seq-detected sites with read counts
- **Access:** Nature Biotechnology supplementary materials (open access tables)
- **Why it matters:** Largest dataset of experimentally verified off-targets. In vitro assay (cell-free), so captures sequence-level cleavability without chromatin filtering. Perfect for our sequence-only baseline.

**2. GUIDE-seq (Tsai et al., 2015, Nat Biotechnol)**
- 13 RGNs (RNA-guided nucleases) in two human cell lines: U2OS and HEK293
- Detects off-targets in living cells (chromatin IS a factor)
- **Key data:** Supplementary tables with identified off-target sites and read counts
- **Access:** Nat Biotech supplementary; also processed via aryeelab/guideseq GitHub
- **Why it matters:** Cell-based assay means chromatin accessibility IS already reflected in the data. If our AlphaGenome-chromatin model improves prediction of GUIDE-seq positives vs. CHANGE-seq positives, that's direct evidence chromatin features add value.

**3. ENCODE experimental chromatin data (gold standard comparison)**
- DNase-seq for K562, HEK293, U2OS: BigWig signal tracks from ENCODE portal
- H3K4me3, H3K27me3, H3K9me3 ChIP-seq for same cell lines
- **Access:** https://www.encodeproject.org/ — all freely downloadable
- **Why it matters:** We need to compare AlphaGenome PREDICTED chromatin vs. ACTUAL experimental chromatin to validate that the predictions are reasonable at off-target loci.

### Reference data
- **Human genome:** hg38 (GRCh38) — AlphaGenome uses hg38 coordinates
- **GENCODE annotation:** v46, available via AlphaGenome SDK as feather file

---

## Detailed Module Specifications

---

### Module 1: `src/offtarget_finder.py` — Off-Target Candidate Generation

**Purpose:** Given a guide RNA sequence, identify all candidate off-target sites genome-wide.

**Tool:** Cas-OFFinder (Bae et al., 2014, Bioinformatics)
- Open source: https://github.com/snugel/cas-offinder
- Searches genome for sites matching guide with up to N mismatches/bulges
- Very fast — C++ with OpenCL acceleration

**Input:**
```python
guide_sequence: str     # 20 nt protospacer, e.g., "GAGTCCGAGCAGAAGAAGAA"
pam: str                # e.g., "NGG"
genome: str             # Path to hg38 FASTA
max_mismatches: int     # 3 or 4 (start with 4)
max_bulges: int         # 1 (DNA or RNA bulge)
```

**Output:** A DataFrame with columns:
```
chromosome | position | strand | off_target_sequence | num_mismatches | mismatch_positions
```

**Expected behavior:**
- A typical guide with 4 mismatches allowed yields 50-5,000 candidate sites
- If a guide returns >10,000 candidates, it's unusually promiscuous (still fine, just more API calls)
- If a guide returns <10 candidates, double-check the sequence and PAM

**Success criteria:**
- [ ] Correctly reproduces known on-target site (0 mismatches) for test guide
- [ ] Number of candidates scales exponentially with max_mismatches (expect ~10x per additional mismatch)
- [ ] Output coordinates are in hg38

**Red flags:**
- Returns 0 candidates → wrong genome build or malformed input
- Returns millions of candidates → PAM specified incorrectly, e.g., "NNN" instead of "NGG"

**Implementation notes:**
- Cas-OFFinder requires a specific input format. Write a wrapper that generates the input file, calls the binary, and parses the output.
- Alternative: FlashFry (McKenna and Shendure, 2018) can also enumerate off-targets. Either works.
- If Cas-OFFinder installation is problematic, a pure-Python fallback using regex + Biopython is acceptable for prototyping (slower but functional for <100 guides).

---

### Module 2: `src/sequence_features.py` — Sequence-Based Feature Extraction

**Purpose:** For each candidate off-target site, compute sequence-level features that predict cleavage likelihood independent of chromatin.

**Features to extract per site:**

| Feature | Type | Description | Expected range |
|---------|------|-------------|----------------|
| `cfd_score` | float | Cutting Frequency Determination score (Doench et al., 2016) | 0.0-1.0 (higher = more likely to cut) |
| `mit_score` | float | MIT specificity score (Hsu et al., 2013) | 0.0-100.0 |
| `num_mismatches` | int | Total mismatches between guide and off-target | 0-6 |
| `mm_pos_1` through `mm_pos_20` | binary | One-hot: is position i a mismatch? | 0 or 1 |
| `mm_type_*` | categorical | Mismatch identity at each position (rG:dA, rC:dC, etc.) | encoded |
| `gc_content_guide` | float | GC fraction of guide | 0.0-1.0 |
| `gc_content_offtarget` | float | GC fraction of off-target | 0.0-1.0 |
| `pam_match` | binary | Whether PAM is canonical NGG vs NAG/NGA | 0 or 1 |
| `seed_mismatches` | int | Mismatches in positions 1-12 (PAM-proximal seed) | 0-12 |
| `nonseed_mismatches` | int | Mismatches in positions 13-20 | 0-8 |
| `consecutive_mm` | int | Max consecutive mismatches | 0-20 |

**CFD Score implementation:**
The CFD score is a lookup-table-based score. Precomputed matrices are available in the Doench et al. 2016 supplementary or in multiple GitHub repos (e.g., `crispr-offtarget-scoring`). It's a product of individual position-mismatch penalties.

```python
def cfd_score(guide: str, off_target: str, pam: str) -> float:
    """
    Returns value between 0 and 1.
    1.0 = identical to on-target (maximum risk)
    0.0 = no predicted activity
    """
```

**Success criteria:**
- [ ] CFD score of on-target site (0 mismatches) = 1.0
- [ ] CFD scores decrease monotonically with more mismatches (on average)
- [ ] Seed-region mismatches produce lower CFD scores than non-seed mismatches
- [ ] Feature matrix has no NaN values for any candidate site

**Red flags:**
- CFD scores all near 0 → scoring matrix loaded incorrectly
- CFD scores all near 1 → mismatches not being detected

---

### Module 3: `src/alphagenome_client.py` — AlphaGenome Chromatin Prediction

**Purpose:** For each candidate off-target site, query AlphaGenome to predict the chromatin state at that locus in a specified cell type. This is the core novel contribution of the project.

**AlphaGenome SDK usage:**

```python
from alphagenome.data import genome
from alphagenome.models import dna_client

# Initialize once
model = dna_client.create(API_KEY)

# For a given off-target locus, create an interval
# AlphaGenome accepts: 16kb, 100kb, 500kb, or 1Mb sequences
# Use 100kb centered on the off-target site (good balance of context vs. speed)
interval = genome.Interval(
    chromosome='chr7',
    start=off_target_position - 50000,
    end=off_target_position + 50000
)

# Request chromatin outputs for a specific cell type
output = model.predict_interval(
    interval=interval,
    requested_outputs=[
        dna_client.OutputType.DNASE,           # Chromatin accessibility
        dna_client.OutputType.ATAC,            # Chromatin accessibility (alternative assay)
        dna_client.OutputType.CHIP_HISTONE,    # Histone modifications
        dna_client.OutputType.CHIP_TF,         # TF binding (CTCF important)
    ],
    ontology_terms=['EFO:0002067'],  # e.g., K562
)
```

**Key ontology terms for our cell types of interest:**

| Cell type | Ontology term | Available in GUIDE-seq? | Available in ENCODE? |
|-----------|---------------|------------------------|---------------------|
| K562 (CML cell line) | `EFO:0002067` | Yes (limited) | Yes (extensive) |
| HEK293 | `EFO:0001182` | Yes | Yes |
| HepG2 | `EFO:0001187` | No | Yes |
| U2OS | Check metadata | Yes | Partial |
| CD4+ T cell | `CL:0000624` | No | Yes |
| iPSC | Check metadata | No | Partial |
| Neural progenitor | Check metadata | No | Partial |
| Hepatocyte | `CL:0000182` | No | No (this is the point!) |

**CRITICAL FIRST STEP — Do this in Week 1 before writing extraction code:**

Run this metadata query to discover which ontology terms have tracks for each output type:

```python
metadata = model.output_metadata(dna_client.Organism.HOMO_SAPIENS)

# Check what DNase tracks exist
dnase_meta = metadata.dnase
print(dnase_meta[['name', 'ontology_curie', 'biosample_name']].head(50))

# Check what histone ChIP tracks exist
histone_meta = metadata.chip_histone
print(histone_meta[['name', 'ontology_curie', 'biosample_name']].head(50))
```

This tells us exactly which cell types and tracks are available. If K562 or HEK293 are missing tracks, we need to know immediately.

**Feature extraction from AlphaGenome output:**

For each off-target site, extract a window of +/- 500 bp around the site from the predicted tracks:

```python
# output.dnase is a TrackData object
# .values is a numpy array of shape (seq_length, num_tracks)
# .metadata is a DataFrame describing each track

# Extract signal at the off-target site +/- 500 bp
site_start = off_target_position - interval.start - 500
site_end = off_target_position - interval.start + 500

dnase_signal = output.dnase.values[site_start:site_end, :]  # shape: (1000, num_tracks)
```

**Chromatin features to extract per off-target site:**

| Feature | Source | Extraction | Expected range |
|---------|--------|------------|----------------|
| `ag_dnase_mean` | DNASE output | Mean signal over +/-500bp window | 0 to ~10 (log-scale typical) |
| `ag_dnase_max` | DNASE output | Max signal in window | 0 to ~50 |
| `ag_dnase_at_site` | DNASE output | Signal at exact cut site position | 0 to ~50 |
| `ag_atac_mean` | ATAC output | Mean ATAC signal over window | 0 to ~10 |
| `ag_h3k4me3_mean` | CHIP_HISTONE | Mean H3K4me3 (active promoter) signal | 0 to ~20 |
| `ag_h3k27me3_mean` | CHIP_HISTONE | Mean H3K27me3 (Polycomb repression) signal | 0 to ~20 |
| `ag_h3k9me3_mean` | CHIP_HISTONE | Mean H3K9me3 (constitutive heterochromatin) signal | 0 to ~10 |
| `ag_h3k27ac_mean` | CHIP_HISTONE | Mean H3K27ac (active enhancer) signal | 0 to ~20 |
| `ag_ctcf_mean` | CHIP_TF | Mean CTCF binding signal | 0 to ~10 |
| `ag_accessibility_binary` | DNASE | Is mean > threshold? (crude open/closed call) | 0 or 1 |

**IMPORTANT:** The histone tracks in CHIP_HISTONE are multiplexed — multiple histone marks in one output. You must filter by track name to separate H3K4me3 from H3K27me3 etc. Use the metadata DataFrame:

```python
histone_data = output.chip_histone
# Filter to only H3K4me3 tracks for our cell type
h3k4me3_mask = histone_data.metadata['name'].str.contains('H3K4me3')
h3k4me3_signal = histone_data.values[:, h3k4me3_mask]
```

**Rate limiting and caching:**

AlphaGenome API is rate-limited. For ~2,000-5,000 candidate off-target sites:
- Each API call covers a 100kb interval
- Multiple sites within the same 100kb window can share one call
- Implement caching: save raw API responses to `data/features/cache/` as pickle files
- Implement exponential backoff: `time.sleep(2 ** attempt)` on rate-limit errors
- Batch by chromosome: process all sites on chr1 together, then chr2, etc.

```python
# Pseudocode for batch extraction
for chrom in sorted(sites_by_chrom.keys()):
    chrom_sites = sites_by_chrom[chrom]
    # Sort by position, merge overlapping windows
    windows = merge_overlapping_windows(chrom_sites, window_size=100000)
    for window in windows:
        cache_path = f"data/features/cache/{chrom}_{window.start}_{window.end}_{cell_type}.pkl"
        if os.path.exists(cache_path):
            output = pickle.load(open(cache_path, 'rb'))
        else:
            output = model.predict_interval(...)
            pickle.dump(output, open(cache_path, 'wb'))
            time.sleep(1)  # Rate limiting courtesy pause
        # Extract features for all sites in this window
        for site in sites_in_window:
            features = extract_chromatin_features(output, site)
```

**Success criteria:**
- [ ] API calls return non-empty TrackData objects
- [ ] DNase signal is NOT flat zero everywhere (would indicate wrong ontology term)
- [ ] DNase signal shows variation across sites (some open, some closed)
- [ ] Correlation between AlphaGenome-predicted DNase and ENCODE experimental DNase is r > 0.5 for K562 (sanity check)
- [ ] Feature extraction completes for all candidate sites without errors

**Red flags:**
- All DNase signals are zero → wrong ontology term or no tracks for that cell type
- API returns errors for most calls → rate limiting hit too hard, increase sleep time
- Features have very low variance → extraction window too small or wrong track selected

**Known limitations:**
- AlphaGenome's 0-based indexing: `Interval('chr1', 0, 1)` is the first base pair. Be careful with coordinate conversions.
- AlphaGenome uses hg38. All off-target coordinates must be in hg38. GUIDE-seq original data was hg19 — use LiftOver if needed.
- Supported sequence lengths: 16kb, 100kb, 500kb, 1Mb. If your interval doesn't match exactly, the SDK rounds up to the next valid size.

---

### Module 4: `src/encode_client.py` — Experimental Chromatin (Gold Standard)

**Purpose:** For the same off-target sites where we have AlphaGenome predictions, retrieve the actual experimental chromatin signal from ENCODE. This lets us build the "gold standard" model (Model B) and validate that AlphaGenome predictions are reasonable.

**Data source:** ENCODE portal BigWig files for matching cell types.

For K562:
- DNase-seq: Search ENCODE portal for K562 DNase-seq hg38 signal p-value BigWig
- H3K4me3: Search ENCODE for K562 H3K4me3 ChIP-seq signal
- H3K27me3: Search ENCODE for K562 H3K27me3 ChIP-seq signal

**Implementation:**
Use `pyBigWig` to query BigWig files at specific loci:

```python
import pyBigWig

bw = pyBigWig.open("data/experimental_chromatin/K562_DNase_hg38.bigWig")
signal = bw.values("chr7", start, end)  # Returns numpy array
mean_signal = np.nanmean(signal)
```

**Features to extract:** Same as AlphaGenome features but from experimental data:
- `exp_dnase_mean`, `exp_dnase_max`, `exp_dnase_at_site`
- `exp_h3k4me3_mean`, `exp_h3k27me3_mean`, `exp_h3k9me3_mean`

**Success criteria:**
- [ ] BigWig files load without errors
- [ ] Signal values are non-zero in known open chromatin regions (e.g., promoters of housekeeping genes)
- [ ] Signal values are near-zero in known heterochromatin regions

---

### Module 5: `src/feature_matrix.py` — Feature Matrix Assembly

**Purpose:** Combine sequence features, AlphaGenome chromatin features, experimental chromatin features, and ground truth labels into a single training matrix.

**Input:**
1. Off-target candidates from Cas-OFFinder (Module 1)
2. Sequence features for each candidate (Module 2)
3. AlphaGenome chromatin features for each candidate (Module 3)
4. Experimental chromatin features for each candidate (Module 4)
5. Ground truth labels: is this site a TRUE off-target? (from GUIDE-seq/CHANGE-seq)

**Label construction — this is critical:**

For CHANGE-seq data (in vitro, no chromatin):
- Positive label (y=1): Site detected by CHANGE-seq with read count > threshold
- Negative label (y=0): Site found by Cas-OFFinder (sequence similar) but NOT detected by CHANGE-seq

For GUIDE-seq data (in vivo, chromatin matters):
- Positive label (y=1): Site detected by GUIDE-seq
- Negative label (y=0): Site found by Cas-OFFinder but NOT detected by GUIDE-seq

**IMPORTANT DESIGN DECISION:** The primary evaluation should use GUIDE-seq labels (cell-based assay where chromatin matters). CHANGE-seq (cell-free) is useful for the sequence-only baseline but won't show chromatin effects.

**Output:** A DataFrame saved as `data/features/feature_matrix_{dataset}_{cell_type}.parquet`:

```
guide_id | chrom | position | strand | label | cfd_score | mit_score | num_mm | ... | ag_dnase_mean | ag_h3k4me3_mean | ... | exp_dnase_mean | ...
```

**Expected dimensions:**
- Rows: ~5,000-50,000 (all candidates for all guides in dataset)
- Columns: ~40-60 features + metadata + label
- Class balance: expect ~1-5% positives (true off-targets are rare relative to sequence-similar sites)

**Success criteria:**
- [ ] No NaN values in feature columns (handle missing data before this step)
- [ ] Label distribution matches expectation (1-5% positive)
- [ ] Features have reasonable distributions (no features with zero variance)
- [ ] At least 100 positive examples for meaningful model training

**Red flags:**
- 0 positive examples → label matching failed, likely coordinate system mismatch between Cas-OFFinder output and GUIDE-seq data
- >50% positive → label matching too permissive, matching window is too large
- Many NaN features → AlphaGenome calls failed for some sites

---

### Module 6: `src/model.py` — Model Training and Evaluation

**Purpose:** Train three models and compare their performance to test whether AlphaGenome-predicted chromatin improves off-target prediction.

**The three models:**

| Model | Features | What it tests |
|-------|----------|---------------|
| **Model A** (Sequence-only baseline) | CFD, MIT, mismatch positions, GC content, etc. | Current state-of-the-art for tools without chromatin |
| **Model B** (Sequence + experimental chromatin) | Model A features + ENCODE DNase/histone signals | Gold standard — what's possible with real data |
| **Model C** (Sequence + AlphaGenome chromatin) | Model A features + AlphaGenome predictions | **Our method** — what's possible without experiments |

**Algorithm:** Start with XGBoost (gradient-boosted trees). Reasons:
- Handles mixed feature types well
- Built-in feature importance
- Fast to train
- Works well with imbalanced data (use `scale_pos_weight`)
- No need for feature scaling
- Interpretable (SHAP values)

```python
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold

# CRITICAL: Use GroupKFold split by GUIDE
# Sites from the same guide must be in the same fold
# Otherwise: data leakage (model memorizes guide-specific patterns)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
groups = feature_matrix['guide_id']

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(y_neg) / len(y_pos),  # Handle class imbalance
    random_state=42,
    eval_metric='aucpr',  # Area under PR curve (better for imbalanced data)
)
```

**Evaluation metrics:**

| Metric | Why | Expected range |
|--------|-----|----------------|
| **AUPRC** (Area Under Precision-Recall Curve) | Primary metric. Better than AUROC for imbalanced data. | 0.1-0.5 (this is a hard task) |
| **AUROC** (Area Under ROC) | Standard but can be misleadingly high with imbalanced data | 0.7-0.95 |
| **Precision@k** | Of top k predicted sites, what fraction are true off-targets? | Variable |
| **Recall@k** | Of all true off-targets, what fraction are in top k predictions? | Variable |
| **Enrichment@10%** | How many true positives fall in top 10% of predictions? | >2x random expected |

**The key result we need:**

```
AUPRC:  Model A (seq-only) < Model C (seq + AlphaGenome) ≈ Model B (seq + experimental)
```

Specifically:
- Model C should be **statistically significantly** better than Model A (paired t-test across CV folds)
- Model C should approach Model B performance (ideally within ~5-10% relative AUPRC)
- If Model C = Model A, the project thesis fails (but see Contingency Plans below)

**Ablation analysis:** After main comparison, test which AlphaGenome features matter most:
- Sequence + DNase only
- Sequence + histone marks only
- Sequence + CTCF only
- Full model

**SHAP analysis:** Use SHAP values to show which features drive predictions:
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Success criteria:**
- [ ] Model A achieves AUROC > 0.7 (if it doesn't, sequence features are broken)
- [ ] Model C AUPRC > Model A AUPRC (our core claim)
- [ ] Model C AUPRC within 80-100% of Model B AUPRC (AlphaGenome approximates experimental data)
- [ ] Results are consistent across CV folds (low variance)
- [ ] SHAP analysis shows chromatin features (DNase, histone marks) among top 10 most important

**Red flags:**
- Model A AUROC < 0.6 → features are wrong or labels are wrong
- Model C = Model A → AlphaGenome features add nothing, thesis fails
- Model C >> Model B → impossible if experimental data is the ground truth, indicates a bug
- Huge variance across CV folds → not enough data or data leakage

---

### Module 7: `src/cell_type_transfer.py` — Cross-Cell-Type Demonstration

**Purpose:** Show the "killer feature" of SafeGenome — re-ranking off-targets across cell types — which is impossible with experimental-data-dependent tools.

**Approach:**
1. Pick a guide that has GUIDE-seq data in one cell type (e.g., K562)
2. Run SafeGenome for that guide across 5-10 different cell types
3. Show that the ranked off-target list changes between cell types
4. Validate: for cell types where GUIDE-seq data exists, show ranking matches experimental data

**Cell types to compare:**

| Cell type | Ontology | Has GUIDE-seq data? | Has ENCODE chromatin? | Novel prediction? |
|-----------|----------|--------------------|-----------------------|-------------------|
| K562 | EFO:0002067 | Yes | Yes | No (validation) |
| HEK293 | EFO:0001182 | Yes | Yes | No (validation) |
| CD4+ T cell | CL:0000624 | No | Yes | Partially novel |
| Hepatocyte | CL:0000182 | No | No | Fully novel |
| Neural progenitor | Check | No | Partial | Fully novel |
| iPSC-cardiomyocyte | Check | No | No | Fully novel |

**Output visualization:** A heatmap where:
- Rows = off-target sites (sorted by sequence-only score)
- Columns = cell types
- Color = SafeGenome risk score
- Show that the same site has high risk (red) in one cell type and low risk (blue) in another

**Success criteria:**
- [ ] Rankings visibly differ between cell types (not all identical)
- [ ] Known heterochromatin regions (e.g., near centromeres) show low risk across most cell types
- [ ] Known housekeeping gene regions show consistently high risk
- [ ] Cell-type-specific genes (e.g., hemoglobin locus) show high risk only in erythroid lineage

---

## Validation Strategy Summary

The paper must convincingly demonstrate three things:

### Validation 1: AlphaGenome predictions correlate with experimental chromatin
**Test:** At K562 off-target sites, compute Pearson/Spearman correlation between AlphaGenome-predicted DNase signal and ENCODE experimental DNase signal.
**Expected:** r > 0.5 (preferably r > 0.7).
**If fails:** AlphaGenome predictions unreliable at these loci. Project at risk.

### Validation 2: Chromatin features improve off-target prediction
**Test:** Model C (seq + AlphaGenome) vs Model A (seq only) AUPRC comparison.
**Expected:** Significant improvement (p < 0.05 by paired test across CV folds).
**If fails:** Negative result paper, still publishable but less exciting.

### Validation 3: Cell-type-specific re-ranking matches biology
**Test:** Off-target sites near cell-type-specific genes are ranked higher in matching cell types.
**Expected:** Qualitative biological plausibility + quantitative difference in rankings.
**If fails:** Visualization still shows differences (inherent to chromatin variation), but biological interpretation is weaker.

---

## Timeline

| Week | Tasks | Deliverable | Go/No-Go Check |
|------|-------|-------------|-----------------|
| **1** | Register AlphaGenome API, install SDK. Download CHANGE-seq + GUIDE-seq supplementary data. Explore AlphaGenome metadata to confirm cell type coverage. Install Cas-OFFinder. | Working API calls, parsed data tables, confirmed ontology terms | Can you make an AlphaGenome API call and get non-empty output? |
| **2** | Build Modules 1 + 2 (offtarget_finder + sequence_features). Test on 2-3 known guides. | Off-target candidates + sequence feature vectors for test guides | Do CFD scores look right? Do candidate counts match published numbers? |
| **3** | Build Module 3 (alphagenome_client). Extract chromatin features for test set of ~100 sites in K562. Validate against ENCODE. | Chromatin features extracted, correlation with ENCODE computed | Is AlphaGenome-ENCODE correlation > 0.5? |
| **4** | Build Module 4 (encode_client). Scale Module 3 to full dataset. Begin feature matrix assembly. | Feature extraction running at scale, caching working | All API calls completing without errors? |
| **5** | Build Modules 5 + 6 (feature_matrix + model). Train three models. Cross-validation. | Model comparison results, PR curves | Does Model C beat Model A? |
| **6** | Ablation analysis. SHAP feature importance. Statistical testing. | Complete model analysis | Results robust across folds? |
| **7** | Build Module 7 (cell_type_transfer). Generate cross-cell-type heatmaps. | Cell-type-specific rankings and heatmap figure | Do rankings differ meaningfully? |
| **8** | Write paper. Generate all figures. | Complete manuscript draft | |
| **9** | Polish, revisions, submit to bioRxiv. | Preprint submitted | |

---

## Contingency Plans

### If AlphaGenome predictions don't correlate with experimental chromatin (Week 3)
- Check: Are you using the correct ontology terms? Run the metadata query.
- Check: Is the interval centered correctly? 0-based indexing is tricky.
- Check: Are you extracting signal from the correct tracks? CHIP_HISTONE contains multiple marks.
- If genuinely poor correlation: Switch from raw signal to binary open/closed classification (more robust).

### If AlphaGenome features don't improve prediction (Week 5)
- First: check for bugs (most likely explanation).
- If genuinely no improvement: this is a meaningful negative result. Write paper as "Investigating the utility of predicted chromatin for CRISPR off-target scoring" — still publishable in NAR Genomics or Bioinformatics.
- Alternative pivot: use AlphaGenome for on-target REPAIR outcome prediction instead (ChromaticCRISPR — the companion project).

### If API rate limiting is prohibitive (Week 3-4)
- Reduce to 100kb intervals (instead of 500kb)
- Reduce candidate sites: use stricter mismatch threshold (3 instead of 4)
- Focus on subset of guides with richest experimental validation
- Use 16kb intervals (fastest, still captures local chromatin)

### If GUIDE-seq data coordinates don't match
- Likely hg19 vs hg38 issue.
- Use UCSC LiftOver: https://genome.ucsc.edu/cgi-bin/hgLiftOver
- Python: `pyliftover` package

---

## Key Design Decisions and Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| ML algorithm | XGBoost, not deep learning | Small dataset (~5k-50k rows), tabular features. DL would overfit. XGBoost is state-of-the-art for this regime. |
| Cross-validation | StratifiedGroupKFold by guide | Prevents data leakage. Sites from same guide share properties and must stay together. |
| Primary metric | AUPRC, not AUROC | Class imbalance (~1-5% positives). AUROC inflated by true negatives. |
| AlphaGenome interval size | 100kb | Smallest size that captures relevant regulatory context. 16kb too narrow for enhancers. 1Mb wasteful for point-level features. |
| Feature window | +/- 500 bp | Cas9 accessibility is a local property. Nucleosome-level features operate at ~150bp scale. 500bp captures 2-3 nucleosomes in each direction. |
| Negative examples | Cas-OFFinder candidates NOT in GUIDE-seq | More realistic than random genomic sites. These are sites with actual sequence similarity — the model must learn that chromatin, not just sequence, determines cleavage. |

---

## Dependencies

```
# requirements.txt
alphagenome          # AlphaGenome SDK (install from GitHub clone)
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
shap>=0.42
matplotlib>=3.7
seaborn>=0.12
pyBigWig>=0.3       # For reading ENCODE BigWig files
pybedtools>=0.9     # For genomic interval operations
pyliftover>=0.4     # For hg19 to hg38 coordinate conversion
tqdm>=4.65          # Progress bars for long-running extraction
requests>=2.31      # HTTP requests
```

---

## CRITICAL: Biological Edge Cases and Pitfalls

**READ THIS ENTIRE SECTION BEFORE TOUCHING ANY DATA.**

Neither the developer (Nick) nor the coding assistant (Claude) are experienced wet-lab
CRISPR biologists. Dr. Chung is a computational biologist, not a bench scientist. This
section documents every known biological trap, edge case, and silent data corruption
risk in this project. When in doubt, flag the issue and do NOT proceed with assumptions.
Organized by project phase.

---

### PHASE 1 PITFALLS: Data Acquisition and Parsing

#### P1.1 — CHANGE-seq is performed on NAKED DNA, not in cells
**Severity: PROJECT-BREAKING if ignored**

CHANGE-seq (Cameron et al. 2020) digests PURIFIED GENOMIC DNA with Cas9 in a test tube.
There is NO chromatin in this assay. It finds every site in the genome that Cas9 CAN cut
based on sequence alone. This means:
- CHANGE-seq data CANNOT be used as positive labels (evidence of in-cell cutting)
- CHANGE-seq data IS useful as a comprehensive catalog of potential off-target sites
- Only GUIDE-seq (performed IN cells, with chromatin intact) provides valid positive labels

**How to use CHANGE-seq correctly:**
- Use it to EXPAND the candidate off-target set beyond Cas-OFFinder predictions
- Use it to find sites that Cas9 can cut on naked DNA but DOESN'T cut in cells (these
  are likely protected by chromatin — exactly what we're trying to predict)
- NEVER treat CHANGE-seq read counts as evidence of in-cell cleavage

**Red flag to watch for:** If you see a paper or tool treating CHANGE-seq and GUIDE-seq
data interchangeably — they are fundamentally different assays measuring different things.

#### P1.2 — Genome build mismatch (hg19 vs hg38)
**Severity: Silent data corruption**

Different datasets and papers report coordinates in different genome builds:
- Tsai et al. 2015 (GUIDE-seq): May use **hg19** (GRCh37) — paper was published in 2015
  when hg19 was standard. CHECK THE SUPPLEMENTARY TABLES CAREFULLY.
- Cameron et al. 2020 (CHANGE-seq): More likely **hg38** but VERIFY.
- ENCODE data: Available in both builds. ALWAYS download **hg38** versions.
- AlphaGenome: Uses **hg38** exclusively (confirmed in their FAQ).

**If you mix hg19 and hg38 coordinates, everything downstream is garbage.** Coordinates
can shift by thousands to millions of base pairs between builds due to gap closures,
centromere updates, and scaffold reordering.

**How to handle:**
1. For EVERY dataset, find the genome build in the paper methods or supplementary info
2. If a dataset is in hg19, use `pyliftover` to convert ALL coordinates to hg38 BEFORE
   any analysis
3. After liftover, sanity check: pick 5-10 known off-target sites, look them up in the
   UCSC Genome Browser in hg38, verify they land in sensible genomic locations
4. Document the original build of every input file in a data manifest

**How to detect this bug:** If your Cas-OFFinder results don't overlap with GUIDE-seq
validated sites even though you're using the same guide, genome build mismatch is the
first thing to check.

#### P1.3 — Coordinate systems: 0-based vs 1-based
**Severity: Off-by-one errors that silently shift all your features**

Bioinformatics uses TWO different coordinate conventions:
- **0-based, half-open (BED format):** Position 0 is the first base. An interval [0, 20)
  means bases 0 through 19. This is what Python uses. This is what we use internally.
- **1-based, inclusive (VCF, GFF, most paper supplementary tables):** Position 1 is the
  first base. An interval [1, 20] means bases 1 through 20.

These differ by exactly 1, which means:
- If you forget to convert, your chromatin features will be extracted 1 bp away from the
  actual off-target site
- At 1 bp resolution (which AlphaGenome provides), this MATTERS
- AlphaGenome uses 0-based indexing (confirmed in their FAQ)
- Cas-OFFinder output is 0-based
- GUIDE-seq supplementary tables are typically 1-based (verify per dataset)
- VCF files are 1-based

**Rule:** Convert everything to 0-based half-open (BED) on input. Document the original
convention for every data source. When writing output for the paper, convert back to
1-based (which is what biologists expect to read).

#### P1.4 — Guide RNA sequence conventions
**Severity: Wrong off-target search, zero valid results**

Guide sequences can be reported in multiple ways:
- **With or without PAM:** Some papers report 20 nt guide + 3 nt PAM (23 nt total),
  others report only the 20 nt spacer sequence
- **Sense vs antisense strand:** The guide targets one strand; off-target search must
  consider BOTH strands
- **DNA vs RNA notation:** Guide is RNA (contains U), genomic sequence is DNA (contains T).
  Papers may report using either convention.
- **5' to 3' vs 3' to 5':** Guide sequence should always be 5' to 3'. Some papers report
  the target site sequence (which is the complement of the guide).

**How to handle:**
1. For each guide, verify: is this the spacer sequence or the target site complement?
2. Strip PAM if included — Cas-OFFinder adds it from the pattern
3. Convert all U to T for consistency in sequence matching
4. Cross-reference: look up the guide in the original paper and verify it matches a known
   gene target (e.g., EMX1, VEGFA, FANCF are commonly used guides in GUIDE-seq studies)

#### P1.5 — Supplementary table format inconsistency
**Severity: Parsing failures, missing data**

GUIDE-seq supplementary tables from different papers have different:
- Column names (some use "Chr", some "Chromosome", some "chrom")
- Coordinate formats (some include "chr" prefix, some don't)
- Delimiter types (tab, comma, mixed)
- Off-target read count columns (may be called "GUIDE-seq reads", "read count",
  "GUIDE-seq_reads", or just a number)
- Some tables include the on-target site in the same table as off-targets

**How to handle:** Write a parser that is specific to EACH paper's supplementary format.
Do NOT write one generic parser and assume tables are compatible. Manually inspect the
first 10 rows of every input file before parsing.

#### P1.6 — On-target vs off-target in GUIDE-seq data
**Severity: Inflated performance metrics**

GUIDE-seq data includes the ON-TARGET site (the intended cut site) along with off-target
sites. The on-target site will have:
- 0 mismatches to the guide
- Very high read count (often 10-100x higher than any off-target)

**You MUST remove the on-target site from training/evaluation data.** If you leave it in:
- Your model will learn that "0 mismatches = high cleavage" as its primary signal
- Metrics will be inflated because predicting the on-target is trivially easy
- The model won't actually be learning anything about off-target chromatin biology

**How to detect:** Check your data for sites with 0 mismatches. There should be exactly
one per guide (the on-target). Remove these rows.

---

### PHASE 2 PITFALLS: AlphaGenome and Chromatin Biology

#### P2.1 — AlphaGenome predicts from SEQUENCE, not from cell state
**Severity: Conceptual misunderstanding that affects interpretation**

AlphaGenome takes DNA SEQUENCE as input and predicts what chromatin SHOULD look like
for a given cell type based on patterns learned from training data. It does NOT:
- Know the actual state of chromatin in your specific cells
- Account for patient-specific epigenetic changes
- Account for drug-induced chromatin remodeling
- Account for disease-state chromatin alterations (e.g., cancer)

This matters because K562 is a CANCER cell line with extensive chromosomal
rearrangements, translocations, and copy number variations. AlphaGenome predicts based
on the REFERENCE sequence (hg38), but K562's actual genome is different from the
reference at many loci due to cancer-driven mutations.

**Implication for validation:** When comparing AlphaGenome predictions to ENCODE
experimental chromatin in K562, some discrepancies may be real (AlphaGenome predicting
what the reference genome should do, while ENCODE measures what the rearranged cancer
genome actually does). This is a KNOWN LIMITATION — document it, don't debug it away.

#### P2.2 — Cell type identifier mapping
**Severity: Getting wrong cell type predictions or API errors**

AlphaGenome was trained on specific cell types from ENCODE/Roadmap with specific
identifiers. You need to know EXACTLY which cell types are available and what they're
called in the API.

**Potential traps:**
- "K562" in the API might be labeled differently (e.g., "k562", "K-562", an ontology
  term like "CL:0000094", or an ENCODE accession)
- "HEK293" might be listed as "HEK293T" (a subline) — these are different cell lines
  with potentially different chromatin profiles
- If you request a cell type that wasn't in training data, AlphaGenome may extrapolate
  (valid) or may fail silently (returning generic predictions without warning)

**How to handle:**
1. BEFORE any analysis, list ALL available cell types in the AlphaGenome API/model
2. Map each to the cell types used in GUIDE-seq experiments
3. If an exact match doesn't exist, document the closest available and note the
   discrepancy in the paper

#### P2.3 — What "chromatin accessibility" actually means at off-target sites
**Severity: Feature engineering trap**

Chromatin accessibility (DNase/ATAC signal) at an off-target site tells you whether
Cas9 can physically REACH the DNA there. But the relationship is not binary:
- Very high accessibility (active promoter): Site is definitely accessible, high risk
- Moderate accessibility: Site may be partially accessible
- Zero accessibility: Site is likely in heterochromatin, low risk
- BUT: Cas9 can sometimes invade moderately closed regions if given enough time
  (this depends on Cas9 concentration and exposure duration, which varies by experiment)

**Additional nuance:** Accessibility is measured as an average over millions of cells in
bulk experiments (ENCODE). In any given single cell, a locus might be open or closed
stochastically. High bulk accessibility means "open in most cells" — but even low-
accessibility sites are open in SOME cells, leading to rare off-target events.

**Implication:** Don't expect a perfect binary classifier. Even with perfect chromatin
data, some off-target sites in "closed" chromatin will still get cut at low rates.
This is why we use precision-recall curves (which handle continuous scores) rather than
simple accuracy (which requires a binary threshold).

#### P2.4 — Histone mark interpretation
**Severity: Wrong feature engineering or wrong biological conclusions**

If you extract histone modification signals from AlphaGenome, know what they mean:

| Mark | What it indicates | Relevance to off-target risk |
|------|-------------------|------------------------------|
| H3K4me3 | Active promoter | High accessibility, HIGH risk |
| H3K27ac | Active enhancer/promoter | High accessibility, HIGH risk |
| H3K36me3 | Actively transcribed gene body | Moderate accessibility, MODERATE risk |
| H3K27me3 | Polycomb-repressed (facultative heterochromatin) | LOW accessibility, LOW risk — but NOT as tightly closed as constitutive heterochromatin |
| H3K9me3 | Constitutive heterochromatin | VERY LOW accessibility, VERY LOW risk — most tightly packed chromatin |
| CTCF | Insulator/boundary element | Marks boundaries between open and closed domains — may have complex effects |

**Key distinction:** H3K27me3 (Polycomb) and H3K9me3 (constitutive heterochromatin) are
BOTH "repressive" marks but represent very different chromatin states:
- H3K27me3 is REVERSIBLE — regions can switch from closed to open between cell types
- H3K9me3 is more STABLE — regions are constitutively closed in almost all cell types
- For off-target prediction, H3K9me3 is a stronger predictor of "definitely not cut"
  than H3K27me3

#### P2.5 — AlphaGenome resolution vs feature window
**Severity: Subtle but can dilute signal**

AlphaGenome predicts at single-base-pair resolution. But our feature extraction uses
summary statistics over a +/- 1 kb window (2 kb total). This means:
- A narrow accessibility peak (50 bp wide) right at the off-target site will be
  DILUTED when you take the mean over 2000 bp
- The "center" feature (signal at exact off-target position) is more important than
  "mean" for capturing this
- Consider also extracting features at a NARROW window (+/- 50 bp) in addition to the
  wide window

**How to detect signal dilution:** If "center" features consistently outperform "mean"
features in importance rankings, you're diluting signal with the wide window. Add narrow
window features.

#### P2.6 — AlphaGenome output tracks vs ENCODE experimental tracks
**Severity: Apples-to-oranges comparison if not handled**

AlphaGenome outputs PREDICTED signal values. ENCODE provides EXPERIMENTAL signal values.
These are NOT on the same scale:
- AlphaGenome outputs may be log-transformed or normalized differently
- ENCODE bigWig files come in various normalizations (raw signal, fold-change over
  control, p-value signal)
- You CANNOT directly subtract one from the other

**For the correlation check (Week 3):** Use RANK correlation (Spearman) rather than
Pearson. This avoids scale mismatch issues. Alternatively, quantile-normalize both
before comparing.

**For the model comparison (Model B vs Model C):** This is actually fine because Model B
and Model C use SEPARATE feature sets. You're not comparing the values directly — you're
comparing model performance when trained on different features. The models handle scale
differences internally.

---

### PHASE 3 PITFALLS: Label Construction and Data Assembly

#### P3.1 — Negative label reliability
**Severity: Systematic bias in model evaluation**

A "negative" in our dataset means: Cas-OFFinder found this site (sequence is similar
to guide), but GUIDE-seq did NOT detect cleavage here. We label it as "not cut."

But GUIDE-seq has LIMITED SENSITIVITY. It can miss real off-target sites that are cut
at low frequency. This means some of our "negatives" are actually false negatives
(weakly positive sites mislabeled as negative).

**Implications:**
- Our model will be trained with noisy labels
- Performance metrics will be somewhat underestimated (model may correctly predict
  some "false negatives" as positive, but be penalized for it)
- This is a KNOWN LIMITATION of every off-target prediction study. It is NOT unique to
  us. Acknowledge it in the paper but don't try to fix it — there's no clean solution.

**How to partially mitigate:**
- When constructing negatives, prefer sites with HIGH mismatch counts (4+ mismatches)
  that are far from any GUIDE-seq hit. These are more reliably true negatives.
- Consider excluding "near-miss" sites (sites 1-2 mismatches that are close to but
  not overlapping GUIDE-seq hits) — these are most likely to be false negatives.

#### P3.2 — Class imbalance is EXTREME and expected
**Severity: Misleading metrics if not handled**

Expected ratio: ~1-5% positives, 95-99% negatives. This is biologically correct — most
sequence-similar sites in the genome are NOT cut by Cas9.

**Traps:**
- ACCURACY is meaningless. A model predicting "never cut" gets 95-99% accuracy.
- ROC-AUC is overly optimistic with extreme imbalance (high true negative rate inflates
  the metric). DO NOT use ROC-AUC as primary metric.
- Use PRECISION-RECALL curves and Average Precision (AP) as primary metrics.
- Report both PR-AUC and ROC-AUC, but lead the paper with PR results.

**In XGBoost:** Set `scale_pos_weight = n_negative / n_positive`. This tells the model
to weight positive examples more heavily during training. Without this, the model may
converge to "predict all negative" which looks good on accuracy but is useless.

#### P3.3 — Data leakage from correlated off-targets
**Severity: Inflated cross-validation performance**

Off-target sites from the SAME guide are correlated:
- They share the same guide sequence (so all sequence-based features are similar)
- They may cluster in similar genomic regions
- Their experimental detection shares the same GUIDE-seq experiment (batch effects)

If you do standard k-fold cross-validation, off-targets from the same guide can appear
in both training and test sets. The model learns guide-specific patterns and "cheats."

**REQUIRED: Leave-one-guide-out cross-validation.** Hold out ALL sites from one guide
at a time. Train on all other guides. This is standard practice in the field and
non-negotiable for a credible paper.

#### P3.4 — Coordinate matching tolerance
**Severity: Missing positives or creating false positives**

When matching Cas-OFFinder candidates to GUIDE-seq validated sites, coordinates may not
match exactly due to:
- Different reporting conventions (start of guide vs start of protospacer vs start of
  PAM)
- Rounding or truncation in supplementary tables
- Strand conventions (a site on the minus strand may be reported from either end)

**Use +/- 10 bp tolerance for matching.** If a Cas-OFFinder site overlaps a GUIDE-seq hit
within 10 bp, call it a match.

**BUT: verify this tolerance works correctly.** After matching, manually spot-check
5-10 matched pairs. Look at the sequences — do they actually correspond to the same
genomic site? If your tolerance is too loose, you'll match DIFFERENT off-target sites
to each other and corrupt labels.

#### P3.5 — Guides with no validated off-targets
**Severity: Bias in training data**

Some guides in GUIDE-seq datasets have very few or ZERO validated off-targets. This
could mean:
- The guide is genuinely very specific (few off-targets)
- The GUIDE-seq experiment failed for that guide (technical failure)
- The guide was poorly expressed (low editing efficiency at on-target too)

**How to handle:** Only include guides where the on-target site shows robust GUIDE-seq
signal (high read count). If on-target signal is weak, the guide likely wasn't active,
and absence of off-targets is due to lack of Cas9 activity, not specificity.

---

### PHASE 4 PITFALLS: Modeling and Evaluation

#### P4.1 — Feature importance vs biological importance
**Severity: Wrong conclusions in paper**

XGBoost feature importance (gain/SHAP) tells you which features the MODEL found useful,
not necessarily which are biologically most important. Traps:
- If two features are highly correlated (e.g., DNase and H3K27ac — both mark open
  chromatin), the model may use one and ignore the other. This doesn't mean the ignored
  one is biologically unimportant.
- CFD score already partially captures some chromatin information implicitly (guides
  with off-targets in open chromatin tend to have more detected off-targets in training
  data, so CFD-correlated features may steal importance)

**How to handle:** Do an ABLATION study. Remove feature groups one at a time and report
performance drop. This is more informative than raw feature importance.

#### P4.2 — Overfitting on small dataset
**Severity: Results that don't replicate**

If we end up with ~30 guides and ~10,000-50,000 candidate sites, but only ~200-500
positives, overfitting is a real risk, especially with 28+ chromatin features.

**How to detect:**
- Large gap between training and test performance in cross-validation
- Model performance varies wildly across CV folds
- Very high performance on one or two guides dominates the average

**How to mitigate:**
- Start with simple model (logistic regression) before XGBoost
- If XGBoost overfits, reduce max_depth (try 3-4 instead of 6)
- Reduce n_estimators with early stopping
- Try feature selection: use only top-5 features from logistic regression

#### P4.3 — The "K562 problem"
**Severity: Results may not generalize**

K562 is a chronic myeloid leukemia cell line with:
- BCR-ABL translocation (Philadelphia chromosome)
- Near-triploid karyotype (many duplicated chromosomes)
- Massive structural variants and copy number changes
- Constitutively active signaling pathways that alter chromatin globally

This means K562 has an UNUSUAL chromatin landscape compared to normal cells. If our
model works great on K562 but fails on HEK293, it may be because:
- K562 has very distinct open/closed chromatin boundaries (easier to predict)
- Other cell types have more nuanced chromatin states

**How to handle:** Always report per-cell-line results separately. If K562 performance
is much higher than HEK293, discuss this as a limitation. The cell-type transfer demo
in Week 7 will test whether the model generalizes.

#### P4.4 — Interpreting the three-model comparison
**Severity: Wrong conclusions about what's driving performance**

The key comparison is:
- Model A (sequence only) vs Model C (sequence + AlphaGenome)
- Model B (sequence + ENCODE experimental) vs Model C

**Trap 1:** If Model A is already very good (AUROC > 0.90), chromatin features may not
add much. This doesn't mean chromatin doesn't matter — it means sequence-based scores
already capture most of the variance in THIS specific dataset (which is selected for
sequence-similar sites by construction).

**Trap 2:** If Model B and Model C perform identically, this could mean AlphaGenome
predictions perfectly match experimental data (great!) OR that neither model is actually
using chromatin features effectively (both are relying on sequence features). Check
feature importances to distinguish these cases.

**Trap 3:** If Model C is BETTER than Model B, this seems too good to be true (how can
predictions beat ground truth?). Possible explanations:
- AlphaGenome provides a "cleaner" signal than noisy experimental data
- AlphaGenome's training data included information about broader sequence context that
  happens to correlate with off-target susceptibility
- Overfitting to specific features
- Bug in feature extraction

If C > B, investigate thoroughly before celebrating. It's publishable either way but the
explanation matters.

---

### PHASE 5 PITFALLS: Cell-Type Transfer and Interpretation

#### P5.1 — "Cell-type-specific" vs "cell-type-invariant" off-targets
**Severity: Misinterpreting results**

Some off-target sites are in constitutively open chromatin (housekeeping gene promoters)
— these will be high-risk in ALL cell types. Others are in constitutive heterochromatin
— low-risk in ALL cell types. Only a fraction of sites will show CELL-TYPE-SPECIFIC
differences.

**Expectation:** Maybe 10-30% of off-target sites will show meaningful variation across
cell types. The rest will be consistently high or consistently low regardless of cell
type. This is BIOLOGICALLY CORRECT — don't be disappointed if the heatmap shows many
rows that don't vary.

The value of SafeGenome is precisely in those 10-30% of sites that DO vary. Even a
small number of cell-type-specific re-rankings is clinically meaningful (one wrongly
ignored off-target in cardiac tissue could cause arrhythmia).

#### P5.2 — Cas9 concentration and delivery efficiency confound multi-cell-line comparison
**Severity: Confounding variable**

Different GUIDE-seq experiments in K562 vs HEK293 used different Cas9 concentrations,
delivery methods (electroporation vs lipofection), and exposure times. Higher Cas9
concentrations = more off-target cutting, even at low-accessibility sites.

This means some differences between K562 and HEK293 off-target profiles may be due to
EXPERIMENTAL PROTOCOL DIFFERENCES, not chromatin differences.

**How to handle:** Acknowledge in the paper. This is why we focus on RANKING (which
off-targets are highest risk) rather than absolute cut frequency. Rankings are more
robust to protocol differences than quantitative cut rates.

#### P5.3 — Validation on cell types without GUIDE-seq data
**Severity: Unvalidatable claims**

For the cell-type transfer demo, we'll run SafeGenome on CD4+ T cells, hepatocytes,
neurons, etc. — cell types where NO GUIDE-seq data exists. We CANNOT validate these
predictions directly.

**How to handle honestly:**
- Present these as PREDICTIONS, not validated results
- Show that the model works on K562 and HEK293 (where we can validate)
- Show that the predictions CHANGE across cell types (demonstrating the tool does
  something, even if we can't prove it's correct)
- Frame it as: "SafeGenome extends off-target analysis to cell types that were
  previously inaccessible" — which is true regardless of accuracy
- Suggest experimental validation as future work

---

### GENERAL PITFALLS: Things That Can Go Wrong Anywhere

#### G1 — Repetitive genomic regions
Off-target sites are enriched near REPETITIVE ELEMENTS (SINEs, LINEs, segmental
duplications) because these regions have many near-identical sequences. Problems:
- Cas-OFFinder may return an excessive number of candidates in repetitive regions
- AlphaGenome may predict poorly in repetitive regions (these are underrepresented in
  training data and often masked during training)
- Sequence alignment is ambiguous in repeats — coordinate matching is harder

**How to detect:** If a guide has >10,000 Cas-OFFinder hits, check whether most are in
repetitive elements. Consider filtering candidates using RepeatMasker annotations.

#### G2 — Strand orientation matters for some features
Cas9 targets ONE strand, but chromatin features are typically strand-independent (DNase-
seq signal is the same regardless of which strand you look at). However:
- Mismatch positions are defined relative to the guide-target alignment, which IS
  strand-specific
- When extracting AlphaGenome features, query the same window regardless of which
  strand the off-target is on
- Some transcription factor binding tracks may be strand-specific. If using TF
  binding features, verify how AlphaGenome reports them.

#### G3 — Missing data handling
If AlphaGenome returns no data for a locus (API error, locus in unresolved gap, etc.):
- Do NOT fill with zeros (zero signal means heterochromatin — this would create false
  features)
- Do NOT fill with mean (this would bias toward average accessibility)
- DROP the row entirely, or flag it for re-query
- Log every failed API call with the exact coordinates for debugging

#### G4 — Multiple testing in guide-by-guide analysis
If you report per-guide results (which you should), remember that with 30 guides, some
will show strong results and some will show weak results by chance. Don't cherry-pick
the best-performing guides for the paper. Report:
- Overall (pooled) metrics as the primary result
- Per-guide results in a supplementary table showing ALL guides
- Median and interquartile range across guides (robust to outliers)

#### G5 — AlphaGenome training data overlap
AlphaGenome was trained on ENCODE data, which includes K562 and HEK293 chromatin data.
This means AlphaGenome's predictions for K562 are partially "memorized" from training,
not pure generalization. This is:
- GOOD for our validation step (predictions should be accurate for training cell types)
- POTENTIALLY MISLEADING for claims about generalization to unseen cell types
- Address in paper: our K562/HEK293 validation tests AlphaGenome on cell types it was
  trained on. True generalization would require validation on a cell type NOT in
  AlphaGenome's training set — which we cannot do because no GUIDE-seq data exists for
  unseen cell types. Acknowledge this circular limitation honestly.

#### G6 — Publication framing for negative results
If the main result is negative (AlphaGenome features don't help), the paper is STILL
publishable but requires different framing:
- Title becomes benchmarking/evaluation focused: "Evaluating DNA foundation model
  predictions for CRISPR off-target safety scoring"
- Focus shifts to: WHY don't the predictions help? Is it resolution? Is it prediction
  accuracy at off-target-enriched genomic contexts? Is it that chromatin state is
  more complex than accessibility alone?
- This analysis of failure modes is valuable to both the CRISPR and genomic AI
  communities
- Target: Bioinformatics, NAR Genomics, or Genome Biology short report

#### G7 — PAM variants and non-SpCas9 nucleases
Our pipeline is designed for SpCas9 (NGG PAM). Do NOT generalize results to:
- SpCas9-HF1 or eSpCas9 (engineered high-fidelity variants with different off-target
  profiles)
- Cas12a/Cpf1 (different PAM, different cleavage mechanism)
- Base editors or prime editors (no DSB, different off-target biology)
State this scope limitation clearly in the paper methods.

#### G8 — Bulge-containing off-targets
Cas-OFFinder can find off-targets with DNA or RNA bulges (insertions/deletions in the
guide-target alignment). These are biologically real but:
- Much rarer than mismatch-only off-targets
- CFD scoring doesn't handle bulges (it was designed for mismatches only)
- Need separate handling in feature engineering (bulge position, bulge size)
- Consider starting with mismatch-only off-targets for simplicity, add bulges later
  if time permits

---

### QUICK REFERENCE: "Something Looks Wrong" Checklist

If results look unexpectedly good or bad, check these IN ORDER:

1. **Genome build:** Are all coordinates in hg38? Did any hg19 data sneak in?
2. **On-target included?** Did you accidentally leave on-target sites in the data?
3. **Coordinate system:** 0-based vs 1-based mismatch?
4. **Cell type mapping:** Is AlphaGenome actually predicting for the right cell type?
5. **Feature scale:** Are AlphaGenome features on the same scale as expected?
6. **Label assignment:** Are positives actually GUIDE-seq validated (not CHANGE-seq)?
7. **Cross-validation:** Is it truly leave-one-guide-out? Any leakage?
8. **Class weights:** Is scale_pos_weight set in XGBoost?
9. **NaN values:** Any missing features that defaulted to 0?
10. **Guide filtering:** Did you exclude guides with failed on-target signal?

---

## Known Issues / Bugs

(Update this section as issues arise)

- None yet — project not started.

---

## References

1. Avsec Z et al. (2026) Advancing regulatory variant effect prediction with AlphaGenome. Nature.
2. Lazzarotto CR et al. (2020) CHANGE-seq reveals genetic and epigenetic effects on CRISPR-Cas9 genome-wide activity. Nat Biotechnol 38:1317-1327.
3. Tsai SQ et al. (2015) GUIDE-seq enables genome-wide profiling of off-target cleavage by CRISPR-Cas nucleases. Nat Biotechnol 33:187-197.
4. Doench JG et al. (2016) Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9. Nat Biotechnol 34:184-191.
5. Kim D and Kim JS (2018) DIG-seq: a genome-wide CRISPR off-target profiling method using chromatin DNA. Genome Res 28:1894-1900.
6. Bae S et al. (2014) Cas-OFFinder: a fast and versatile algorithm that searches for potential off-target sites of Cas9 RNA-guided endonucleases. Bioinformatics 30:1473-1475.
7. Kuscu C et al. (2014) Genome-wide analysis reveals characteristics of off-target sites bound by the Cas9 endonuclease. Nat Biotechnol 32:677-683.

---

## Session Log

(Update at the end of every Claude session)

| Date | Session | What was done | What's next |
|------|---------|---------------|-------------|
| 2026-02-11 | Planning | Full project spec, architecture, validation strategy | Download data, set up environment |

---

*Last updated: 2026-02-11*
