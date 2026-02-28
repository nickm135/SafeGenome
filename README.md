# SafeGenome

**Cell-type-specific CRISPR off-target prediction using AlphaGenome-predicted chromatin**

SafeGenome predicts CRISPR-Cas9 off-target cleavage risk for any cell type by replacing expensive experimental chromatin data (DNase-seq, ChIP-seq) with AlphaGenome's sequence-based chromatin predictions. It combines sequence features (CFD score, mismatch encoding) with predicted chromatin accessibility and histone modifications in an XGBoost classifier trained via leave-one-guide-out cross-validation. This enables off-target safety assessment in cell types that lack experimental epigenomic data — a key bottleneck for clinical CRISPR applications.

## Key Results

Three models compared on 4,719 off-target sites (18 guides, 266 positives) using leave-one-guide-out CV:

| Model | Features | Mean AUPRC | OOF AUPRC | Bootstrap p vs Model C |
|-------|----------|-----------|-----------|----------------------|
| **Model A** | Sequence only | 0.673 +/- 0.242 | 0.441 | p < 0.0001 |
| **Model B** | Sequence + ENCODE experimental | 0.687 +/- 0.218 | 0.463 | p = 0.0008 |
| **Model C** | Sequence + AlphaGenome predicted | **0.721 +/- 0.207** | **0.507** | -- |
| CFD score | Raw score (no ML) | -- | 0.297 | -- |
| MIT score | Raw score (no ML) | -- | 0.263 | -- |

Model C (AlphaGenome-predicted chromatin) significantly outperforms both the sequence-only baseline and the experimental chromatin gold standard (paired bootstrap, 10,000 resamples). SHAP analysis confirms 7 of the top 15 features are AlphaGenome chromatin predictions.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/safegenome.git
cd safegenome
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Additional requirements

- **AlphaGenome API key**: Add to `config.py` (see `config.py.example`)
- **Cas-OFFinder**: [Download v2.4.1](https://github.com/nicholasgasior/cas-offinder/releases) and set path in `config.py`
- **ENCODE BigWig files**: Download via instructions in `scripts/04_extract_features.py`
- **hg38 reference genome**: Download and split into per-chromosome FASTAs

## Usage

The pipeline runs as numbered scripts:

```bash
# 1. Parse experimental datasets (GUIDE-seq, CHANGE-seq, DeepCRISPR, DISCOVER-seq)
python scripts/02_parse_datasets.py

# 2. Generate candidate off-target sites via Cas-OFFinder
python scripts/03_generate_candidates.py

# 3. Extract sequence + chromatin features
python scripts/04_extract_features.py

# 4. Train and evaluate three-model comparison (LOGO CV)
python scripts/05_train_models.py

# 5. Cell-type transfer demo (K562, HEK293, CD4+ T cells, Hepatocytes)
python scripts/06_cell_type_transfer.py --validate

# 6. Paper analyses (AG vs ENCODE correlation, SHAP importance)
python scripts/07_paper_analyses.py

# 7. Benchmarks (CFD/MIT baselines, held-out validation)
python scripts/08_benchmarks.py

# 8. Bootstrap significance test
python scripts/09_bootstrap_test.py
```

### Running tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
safegenome/
├── src/                          # Core library modules
│   ├── utils.py                  # Genomic utilities, logging, caching
│   ├── sequence_features.py      # CFD/MIT scoring, mismatch features
│   ├── offtarget_finder.py       # Cas-OFFinder wrapper
│   ├── alphagenome_client.py     # AlphaGenome API client + caching
│   ├── encode_client.py          # ENCODE BigWig feature extraction
│   ├── feature_matrix.py         # Feature assembly + imputation
│   ├── model.py                  # XGBoost training, CV, evaluation
│   └── cell_type_transfer.py     # Cross-cell-type prediction
├── scripts/                      # Numbered pipeline scripts (02-09)
├── tests/                        # Unit tests (pytest)
├── data/
│   ├── processed/                # Parsed datasets (parquet)
│   ├── models/                   # Trained models + metrics
│   └── results/                  # Analysis outputs (CSV)
└── figures/                      # Publication figures (PNG)
```

## Data Sources

- **GUIDE-seq**: Tsai et al., Nature Biotechnology 2015 (HEK293 + U2OS)
- **CHANGE-seq**: Lazzarotto et al., Nature Biotechnology 2020 (T cells)
- **DeepCRISPR**: Chuai et al., Genome Biology 2018 (K562)
- **DISCOVER-seq**: Wienert et al., Science 2019 (K562)
- **AlphaGenome**: Avsec et al., Nature 2026

## Citation

```
@article{safegenome2026,
  title={SafeGenome: Cell-type-specific CRISPR off-target prediction using AlphaGenome-predicted chromatin},
  author={TODO},
  journal={TODO},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
