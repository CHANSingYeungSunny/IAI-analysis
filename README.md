# IAI — Immune Ageing Index for H3N2 Vaccine Response Decline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reproducibility package for the manuscript:

> **An Immune Ageing Index Integrates Single-Cell, Repertoire, and Transcriptomic Data to Track H3N2 Vaccine Response Decline**

## Overview

The Immune Ageing Index (IAI) is a composite, weekly-resolved metric integrating:
- **Zcells**: Per-project B/T cell ratio from 4.29M PBMC single-cell profiles (553 donors, 24 projects)
- **Zclonotype**: Shannon entropy of VDJ clonotype diversity (111 samples)
- **Zgene**: Mean expression of 500 highly variable genes in GTEx blood (1,111 samples)

IAI is aligned with US FluNet H3N2 surveillance (2017–2024) and CDC VE 2023–24 data. ARIMA, LSTM, and TFT models forecast IAI with external covariates. Interpretability via permutation importance and attention weights.

## Prerequisites

- Python 3.10+ (tested on 3.10 with CUDA 11.8)
- GPU recommended (NVIDIA GTX 1650 or better) for LSTM/TFT training
- ~16 GB RAM for large CSV processing

## Installation

```bash
git clone https://github.com/CHANSingYeungSunny/IAI-analysis.git
cd IAI-analysis
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Data

Place the following files in `input_data/`:

| File | Source | Size |
|------|--------|------|
| `pbmc_obs.csv` | PBMC single-cell metadata (4.29M cells) | 2.9 GB |
| `pbmc_var.csv` | PBMC highly variable gene statistics | < 1 MB |
| `vdj_obs.csv.gz` | VDJ immune repertoire data | < 1 MB |
| `GTEx_v11_gene_tpm.gct.gz` | GTEx v11 TPM expression matrix | 2.5 GB |
| `GTEx_v11_SampleAttributesDS.txt` | GTEx sample metadata | < 10 MB |
| `VIW_FNT.csv` | WHO FluNet surveillance data | 30 MB |
| `CDC_VE_2023_24_full.csv` | CDC VE 2023–24 (age-stratified) | < 1 MB |

The pipeline also reads `input_data/CDC_VE_rows.csv` (2025–26 weekly VE) if available.

### Data Availability

Two large files exceed GitHub's 100 MB limit and are hosted on Zenodo:

| File | Size | DOI |
|------|------|-----|
| `pbmc_obs.csv` | 2.9 GB | [10.5281/zenodo.20496692](https://doi.org/10.5281/zenodo.20496692) |
| `GTEx_v11_gene_tpm.gct.gz` | 2.5 GB | [10.5281/zenodo.20496692](https://doi.org/10.5281/zenodo.20496692) |

**Complete dataset**: Zenodo DOI [10.5281/zenodo.20496692](https://doi.org/10.5281/zenodo.20496692).  
**Code and small data files**: GitHub Release DOI [10.5281/zenodo.20497072](https://doi.org/10.5281/zenodo.20497072).

To reproduce the full pipeline, download the two large files from Zenodo and place them in `input_data/`. All other files are included in this repository.

## Usage

```bash
# Full pipeline (IAI computation + forecasting + plots)
python main.py

# Generate CSV tables for all figures
python generate_manuscript_tables.py
```

## Output

| File | Description |
|------|-------------|
| `results/IAI_time_series.csv` | 350 weekly IAI values (2017–2024) |
| `results/Forecast_results.csv` | ARIMA + LSTM 12-step forecasts |
| `results/Correlation_results.csv` | Pearson, Spearman, CDC VE comparisons |
| `results/tables/Table*.csv` | All tables as CSV |
| `results/*.png` | Distribution, forecast, SHAP, attention, scatter plots |

## Key Findings

- **IAI vs H3N2**: Spearman ρ = −0.431 (p = 2.79 × 10⁻¹⁷), 7.1× AH3 reduction Q1→Q4
- **2023–24 flu season**: Mean IAI = −0.468; children H3N2 VE = 28.3%, adults = 24.5%
- **LSTM forecast**: Recovery −0.574 → −0.512; ARIMA: Decline to −0.633
- **GPU-accelerated** LSTM (86% loss reduction) and TFT (18.1K params)

## Project Structure

```
IAI-analysis/
├── main.py                          # Pipeline entry point
├── IAI_Pipeline_ImmunityAgeing.py   # Core computation modules
├── generate_manuscript_tables.py    # CSV table generator
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── LICENSE                          # MIT License
├── input_data/                      # Input data files
├── results/                         # Output files
│   ├── tables/                      # Manuscript tables (CSV)
│   ├── IAI_time_series.csv
│   ├── Forecast_results.csv
│   ├── Correlation_results.csv
│   ├── RESULTS_DESCRIPTION.txt
│   └── *.png
└── test_*.py                        # Test scripts
```

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use this code or data, please cite the associated manuscript and DOI (Zenodo). The DOI will be generated upon the first GitHub Release.
