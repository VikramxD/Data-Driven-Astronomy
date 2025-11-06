# Data Directory

This directory contains all data files for the Data-Driven Astronomy project.

## Structure

```
data/
├── raw/              # Original, unprocessed data files
│                     # Place CSV files from SDSS, GMRT surveys here
│
├── processed/        # Cleaned and preprocessed data
│                     # Transformed datasets ready for ML models
│
└── results/          # Output files from analysis
    ├── matched_catalogs_v1.txt
    ├── matched_catalogs_v2.txt
    └── matched_catalogs_v3.txt
```

## Data Sources

### SDSS Optical Data
- **Source**: Sloan Digital Sky Survey (SDSS)
- **Format**: CSV files with photometric data
- **Features**: 5-band photometry (u, g, r, i, z), morphological measurements
- **Size**: ~370,000 galaxies

### GMRT Radio Data
- **Source**: Giant Metrewave Radio Telescope (GMRT)
- **Format**: CSV files with radio source positions
- **Size**: ~5,400 radio sources

## Usage

Place your data files in the appropriate directories:

1. **Raw data** → `data/raw/`
   - `gmrt.csv` - GMRT radio catalog
   - `opticaldata.csv` - SDSS optical catalog
   - `opticaldatafinals_SDSS.csv` - Final SDSS dataset

2. **Processed data** → `data/processed/`
   - Feature-engineered datasets
   - Normalized/standardized data

3. **Results** → `data/results/`
   - Cross-matched catalogs
   - Model predictions
   - Analysis outputs

## Note

Large data files (>100MB) should be added to `.gitignore` and downloaded separately.
For data sources and download instructions, see the main README.md.
