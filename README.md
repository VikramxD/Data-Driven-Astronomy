# Data-Driven Astronomy

<div align="center">

**Automated galaxy classification and photometric redshift prediction using machine learning**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#features) •
[Installation](#installation) •
[Usage](#usage) •
[Documentation](#documentation) •
[Contributing](#contributing)

</div>

---

## Overview

This project implements machine learning algorithms for automated astronomical analysis, specifically:

- **Galaxy Morphology Classification**: Automated classification into Hubble sequence types (Elliptical, Spiral, Irregular, Lenticular)
- **Photometric Redshift Estimation**: Distance prediction from multi-band optical colors
- **Catalog Cross-Matching**: Spatial matching of astronomical sources across different surveys

Built on **370,000+ galaxies** from the Sloan Digital Sky Survey (SDSS) and inspired by the [Galaxy Zoo](https://www.galaxyzoo.org/) project.

---

## Features

### 🔭 Astronomical Analysis
- **Multi-survey catalog cross-matching** using `astropy` coordinate matching
- **5-band photometry** analysis (u, g, r, i, z filters)
- **Morphological feature extraction** (ellipticity, concentration indices, color indices)

### 🤖 Machine Learning
- **Decision Tree classifiers** for galaxy morphology
- **Decision Tree regressors** for photometric redshift prediction
- **K-Fold cross-validation** for robust model evaluation
- **Feature engineering** for astronomical properties

### 📊 Data Processing
- GMRT radio catalog cross-matching (~5,400 sources)
- SDSS optical catalog processing (370K+ galaxies)
- Automated data pipeline from raw catalogs to ML-ready features

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/VikramxD/Data-Driven-Astronomy.git
cd Data-Driven-Astronomy

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebooks
jupyter notebook notebooks/
```

---

## Usage

### Quick Start Example

#### 1. Catalog Cross-Matching

```python
from src.crossmatch import crossmatch, save_matches
import numpy as np

# Load catalogs (RA, Dec in degrees)
gmrt_catalog = np.genfromtxt('data/raw/gmrt.csv', delimiter=',',
                              skip_header=55, usecols=[5, 6])
sdss_catalog = np.genfromtxt('data/raw/opticaldata.csv', delimiter=',',
                              skip_header=1, usecols=[1, 2])

# Cross-match with 5 degree maximum distance
matches, no_matches, time_taken = crossmatch(gmrt_catalog, sdss_catalog, max_dist=5)

# Save results
save_matches(matches, 'data/results/matched_catalogs.txt')
print(f"Matched {len(matches)} sources in {time_taken:.2f}s")
```

#### 2. Photometric Redshift Prediction

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load SDSS data
data = pd.read_csv('data/raw/opticaldatafinals_SDSS.csv')

# Create color index features
features = np.column_stack([
    data['u'] - data['g'],  # u-g color
    data['g'] - data['r'],  # g-r color
    data['r'] - data['i'],  # r-i color
    data['i'] - data['z']   # i-z color
])
targets = data['redshift']

# Train model
model = DecisionTreeRegressor(max_depth=19)
model.fit(features, targets)

# Predict redshifts
predictions = model.predict(features)
median_error = np.median(np.abs(predictions - targets))
print(f"Median prediction error: {median_error:.4f}")
```

### Jupyter Notebooks

The project includes numbered notebooks for complete workflows:

1. **`01_catalog_crossmatching.ipynb`** - Match GMRT and SDSS catalogs
2. **`02_galaxy_classification.ipynb`** - Train morphology classifiers
3. **`03_photometric_redshift.ipynb`** - Redshift estimation pipeline
4. **`04_optical_analysis.ipynb`** - Optical data exploration

```bash
# Run all notebooks in order
jupyter notebook notebooks/
```

---

## Repository Structure

```
Data-Driven-Astronomy/
├── notebooks/              # Jupyter notebooks (numbered workflow)
│   ├── 01_catalog_crossmatching.ipynb
│   ├── 02_galaxy_classification.ipynb
│   ├── 03_photometric_redshift.ipynb
│   ├── 04_optical_analysis.ipynb
│   └── exploratory/       # Experimental analyses
│
├── src/                   # Source code modules
│   ├── crossmatch/        # Catalog matching algorithms
│   └── models/            # ML model definitions
│
├── data/                  # Data directory (see data/README.md)
│   ├── raw/              # Original survey data
│   ├── processed/        # Cleaned datasets
│   └── results/          # Analysis outputs
│
├── docs/                  # Documentation
│   ├── galaxy_classification_theory.md  # Detailed astrophysics background
│   ├── images/           # Galaxy morphology examples
│   └── reports/          # Project reports
│
└── tests/                # Unit tests
```

---

## Data Sources

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| **SDSS Optical** | [Kaggle](https://www.kaggle.com/bhanvimenghani/optical-csv) | 370K+ galaxies | 5-band photometry + morphology |
| **GMRT Radio** | GMRT Survey | ~5,400 sources | Radio source positions |

### Data Download

Large data files are not included in the repository. Download them from:

- **SDSS Data**: https://www.kaggle.com/bhanvimenghani/optical-csv
- **GMRT Catalog**: Place `gmrt.csv` in `data/raw/`

Place downloaded files in `data/raw/` before running notebooks.

---

## Model Performance

| Task | Model | Metric | Performance |
|------|-------|--------|-------------|
| Photometric Redshift | Decision Tree (depth=19) | Median Δz | 0.014 |
| Galaxy Classification | Decision Tree | 10-Fold CV Accuracy | ~85-90%* |

*Performance varies by morphological class

---

## Documentation

### For Users
- 📘 [Galaxy Classification Theory](docs/galaxy_classification_theory.md) - Detailed astrophysics background
- 📊 [Data Documentation](data/README.md) - Data sources and format specifications
- 🔬 [Project Report](docs/reports/project_report.docx) - Complete analysis report

### For Developers
- 🧪 Run tests: `pytest tests/` (coming soon)
- 📝 Code style: [Black](https://black.readthedocs.io/) formatting
- 🔍 Type hints: Python 3.7+ annotations

---

## Technologies

<table>
<tr>
<td>

**Core Libraries**
- Python 3.7+
- NumPy
- Pandas
- Matplotlib

</td>
<td>

**Astronomy**
- Astropy
- Celestial coordinates
- Unit conversions

</td>
<td>

**Machine Learning**
- Scikit-learn
- Decision Trees
- Cross-validation

</td>
</tr>
</table>

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

## Roadmap

- [ ] Add support for ensemble models (Random Forest, XGBoost)
- [ ] Implement neural network classifiers
- [ ] Add automated data download scripts
- [ ] Create REST API for predictions
- [ ] Add comprehensive test suite
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{data_driven_astronomy,
  author = {Your Name},
  title = {Data-Driven Astronomy: Machine Learning for Galaxy Classification},
  year = {2024},
  url = {https://github.com/VikramxD/Data-Driven-Astronomy}
}
```

### References

- Galaxy Zoo Project: https://www.galaxyzoo.org/
- SDSS Data: https://www.sdss.org/
- Kaggle Dataset: https://www.kaggle.com/bhanvimenghani/optical-csv

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Galaxy Zoo Team** - Inspiration for crowd-sourced classification
- **Sloan Digital Sky Survey (SDSS)** - Providing open astronomical data
- **GMRT Survey** - Radio catalog data
- **Kaggle Community** - Data preprocessing and sharing

---

## Contact

**Project Maintainer**: [Your Name]

- GitHub: [@VikramxD](https://github.com/VikramxD)
- Issues: [Report a bug](https://github.com/VikramxD/Data-Driven-Astronomy/issues)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the astronomy and data science community

</div>
