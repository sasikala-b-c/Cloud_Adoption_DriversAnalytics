# Cloud Adoption Drivers Analysis (SDG 9)

Analyze key drivers of cloud adoption aligned with UN SDG 9 (Industry, Innovation and Infrastructure). The project generates a synthetic dataset, runs exploratory data analysis and simple models, and saves plots and a markdown report for quick publication on GitHub.

## Why SDG 9?
SDG 9 promotes resilient infrastructure, inclusive and sustainable industrialization, and fosters innovation. Cloud adoption is a core digital infrastructure capability that enables productivity, R&D, SME digitization, and scalable services.

## Features
- Self-contained: generates its own synthetic dataset.
- EDA visuals: correlation heatmap, feature distributions, and importance charts.
- Models: Linear Regression (interpretable) and Random Forest (nonlinear).
- Outputs: CSV, PNG plots, and a Markdown report in the `outputs/` folder.

## Repository Structure
```
sdg9-cloud-adoption/
  ├─ data/
  │  └─ .gitkeep
  ├─ outputs/
  │  └─ .gitkeep
  ├─ src/
  │  ├─ __init__.py
  │  └─ analyze.py
  ├─ .gitignore
  ├─ README.md
  └─ requirements.txt
```

## How to Run (Windows PowerShell)
```
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\analyze.py
```
Outputs will be written to `outputs/`:
- `cloud_adoption_synthetic.csv`
- `correlation_heatmap.png`
- `feature_distributions.png`
- `linear_coefficients.png`
- `rf_feature_importance.png`
- `report.md`

## Interpretation Guide
- Higher infrastructure quality, broadband penetration, R&D intensity, and human capital typically increase adoption.
- Higher cloud cost index tends to reduce adoption.
- Random Forest importance shows nonlinear and interaction effects; Linear Regression coefficients provide directionality and magnitude.

## License
MIT
# Cloud_Adoption_DriversAnalytics
