from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def ensure_dirs(base: Path) -> dict:
    data_dir = base / "data"
    out_dir = base / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"data": data_dir, "out": out_dir}


def generate_synthetic_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Core drivers related to SDG 9
    infrastructure_quality = rng.uniform(0.3, 0.95, n)
    broadband_penetration = rng.uniform(0.2, 0.98, n)
    rnd_intensity = rng.normal(0.02, 0.01, n).clip(0, 0.08)  # R&D spending share
    sme_digital_literacy = rng.uniform(0.2, 0.95, n)
    regulatory_environment = rng.uniform(0.2, 0.9, n)
    human_capital_index = rng.uniform(0.3, 0.95, n)
    cybersecurity_readiness = rng.uniform(0.2, 0.9, n)

    # Economic/structural context
    cloud_cost_index = rng.normal(0.6, 0.15, n).clip(0.2, 1.2)  # higher is more costly
    energy_reliability = rng.uniform(0.3, 0.95, n)
    manufacturing_share = rng.normal(0.22, 0.08, n).clip(0.05, 0.5)
    urbanization_rate = rng.normal(0.55, 0.15, n).clip(0.2, 0.95)

    # Nonlinear ground-truth function
    base = (
        0.28 * infrastructure_quality
        + 0.22 * broadband_penetration
        + 0.18 * rnd_intensity / 0.08
        + 0.16 * sme_digital_literacy
        + 0.12 * regulatory_environment
        + 0.18 * human_capital_index
        + 0.12 * cybersecurity_readiness
        - 0.25 * cloud_cost_index
        + 0.10 * energy_reliability
        + 0.05 * manufacturing_share
        + 0.07 * urbanization_rate
    )

    interactions = (
        0.10 * infrastructure_quality * broadband_penetration
        + 0.06 * human_capital_index * sme_digital_literacy
        - 0.08 * cloud_cost_index * energy_reliability
    )

    noise = np.random.default_rng(seed + 1).normal(0, 0.04, n)

    adoption = (base + interactions + noise).clip(0, 1)

    df = pd.DataFrame(
        {
            "infrastructure_quality": infrastructure_quality,
            "broadband_penetration": broadband_penetration,
            "rnd_intensity": rnd_intensity,
            "sme_digital_literacy": sme_digital_literacy,
            "regulatory_environment": regulatory_environment,
            "human_capital_index": human_capital_index,
            "cybersecurity_readiness": cybersecurity_readiness,
            "cloud_cost_index": cloud_cost_index,
            "energy_reliability": energy_reliability,
            "manufacturing_share": manufacturing_share,
            "urbanization_rate": urbanization_rate,
            "cloud_adoption_rate": adoption,
        }
    )
    return df


def plot_correlation(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="viridis", annot=False, square=True)
    plt.title("Correlation Heatmap: Cloud Adoption Drivers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, out_path: Path):
    features = [c for c in df.columns if c != "cloud_adoption_rate"]
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()
    for i, f in enumerate(features):
        sns.histplot(df[f], kde=True, ax=axes[i], color="#2b8cbe")
        axes[i].set_title(f)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def fit_linear_regression(df: pd.DataFrame):
    X = df.drop(columns=["cloud_adoption_rate"])  # sklearn
    y = df["cloud_adoption_rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }

    # Statsmodels for coefficients with p-values
    X_const = sm.add_constant(X)
    sm_model = sm.OLS(y, X_const).fit()

    return lr, metrics, sm_model


def plot_linear_coefficients(lr: LinearRegression, cols: list[str], out_path: Path):
    coef = pd.Series(lr.coef_, index=cols).sort_values()
    plt.figure(figsize=(8, 6))
    coef.plot(kind="barh", color=["#e34a33" if v < 0 else "#31a354" for v in coef.values])
    plt.title("Linear Regression Coefficients (Directionality)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def fit_random_forest(df: pd.DataFrame, seed: int = 7):
    X = df.drop(columns=["cloud_adoption_rate"])  # sklearn
    y = df["cloud_adoption_rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    rf = RandomForestRegressor(
        n_estimators=350,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    return rf, metrics, importance


def plot_importance(importance: pd.Series, out_path: Path):
    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="#3182bd")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_report(paths: dict, lin_metrics: dict, rf_metrics: dict, sm_summary: str):
    report_path = paths["out"] / "report.md"
    lines = []
    lines.append("# Cloud Adoption Drivers Analysis (SDG 9)\n\n")
    lines.append("This report summarizes exploratory data analysis and simple models on a synthetic dataset of cloud adoption drivers aligned with SDG 9.\n\n")
    lines.append("## Key Outputs\n\n")
    lines.append("- Correlation Heatmap: ![Correlation](correlation_heatmap.png)\n")
    lines.append("- Feature Distributions: ![Distributions](feature_distributions.png)\n")
    lines.append("- Linear Coefficients: ![LR Coefficients](linear_coefficients.png)\n")
    lines.append("- Random Forest Importance: ![RF Importance](rf_feature_importance.png)\n\n")

    lines.append("## Metrics\n\n")
    lines.append(f"- Linear Regression R2: {lin_metrics['r2']:.3f}, MAE: {lin_metrics['mae']:.3f}\n")
    lines.append(f"- Random Forest R2: {rf_metrics['r2']:.3f}, MAE: {rf_metrics['mae']:.3f}\n\n")

    lines.append("## Linear Model Summary (Statsmodels)\n\n")
    lines.append("```\n")
    lines.append(sm_summary)
    lines.append("```\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    base = Path(__file__).resolve().parents[1]
    paths = ensure_dirs(base)

    df = generate_synthetic_data(n=500, seed=73)
    csv_path = paths["out"] / "cloud_adoption_synthetic.csv"
    df.to_csv(csv_path, index=False)

    # EDA plots
    plot_correlation(df, paths["out"] / "correlation_heatmap.png")
    plot_feature_distributions(df, paths["out"] / "feature_distributions.png")

    # Linear Regression
    lr, lr_metrics, sm_model = fit_linear_regression(df)
    plot_linear_coefficients(lr, [c for c in df.columns if c != "cloud_adoption_rate"], paths["out"] / "linear_coefficients.png")

    # Random Forest
    rf, rf_metrics, importance = fit_random_forest(df)
    plot_importance(importance, paths["out"] / "rf_feature_importance.png")

    # Report
    write_report(paths, lr_metrics, rf_metrics, sm_model.summary().as_text())

    # Save metrics JSON for quick reference
    with open(paths["out"] / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"linear_regression": lr_metrics, "random_forest": rf_metrics}, f, indent=2)

    print("Done. Outputs written to:", paths["out"].as_posix())


if __name__ == "__main__":
    main()
