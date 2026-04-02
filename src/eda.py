import os
import logging
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Paths (use project-root relative `data` and `outputs`)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "src/data", "VisaFile.csv")
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "outputs"))

os.makedirs(OUTPUT_DIR, exist_ok=True)


def cap_series(series: pd.Series) -> pd.Series:
    """Cap outliers using the 1.5*IQR rule."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logging.error("Data file not found: %s", path)
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="latin1", low_memory=False)


def load_and_clean(path: str) -> pd.DataFrame:
    logging.info("Loading data from %s", path)
    df = safe_read_csv(path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop rows with no city if that column exists
    if "work_city" in df.columns:
        df = df.dropna(subset=["work_city"])

    # Clean boolean column if present
    if "full_time_position_y_n" in df.columns:
        df["full_time_position_y_n"] = (
            df["full_time_position_y_n"].astype(str).str.strip().str.upper().replace({"YES": "Y", "NO": "N", "NAN": np.nan})
        )
        if df["full_time_position_y_n"].isnull().any():
            df["full_time_position_y_n"] = df["full_time_position_y_n"].fillna(df["full_time_position_y_n"].mode().iloc[0])
        df["full_time_position_y_n"] = df["full_time_position_y_n"].map({"Y": 1, "N": 0})

    # Parse dates and compute processing_time_days
    for col in ("case_received_date", "decision_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "case_received_date" in df.columns and "decision_date" in df.columns:
        df = df.dropna(subset=["case_received_date", "decision_date"])  # require both
        df["processing_time_days"] = (df["decision_date"] - df["case_received_date"]).dt.days
        df = df[df["processing_time_days"] >= 0]
    elif "processing_time_days" not in df.columns:
        logging.error("Cannot compute processing_time_days: missing date columns and no processing_time_days present")
        raise RuntimeError("Missing time information to compute processing_time_days")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Cap outliers grouped by visa_status if available
    if "visa_status" in df.columns:
        df["processing_time_days"] = df.groupby("visa_status")["processing_time_days"].transform(cap_series)
    else:
        df["processing_time_days"] = cap_series(df["processing_time_days"])

    # Time features
    if "case_received_date" in df.columns:
        df["year"] = df["case_received_date"].dt.year
        df["month"] = df["case_received_date"].dt.month
        df["quarter"] = df["case_received_date"].dt.quarter
        df["season"] = df["month"].apply(lambda x: "Peak" if x in [1, 2, 12] else "Off-Peak")

    # Aggregations (named to match your dataset expectations)
    if "work_city" in df.columns:
        country_avg = df.groupby("work_city")["processing_time_days"].mean()
        df["country_avg"] = df["work_city"].map(country_avg)

    if "visa_status" in df.columns:
        visa_avg = df.groupby("visa_status")["processing_time_days"].mean()
        df["visa_avg"] = df["visa_status"].map(visa_avg)

    if "work_state" in df.columns:
        state_avg = df.groupby("work_state")["processing_time_days"].mean()
        df["state_avg"] = df["work_state"].map(state_avg)

    # Monthly application volume (Backlog proxy) named `monthly_volume`
    if set(["year", "month"]).issubset(df.columns):
        df["monthly_volume"] = df.groupby(["year", "month"])["processing_time_days"].transform("count")

    return df


def encode_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["visa_class", "visa_status", "work_state", "season", "visa_type", "processing_center"]
    cats = [c for c in candidates if c in df.columns]
    if cats:
        return pd.get_dummies(df, columns=cats, drop_first=True)
    return df.copy()


def save_visuals(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    sns.set_theme(style="whitegrid")

    # 1) Distribution (PNG)
    png1 = os.path.join(output_dir, "eda_processing_time_distribution.png")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["processing_time_days"].dropna(), bins=50, kde=False, color="teal")
    plt.title("Distribution of Visa Processing Time")
    plt.xlabel("processing_time_days")
    plt.tight_layout()
    plt.savefig(png1)
    plt.close()
    logging.info("Saved: %s", png1)

    # 2) Interactive histogram (HTML) using plotly if available
    if _HAS_PLOTLY:
        try:
            html1 = os.path.join(output_dir, "eda_interactive_processing_time.html")
            fig = px.histogram(df, x="processing_time_days", nbins=40, title="Interactive Processing Time Distribution")
            fig.write_html(html1)
            logging.info("Saved: %s", html1)
        except Exception:
            logging.warning("Plotly interactive chart generation failed; skipping interactive output.")

    # We'll generate the following files in output_dir:
    # - correlation_heatmap.png
    # - feature_importance.png (approximate via model or correlation)
    # - missing_values_heatmap.png
    # - monthly_trend.png
    # - numerical_distribution.png
    # - pairplot.png
    # - processing_time_distribution.png
    # - season_vs_processing.png
    # - shap_feature_importance.png (if shap available)
    # - shap_summary.png (if shap available)
    # - visa_status_avg_processing.png
    # - visa_status_boxplot.png
    # - work_city_analysis.png
    # - work_state_analysis.png

    # 1) processing_time_distribution.png
    p_proc = os.path.join(output_dir, "processing_time_distribution.png")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["processing_time_days"].dropna(), bins=50, kde=False, color="teal")
    plt.title("Distribution of Visa Processing Time")
    plt.xlabel("processing_time_days")
    plt.tight_layout()
    plt.savefig(p_proc)
    plt.close()
    logging.info("Saved: %s", p_proc)

    # 2) numerical_distribution.png - grid of histograms for top numeric cols
    num_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
    num_cols = [c for c in num_cols if c != "processing_time_days"]
    num_sample = num_cols[:6]
    if num_sample:
        p_num = os.path.join(output_dir, "numerical_distribution.png")
        n = len(num_sample)
        cols = 2
        rows = (n + 1) // cols
        plt.figure(figsize=(6 * cols, 3 * rows))
        for i, col in enumerate(num_sample, 1):
            plt.subplot(rows, cols, i)
            sns.histplot(df[col].dropna(), bins=40, kde=False)
            plt.title(col)
        plt.tight_layout()
        plt.savefig(p_num)
        plt.close()
        logging.info("Saved: %s", p_num)

    # 3) pairplot.png (sampled to avoid huge output)
    p_pair = os.path.join(output_dir, "pairplot.png")
    try:
        pair_cols = ["processing_time_days"] + num_sample[:4]
        pair_df = df[pair_cols].dropna().sample(n=min(1000, len(df)), random_state=1)
        pp = sns.pairplot(pair_df)
        pp.savefig(p_pair)
        plt.close()
        logging.info("Saved: %s", p_pair)
    except Exception as e:
        logging.warning("Pairplot generation failed: %s", e)

    # 4) correlation_heatmap.png
    if len(num_cols) > 0:
        p_corr = os.path.join(output_dir, "correlation_heatmap.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include=["int64", "float64"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(p_corr)
        plt.close()
        logging.info("Saved: %s", p_corr)

    # 5) monthly_trend.png
    if "month" in df.columns:
        p_month = os.path.join(output_dir, "monthly_trend.png")
        monthly_avg = df.groupby("month")["processing_time_days"].mean()
        plt.figure(figsize=(9, 5))
        monthly_avg.plot(marker="o")
        plt.title("Monthly Trend in Visa Processing Time")
        plt.xlabel("month")
        plt.ylabel("avg_processing_time_days")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(p_month)
        plt.close()
        logging.info("Saved: %s", p_month)

    # 6) season_vs_processing.png
    if "season" in df.columns:
        p_season = os.path.join(output_dir, "season_vs_processing.png")
        plt.figure(figsize=(8, 5))
        sns.barplot(x="season", y="processing_time_days", data=df)
        plt.title("Peak vs Off-Peak Processing Time")
        plt.tight_layout()
        plt.savefig(p_season)
        plt.close()
        logging.info("Saved: %s", p_season)

    # 7) visa_status_avg_processing.png and visa_status_boxplot.png
    if "visa_status" in df.columns:
        p_vs_avg = os.path.join(output_dir, "visa_status_avg_processing.png")
        vs_avg = df.groupby("visa_status")["processing_time_days"].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=vs_avg.index, y=vs_avg.values)
        plt.title("Average Processing Time by Visa Status")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(p_vs_avg)
        plt.close()
        logging.info("Saved: %s", p_vs_avg)

        p_vs_box = os.path.join(output_dir, "visa_status_boxplot.png")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="visa_status", y="processing_time_days", data=df)
        plt.title("Processing Time Distribution by Visa Status")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(p_vs_box)
        plt.close()
        logging.info("Saved: %s", p_vs_box)

    # 8) work_city_analysis.png (top 20 cities by avg processing time)
    if "work_city" in df.columns:
        p_city = os.path.join(output_dir, "work_city_analysis.png")
        city_avg = df.groupby("work_city")["processing_time_days"].mean().sort_values(ascending=False).head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=city_avg.values, y=city_avg.index)
        plt.title("Top 20 Cities by Avg Processing Time")
        plt.xlabel("avg_processing_time_days")
        plt.tight_layout()
        plt.savefig(p_city)
        plt.close()
        logging.info("Saved: %s", p_city)

    # 9) work_state_analysis.png (top 20 states by avg processing time)
    if "work_state" in df.columns:
        p_state = os.path.join(output_dir, "work_state_analysis.png")
        state_avg = df.groupby("work_state")["processing_time_days"].mean().sort_values(ascending=False).head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=state_avg.values, y=state_avg.index)
        plt.title("Top 20 States by Avg Processing Time")
        plt.xlabel("avg_processing_time_days")
        plt.tight_layout()
        plt.savefig(p_state)
        plt.close()
        logging.info("Saved: %s", p_state)

    # 10) missing_values_heatmap.png
    p_missing = os.path.join(output_dir, "missing_values_heatmap.png")
    try:
        mv = df.isnull().astype(int)
        plt.figure(figsize=(12, max(4, mv.shape[1] / 2)))
        sns.heatmap(mv.T, cbar=False)
        plt.title("Missing Values (rows x columns)")
        plt.tight_layout()
        plt.savefig(p_missing)
        plt.close()
        logging.info("Saved: %s", p_missing)
    except Exception as e:
        logging.warning("Failed to create missing values heatmap: %s", e)

    # 11) feature_importance.png and SHAP plots: attempt model-based importance
    p_fi = os.path.join(output_dir, "feature_importance.png")
    p_shap_imp = os.path.join(output_dir, "shap_feature_importance.png")
    p_shap_sum = os.path.join(output_dir, "shap_summary.png")

    try:
        # Try sklearn RandomForestRegressor for importance on a small sample
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        # Prepare feature matrix: numeric + simple dummies for top categorical
        df_model = df.copy()
        # select candidate features
        features = []
        for c in df_model.select_dtypes(include=["int64", "float64"]).columns:
            if c != "processing_time_days":
                features.append(c)
        # add a few categorical dummies if present
        for c in ["visa_status", "visa_type", "work_state"]:
            if c in df_model.columns:
                dummies = pd.get_dummies(df_model[c], prefix=c, drop_first=True)
                df_model = pd.concat([df_model, dummies], axis=1)
                features += list(dummies.columns)

        features = [f for f in features if f in df_model.columns]
        if features:
            X = df_model[features].fillna(0).sample(n=min(5000, len(df_model)), random_state=1)
            y = df_model.loc[X.index, "processing_time_days"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            rf = RandomForestRegressor(n_estimators=50, random_state=1, n_jobs=1)
            rf.fit(X_train, y_train)
            importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(30)
            plt.figure(figsize=(8, max(6, len(importances) * 0.25)))
            sns.barplot(x=importances.values, y=importances.index)
            plt.title("Feature importance (RandomForest)")
            plt.tight_layout()
            plt.savefig(p_fi)
            plt.close()
            logging.info("Saved: %s", p_fi)

            # Attempt SHAP if available
            try:
                import shap
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X_train)
                # summary plot
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values, X_train, show=False)
                plt.tight_layout()
                plt.savefig(p_shap_sum)
                plt.close()
                logging.info("Saved: %s", p_shap_sum)

                # feature importance (bar)
                plt.figure(figsize=(8, max(6, len(importances) * 0.25)))
                shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(p_shap_imp)
                plt.close()
                logging.info("Saved: %s", p_shap_imp)
            except Exception as e:
                logging.warning("SHAP not available or failed: %s", e)
    except Exception as e:
        logging.warning("Model-based feature importance skipped: %s", e)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 1:
        png2 = os.path.join(output_dir, "eda_correlation_heatmap.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(png2)
        plt.close()
        logging.info("Saved: %s", png2)

    # 4) Monthly trend (PNG)
    if "month" in df.columns:
        monthly_avg = df.groupby("month")["processing_time_days"].mean()
        png3 = os.path.join(output_dir, "eda_monthly_trend.png")
        plt.figure(figsize=(9, 5))
        monthly_avg.plot(marker="o")
        plt.title("Monthly Trend in Visa Processing Time")
        plt.xlabel("month")
        plt.ylabel("avg_processing_time_days")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png3)
        plt.close()
        logging.info("Saved: %s", png3)

    # 5) Season comparison (PNG)
    if "season" in df.columns:
        png4 = os.path.join(output_dir, "eda_season_vs_processing.png")
        plt.figure(figsize=(8, 5))
        sns.barplot(x="season", y="processing_time_days", data=df)
        plt.title("Peak vs Off-Peak Processing Time")
        plt.tight_layout()
        plt.savefig(png4)
        plt.close()
        logging.info("Saved: %s", png4)


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    df_encoded = encode_for_modeling(df)

    logging.info("Shape before encoding: %s | after encoding: %s", df.shape, df_encoded.shape)

    try:
        save_visuals(df, OUTPUT_DIR)
    except Exception as exc:
        logging.exception("Failed while saving visuals: %s", exc)

    logging.info("Remaining missing values (total): %s", int(df.isnull().sum().sum()))

    return df, df_encoded


if __name__ == "__main__":
    df_out, df_enc = main()
    logging.info("EDA complete. Outputs written to %s", OUTPUT_DIR)
