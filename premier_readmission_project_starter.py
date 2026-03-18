
"""
Premier-style hospital readmission project starter
--------------------------------------------------
How to use:
1) Download the UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset.
2) Put diabetic_data.csv in the same folder as this script, or update DATA_PATH.
3) Run: python premier_readmission_project_starter.py

Outputs:
- cleaned_preview.csv
- readmission_by_age.png
- top_diag_readmission.png
- feature_importance_logreg.csv
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("diabetic_data.csv")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace placeholder missing values used by this dataset
    df = df.replace("?", np.nan)

    # Create target: 1 if readmitted within 30 days, else 0
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)

    # Drop IDs and leakage-heavy outcome columns
    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # Remove categories unlikely to help a quick demo
    if "weight" in df.columns:
        missing_rate = df["weight"].isna().mean()
        if missing_rate > 0.5:
            df = df.drop(columns=["weight"])

    if "payer_code" in df.columns:
        missing_rate = df["payer_code"].isna().mean()
        if missing_rate > 0.4:
            df = df.drop(columns=["payer_code"])

    if "medical_specialty" in df.columns:
        top_specs = df["medical_specialty"].fillna("Unknown").value_counts().head(10).index
        df["medical_specialty"] = np.where(
            df["medical_specialty"].isin(top_specs),
            df["medical_specialty"],
            "Other"
        )

    return df


def exploratory_outputs(df: pd.DataFrame) -> None:
    out = df.copy()

    # Readmission rate by age band
    if "age" in out.columns:
        age_plot = (
            out.groupby("age", dropna=False)["readmit_30"]
            .mean()
            .sort_index()
            .reset_index()
        )
        plt.figure(figsize=(10, 5))
        plt.plot(age_plot["age"], age_plot["readmit_30"], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("30-day readmission rate")
        plt.xlabel("Age band")
        plt.title("30-day Readmission Rate by Age Band")
        plt.tight_layout()
        plt.savefig("readmission_by_age.png", dpi=150)
        plt.close()

    # Top diagnosis groupings from diag_1
    if "diag_1" in out.columns:
        diag = out["diag_1"].fillna("Unknown").astype(str).str[:3]
        tmp = pd.DataFrame({"diag_1_short": diag, "readmit_30": out["readmit_30"]})
        top = (
            tmp.groupby("diag_1_short")["readmit_30"]
            .agg(["mean", "count"])
            .query("count >= 200")
            .sort_values("mean", ascending=False)
            .head(10)
            .reset_index()
        )
        plt.figure(figsize=(10, 5))
        plt.bar(top["diag_1_short"], top["mean"])
        plt.ylabel("30-day readmission rate")
        plt.xlabel("Primary diagnosis code (truncated)")
        plt.title("Highest Readmission Diagnosis Groups")
        plt.tight_layout()
        plt.savefig("top_diag_readmission.png", dpi=150)
        plt.close()

    out.head(250).to_csv("cleaned_preview.csv", index=False)


def build_model(df: pd.DataFrame) -> None:
    y = df["readmit_30"]
    X = df.drop(columns=["readmit_30"])

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    import pickle
    pickle.dump(clf, open("model.pkl", "wb"))


    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

    # Export top coefficients
    preprocessor_fit = clf.named_steps["preprocessor"]
    model_fit = clf.named_steps["model"]

    feature_names = preprocessor_fit.get_feature_names_out()
    coefs = pd.DataFrame(
        {"feature": feature_names, "coefficient": model_fit.coef_[0]}
    )
    coefs["abs_coef"] = coefs["coefficient"].abs()
    coefs.sort_values("abs_coef", ascending=False).head(25).to_csv(
        "feature_importance_logreg.csv", index=False
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "diabetic_data.csv not found. Put the dataset CSV in this folder first."
        )

    df = load_data(DATA_PATH)
    df = clean_data(df)
    exploratory_outputs(df)
    build_model(df)

    print("\nSaved files:")
    print("- cleaned_preview.csv")
    print("- readmission_by_age.png")
    print("- top_diag_readmission.png")
    print("- feature_importance_logreg.csv")


if __name__ == "__main__":
    main()


