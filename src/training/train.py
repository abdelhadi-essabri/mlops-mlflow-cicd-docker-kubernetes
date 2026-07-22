"""
Entraînement baseline : XGBoost avec encoding minimal,
tracké dans MLflow.
"""

import json
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/clean.csv")
TARGET_COL = "isFraud"
ID_COL = "TransactionID"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection-baseline")


def load_data() -> pd.DataFrame:
    logger.info(f"Chargement de {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Shape : {df.shape}")
    return df


def simple_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding minimal : catégorielles -> codes numériques."""
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    logger.info(f"Colonnes catégorielles à encoder : {len(cat_cols)}")

    for col in cat_cols:
        # Category codes : NaN -> -1, chaque catégorie -> un entier
        df[col] = df[col].astype("category").cat.codes

    return df


def plot_confusion_matrix(y_true, y_pred, output_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    df = load_data()
    df = simple_encoding(df)

    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train : {X_train.shape}, Test : {X_test.shape}")

    # Ratio pour gérer le déséquilibre de classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight : {scale_pos_weight:.2f}")

    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="baseline_xgboost"):
        mlflow.log_params(params)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])

        logger.info("Entraînement du modèle...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        logger.info("Évaluation...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_proba),
            "auc_pr": average_precision_score(y_test, y_proba),
        }

        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
            mlflow.log_metric(name, value)

        # Matrice de confusion en artefact
        cm_path = "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        # Feature importance (top 20)
        importance = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(20)
        importance.to_csv("feature_importance.csv")
        mlflow.log_artifact("feature_importance.csv")

        # Modèle
        mlflow.xgboost.log_model(model, "model")

        # Sauvegarde des métriques pour DVC
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Run terminé : {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()