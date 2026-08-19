"""
Script d'ingestion : charge les données brutes IEEE-CIS,
fait des checks qualité de base, et sort un dataset nettoyé.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Chemins (pathlib gère Windows/Linux automatiquement)
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRANSACTION_PATH = RAW_DIR / "train_transaction.csv"
IDENTITY_PATH = RAW_DIR / "train_identity.csv"
OUTPUT_PATH = PROCESSED_DIR / "clean.csv"


def load_raw_data() -> pd.DataFrame:
    """Charge et joint transaction + identity."""
    logger.info(f"Chargement de {TRANSACTION_PATH}")
    df_trans = pd.read_csv(TRANSACTION_PATH)
    logger.info(f"Transactions chargées : {df_trans.shape}")

    logger.info(f"Chargement de {IDENTITY_PATH}")
    df_identity = pd.read_csv(IDENTITY_PATH)
    logger.info(f"Identity chargées : {df_identity.shape}")

    # Jointure left sur TransactionID (toutes les transactions n'ont pas d'identity)
    df = df_trans.merge(df_identity, on="TransactionID", how="left")
    logger.info(f"Dataset joint : {df.shape}")

    return df


def quality_report(df: pd.DataFrame) -> None:
    """Affiche un rapport qualité basique."""
    logger.info("=== RAPPORT QUALITÉ ===")
    logger.info(f"Shape totale : {df.shape}")

    # Distribution de la target
    fraud_rate = df["isFraud"].mean()
    logger.info(f"Taux de fraude : {fraud_rate:.4%}")
    logger.info(f"Nombre de fraudes : {df['isFraud'].sum()} / {len(df)}")

    # Taux de nulls par colonne (top 10 les plus incomplètes)
    null_rates = df.isnull().mean().sort_values(ascending=False)
    logger.info("Top 10 colonnes avec le plus de valeurs manquantes :")
    for col, rate in null_rates.head(10).items():
        logger.info(f"  {col}: {rate:.2%}")

    # Colonnes totalement vides ou quasi-constantes (candidates à drop plus tard)
    nearly_empty = null_rates[null_rates > 0.9]
    logger.info(f"Colonnes avec >90% de nulls : {len(nearly_empty)}")

    # Types de données
    logger.info(f"Colonnes numériques : {df.select_dtypes(include='number').shape[1]}")
    logger.info(f"Colonnes catégorielles : {df.select_dtypes(include='object').shape[1]}")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage minimal pour le baseline (on affinera au Sprint 2)."""
    initial_cols = df.shape[1]

    # Drop des colonnes avec plus de 90% de nulls (trop peu d'info)
    null_rates = df.isnull().mean()
    cols_to_drop = null_rates[null_rates > 0.9].index.tolist()
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Colonnes supprimées (>90% nulls) : {len(cols_to_drop)} / {initial_cols}")

    return df


def main():
    df = load_raw_data()
    quality_report(df)
    df_clean = basic_cleaning(df)

    logger.info(f"Sauvegarde vers {OUTPUT_PATH}")
    df_clean.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Shape finale sauvegardée : {df_clean.shape}")


if __name__ == "__main__":
    main()