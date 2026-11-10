def validate(data, target_col):
    """
    Validation du modèle sur un jeu de données donné.
    """
    logger.info("Validation du modèle...")
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calcul des métriques
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")