import os
from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_data_path = base_dir / "data" / "raw" / "housing.csv"
    processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Paso 1: Carga del conjunto de datos y división en train y test
    total_data = pd.read_csv(raw_data_path)
    features = ["Latitude", "Longitude", "MedInc"]
    data = total_data[features].copy()

    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

    # Paso 2: Construir K-Means con 6 clusters
    model_kmeans = KMeans(n_clusters=6, random_state=42)
    model_kmeans.fit(X_train[features])
    X_train["cluster"] = pd.Series(model_kmeans.labels_, index=X_train.index, dtype="category")

    # Paso 3: Predecir con el conjunto de test
    test_clusters = model_kmeans.predict(X_test[features])
    X_test["cluster"] = pd.Series(test_clusters, index=X_test.index, dtype="category")

    # Guardar conjuntos procesados
    X_train.to_csv(processed_dir / "clean_train.csv", index=False)
    X_test.to_csv(processed_dir / "clean_test.csv", index=False)

    # Paso 4: Entrenar modelo de clasificación supervisada
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train[features], X_train["cluster"])

    y_pred = model_rf.predict(X_test[features])
    acc = accuracy_score(X_test["cluster"], y_pred)
    print(f"Accuracy del modelo supervisado (Random Forest) en Test: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(X_test["cluster"], y_pred))

    # Paso 5: Guardar ambos modelos
    kmeans_file = models_dir / "k-means_default_42.sav"
    rf_file = models_dir / "random_forest_classifier_default_42.sav"

    with open(kmeans_file, "wb") as f:
        dump(model_kmeans, f)
    with open(rf_file, "wb") as f:
        dump(model_rf, f)

    print(f"Modelos guardados en: {models_dir}")


if __name__ == "__main__":
    main()

