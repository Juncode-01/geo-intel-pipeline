"""
Classifier Agent
----------------
Trains a Random Forest classifier on labelled BC Catalogue metadata
and uses it to predict relevance scores for new unseen datasets.

Workflow:
1. Load labelled CSV
2. Build text representation of each dataset
3. Generate embeddings using sentence-transformers
4. Train Random Forest on embeddings + labels
5. Evaluate performance
6. Save trained model for reuse
"""

import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
from config import LABELLED_DATA_PATH, MODEL_SAVE_PATH


class ClassifierAgent:

    def __init__(self):
        # Sentence transformer for generating embeddings
        # This model runs locally, no API key needed
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded.\n")

        self.classifier = None
        self.is_trained = False

    # ----------------------------------------------------------
    # TEXT PREPARATION
    # ----------------------------------------------------------

    def build_text_representation(self, row):
        """
        Combine relevant metadata fields into a single string
        for embedding. More fields = richer signal for the model.
        """
        parts = []

        if pd.notna(row.get("title", "")):
            parts.append(f"Title: {row['title']}")

        if pd.notna(row.get("notes", "")):
            # Truncate long descriptions
            notes = str(row["notes"])[:400]
            parts.append(f"Description: {notes}")

        if pd.notna(row.get("tags", "")):
            parts.append(f"Tags: {row['tags']}")

        if pd.notna(row.get("organization", "")):
            parts.append(f"Organization: {row['organization']}")

        if pd.notna(row.get("formats", "")):
            parts.append(f"Formats: {row['formats']}")

        return " | ".join(parts)

    def generate_embeddings(self, texts):
        """
        Convert a list of text strings into embedding vectors.
        Returns a numpy array of shape (n_samples, embedding_dim).
        """
        print(f"Generating embeddings for {len(texts)} records...")
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        print(f"Embeddings shape: {embeddings.shape}\n")
        return embeddings

    # ----------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------

    def train(self, label_column="relevance"):
        """
        Full training workflow:
        1. Load labelled data
        2. Build text representations
        3. Generate embeddings
        4. Train/evaluate Random Forest
        5. Save model
        """
        # --- Load labelled data ---
        print(f"Loading labelled data from {LABELLED_DATA_PATH}...")
        df = pd.read_csv(LABELLED_DATA_PATH)

        # Drop rows where label is missing
        df = df.dropna(subset=[label_column])
        df[label_column] = df[label_column].astype(int)

        print(f"Loaded {len(df)} labelled records")
        print(f"Label distribution:")
        print(df[label_column].value_counts().to_string())
        print()

        # --- Build text and embed ---
        texts = df.apply(
            self.build_text_representation, axis=1
        ).tolist()

        embeddings = self.generate_embeddings(texts)
        labels = df[label_column].values

        # --- Train/test split ---
        # Stratify ensures both classes appear in train and test
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples:  {len(X_test)}\n")

        # --- Train Random Forest ---
        print("Training Random Forest...")
        self.classifier = RandomForestClassifier(
            n_estimators=200,      # number of trees
            max_depth=None,        # let trees grow fully
            min_samples_split=2,
            class_weight="balanced",  # handles unequal 1/0 counts
            random_state=42,
            n_jobs=-1              # use all CPU cores
        )

        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.\n")

        # --- Evaluate ---
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test)[:, 1]

        print("=" * 50)
        print("CLASSIFIER PERFORMANCE")
        print("=" * 50)
        print(classification_report(
            y_test, y_pred,
            target_names=["Not Relevant", "Relevant"]
        ))

        print("Confusion Matrix:")
        print("(rows=actual, cols=predicted)")
        cm = confusion_matrix(y_test, y_pred)
        print(f"               Not Relevant  Relevant")
        print(f"  Not Relevant      {cm[0][0]:<10}  {cm[0][1]}")
        print(f"  Relevant          {cm[1][0]:<10}  {cm[1][1]}")

        f1 = f1_score(y_test, y_pred)
        print(f"\nF1 Score: {f1:.4f}")
        print("=" * 50)

        # --- Save model ---
        self._save_model()

        return f1

    # ----------------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------------

    def predict(self, df_new):
        """
        Score a dataframe of new unseen datasets.
        Adds two columns:
          - predicted_relevant: 1 or 0
          - relevance_score: probability 0.0 to 1.0
        Returns the dataframe sorted by relevance score descending.
        """
        if not self.is_trained and not self._load_model():
            raise RuntimeError(
                "No trained model found. Run train() first."
            )

        texts = df_new.apply(
            self.build_text_representation, axis=1
        ).tolist()

        embeddings = self.generate_embeddings(texts)

        predictions = self.classifier.predict(embeddings)
        probabilities = self.classifier.predict_proba(embeddings)[:, 1]

        df_scored = df_new.copy()
        df_scored["predicted_relevant"] = predictions
        df_scored["relevance_score"] = probabilities.round(4)

        df_scored = df_scored.sort_values(
            "relevance_score", ascending=False
        ).reset_index(drop=True)

        relevant_count = predictions.sum()
        print(f"\nClassification results:")
        print(f"  Total datasets scored:  {len(df_new)}")
        print(f"  Predicted relevant:     {relevant_count}")
        print(f"  Predicted not relevant: {len(df_new) - relevant_count}")

        return df_scored

    # ----------------------------------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------------------------------

    def show_what_matters(self):
        """
        Random Forests can tell you which embedding dimensions
        drove the most decisions. More useful: we can test which
        keywords in titles tend to appear in high-scoring datasets.
        """
        if not self.is_trained:
            print("Model not trained yet.")
            return

        importances = self.classifier.feature_importances_
        print(f"Top embedding dimensions by importance:")
        top_indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Dimension {idx}: {importances[idx]:.4f}")

    # ----------------------------------------------------------
    # MODEL PERSISTENCE
    # ----------------------------------------------------------

    def _save_model(self):
        """Save trained classifier to disk for reuse."""
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(self.classifier, MODEL_SAVE_PATH)
        print(f"\nModel saved to {MODEL_SAVE_PATH}")

    def _load_model(self):
        """Load a previously trained classifier from disk."""
        if os.path.exists(MODEL_SAVE_PATH):
            self.classifier = joblib.load(MODEL_SAVE_PATH)
            self.is_trained = True
            print(f"Loaded existing model from {MODEL_SAVE_PATH}")
            return True
        return False


# Run directly to train and evaluate
if __name__ == "__main__":
    agent = ClassifierAgent()
    f1 = agent.train()

    if f1 > 0.7:
        print(f"\nModel performance is good (F1={f1:.2f}). Ready for use.")
    else:
        print(f"\nF1={f1:.2f} - consider adding more labelled examples.")
        print("Aim for at least 0.75 before using in production.")