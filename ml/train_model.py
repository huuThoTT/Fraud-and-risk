import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
import argparse
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature


def train_fraud_model(
    data_path='../data/synthetic_transactions.csv',
    model_path='fraud_model.pkl',
    metrics_path='../metrics.json',
    plots_dir='../plots',
    experiment_name='fraud-detection',
    mlflow_tracking_uri='http://localhost:5000'
):
    # ── MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"[INFO] Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Feature Engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    categorical_cols = ['device_type', 'location', 'payment_method']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    exclude_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud']
    features = [col for col in df_encoded.columns if col not in exclude_cols]

    X = df_encoded[features]
    y = df_encoded['is_fraud']

    print(f"[INFO] Dataset shape: {X.shape} | Fraud ratio: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (len(y) - y.sum()) / y.sum()

    # Model hyperparameters (logged to MLflow)
    params = {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.08,
        'scale_pos_weight': float(round(scale_pos_weight, 4)),
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': -1,
    }

    # ── MLflow Run ─────────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n[MLflow] Run ID: {run_id}")
        print(f"[MLflow] Experiment: {experiment_name}")

        # Log dataset info as tags
        mlflow.set_tags({
            'model_type': 'XGBoost',
            'data_source': 'synthetic',
            'task': 'binary_classification',
            'team': 'fraud-detection',
        })

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))
        mlflow.log_param('n_features', len(features))
        mlflow.log_param('fraud_ratio', round(float(y.mean()), 4))

        # Train
        print("\n[INFO] Training XGBoost Classifier...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        auc   = roc_auc_score(y_test, y_prob)

        print(classification_report(y_test, y_pred))
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}")

        # ── Log metrics to MLflow ──────────────────────────────────────────
        mlflow.log_metrics({
            'accuracy':  round(acc, 4),
            'precision': round(prec, 4),
            'recall':    round(rec, 4),
            'f1_score':  round(f1, 4),
            'auc_roc':   round(auc, 4),
        })

        # ── Save metrics.json (for CML report) ────────────────────────────
        metrics = {
            'accuracy':     round(acc, 4),
            'precision':    round(prec, 4),
            'recall':       round(rec, 4),
            'f1_score':     round(f1, 4),
            'auc_roc':      round(auc, 4),
            'train_samples': int(len(X_train)),
            'test_samples':  int(len(X_test)),
            'fraud_ratio':   round(float(y.mean()), 4),
            'mlflow_run_id': run_id,
        }
        os.makedirs(os.path.dirname(metrics_path) if os.path.dirname(metrics_path) else '.', exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path, artifact_path='metrics')
        print(f"\n[INFO] Metrics saved → {metrics_path}")

        # ── Confusion Matrix plot ──────────────────────────────────────────
        os.makedirs(plots_dir, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'], ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix\nF1={f1:.3f}  AUC={auc:.3f}', fontsize=13)
        plt.tight_layout()
        cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=120)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path='plots')

        # ── Feature Importance plot ────────────────────────────────────────
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fi.sort_values().plot(kind='barh', ax=ax2, color='steelblue')
        ax2.set_title('Top 15 Feature Importances', fontsize=13)
        ax2.set_xlabel('Importance Score')
        plt.tight_layout()
        fi_path = os.path.join(plots_dir, 'feature_importance.png')
        plt.savefig(fi_path, dpi=120)
        plt.close()
        mlflow.log_artifact(fi_path, artifact_path='plots')

        # ── Log model to MLflow Model Registry ────────────────────────────
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path='xgboost_fraud_model',
            signature=signature,
            input_example=X_train.iloc[:3],
            registered_model_name='fraud-detection-xgboost',  # Model Registry
        )
        print(f"[MLflow] Model URI: {model_info.model_uri}")

        # ── Also save .pkl for backward compat (Kafka pipeline) ───────────
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'features': features}, f)
        mlflow.log_artifact(model_path, artifact_path='pkl_model')
        print(f"[INFO] Model pkl saved → {model_path}")

        print(f"\n[MLflow] ✅ Run complete: {mlflow_tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run_id}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Fraud Detection Model with MLflow')
    parser.add_argument('--data-path',    default='../data/synthetic_transactions.csv')
    parser.add_argument('--model-path',   default='ml/fraud_model.pkl')
    parser.add_argument('--metrics-path', default='metrics.json')
    parser.add_argument('--plots-dir',    default='plots')
    parser.add_argument('--experiment',   default='fraud-detection',
                        help='MLflow experiment name')
    parser.add_argument('--mlflow-uri',   default=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
                        help='MLflow tracking server URI')
    args = parser.parse_args()

    train_fraud_model(
        data_path=args.data_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        plots_dir=args.plots_dir,
        experiment_name=args.experiment,
        mlflow_tracking_uri=args.mlflow_uri,
    )
