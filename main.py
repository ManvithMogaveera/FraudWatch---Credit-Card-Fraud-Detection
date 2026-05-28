
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    classification_report,
    recall_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("Loading dataset...")
data = pd.read_csv("creditcard.csv")

print(data.head())

print("\nCreating stratified split...")

data["Amount_cat"] = pd.cut(
    data["Amount"],
    bins=[-1, 25, 200, 400, 1025, np.inf],
    labels=[1, 2, 3, 4, 5]
)

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_idx, test_idx in splitter.split(data, data["Amount_cat"]):
    train_set = data.loc[train_idx]
    test_set = data.loc[test_idx]

X_train = train_set.drop(["Class", "Amount_cat"], axis=1)
Y_train = train_set["Class"]

X_test = test_set.drop(["Class", "Amount_cat"], axis=1)
Y_test = test_set["Class"]

print("\nBuilding preprocessing pipeline...")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)


print("\nApplying SMOTE oversampling...")

smote = SMOTE(random_state=42)

X_resampled, Y_resampled = smote.fit_resample(
    X_train_transformed,
    Y_train
)

print("Before SMOTE:", Y_train.value_counts())
print("After SMOTE:", pd.Series(Y_resampled).value_counts())

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.9,
    gamma=0.1,
    min_child_weight=5,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_resampled, Y_resampled)


print("\nEvaluating model...")

y_pred = model.predict(X_test_transformed)
y_prob = model.predict_proba(X_test_transformed)[:, 1]

accuracy = accuracy_score(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, y_prob)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(Y_test, y_pred))

cm = confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")

fpr, tpr, thresholds = roc_curve(Y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("plots/roc_curve.png")

feature_importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.head(15).plot(kind='barh')

plt.title("Top 15 Important Features")

plt.tight_layout()
plt.savefig("plots/feature_importance.png")

plt.figure(figsize=(6, 4))

sns.countplot(x=data["Class"])

plt.title("Fraud vs Non-Fraud Distribution")

plt.tight_layout()
plt.savefig("plots/fraud_distribution.png")


joblib.dump(model, MODEL_FILE)
joblib.dump(pipeline, PIPELINE_FILE)

print("\nModel saved successfully!")
print("Plots saved in /plots")
