import pandas as pd
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

disp.plot()
plt.title("Credit Card Fraud Detection")
plt.savefig("confusion_matrix.png")
plt.close()
y_pred = model.predict(X_test)
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Fraud Detection Metrics")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probability of fraud
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC
auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", auc)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Credit Card Fraud Detection - ROC Curve")
plt.legend()
plt.grid()

plt.savefig("roc_curve.png")
plt.close()

# Feature Importance

import pandas as pd
import matplotlib.pyplot as plt

if hasattr(model, "feature_importances_"):

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    importance = importance.sort_values(ascending=False)

    print("\nTop 10 Important Features:")
    print(importance.head(10))

    plt.figure(figsize=(10, 6))
    importance.head(10).plot(kind="bar")

    plt.title("Top 10 Features for Credit Card Fraud Detection")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

else:
    print("Feature importance is not available for this model.")
# Feature Importance

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
importance.head(10).plot(kind="bar")

plt.title("Top 10 Features for Credit Card Fraud Detection")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import joblib
joblib.dump(model, "fraud_model.pkl")

print("Model saved successfully!")

# Test fraud prediction




    # Test fraud prediction

sample = X_test.iloc[0:1]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Prediction: FRAUD")
else:
    print("Prediction: NORMAL")
    print("Model saved successfully!")

# Test fraud prediction

sample = X_test.iloc[0:1]

# =========================================================
# SAVE MODEL
# =========================================================

import joblib

joblib.dump(
    model,
    "D:/Credit_Card_Fraud_Detection/fraud_model.pkl"
)

print("Model saved successfully!")


# =========================================================
# PR-AUC SCORE
# =========================================================

from sklearn.metrics import average_precision_score

y_prob = model.predict_proba(X_test)[:, 1]

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("PR-AUC Score:", pr_auc)