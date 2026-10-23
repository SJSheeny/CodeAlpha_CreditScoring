""" TASK 1: Credit Scoring Model
Goal: predict whether a person will DEFAULT (seriously fail to repay within 2 years) using their past financial data.
Data: credit_data.csv - a synthetic file with the same columns as the Kaggle "Give Me Some Credit" dataset. Create it with:  python generate_data.py
How to run:   python credit_scoring.py
Everything is saved into the "outputs" folder (charts, tables, trained model).

The script is split into numbered steps. In VS Code or Google Colab you can run each step separately by clicking the "Run Cell" button above it."""

# %% STEP 0: Imports and settings
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, roc_curve,confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42          # makes results repeatable
TARGET = "default"         # 1 = person defaulted (bad), 0 = person repaid (good)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "credit_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def finish_plot(filename):
    """Save the current chart as a PNG (and show it if a screen is available)."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()


# %% STEP 1: Load the data
if not os.path.exists(CSV_PATH):
    raise SystemExit("credit_data.csv was not found. Run 'python generate_data.py' first.")

df = pd.read_csv(CSV_PATH)
# The real Kaggle file has an extra index column called "Unnamed: 0"; drop it if present
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

# Give the columns shorter, easier names
df = df.rename(columns={
    "SeriousDlqin2yrs": "default",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",   # share of credit limit in use
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",              # times 30-59 days late
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",              # times 60-89 days late
    "NumberOfTimes90DaysLate": "late_90",                              # times 90+ days late
    "DebtRatio": "debt_ratio",                                         # monthly debt payments / income
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfDependents": "dependents",
})

print("Rows, columns:", df.shape)
print("\nFirst 5 rows:\n", df.head().to_string())
missing = df.isna().sum()
print("\nMissing values per column:\n", missing[missing > 0])
print("\nShare of defaulters (1) vs non-defaulters (0):\n", df[TARGET].value_counts(normalize=True).round(3))


# %% STEP 2: Clean the data (fix impossible or extreme values)
# .where(condition) keeps a value if the condition is True, otherwise makes it missing (NaN)
for col in ["late_30_59", "late_60_89", "late_90"]:
    df[col] = df[col].where(df[col] <= 20)      # real data uses codes like 96/98 for "unknown"
df["age"] = df["age"].where(df["age"] >= 18)    # nobody under 18 can have a loan
df["revolving_utilization"] = df["revolving_utilization"].clip(upper=2)   # cap extreme outliers
df["debt_ratio"] = df["debt_ratio"].clip(upper=5)


# %% STEP 3: Feature engineering (create smarter columns from the raw ones)
def add_features(data):
    data = data.copy()
    # A missing income can itself be a warning sign, so we record it
    data["income_missing"] = data["monthly_income"].isna().astype(int)
    # Total number of late payments of any length
    data["total_late_payments"] = data[["late_30_59", "late_60_89", "late_90"]].sum(axis=1)
    # Has the person EVER been 90+ days late? (yes/no)
    data["ever_90_days_late"] = (data["late_90"] > 0).astype(int)
    # Debt ratio is "payments / income", so multiplying gives the monthly payment amount
    data["monthly_debt_payment"] = data["debt_ratio"] * data["monthly_income"]
    # Income shared across the household
    data["income_per_person"] = data["monthly_income"] / (data["dependents"] + 1)
    return data.replace([np.inf, -np.inf], np.nan)   # divide-by-zero -> treat as missing


df = add_features(df)
print("\nColumns after feature engineering:", list(df.columns))


# %% STEP 4: Split into training data (to learn) and test data (to grade the model)
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)   # stratify keeps the default ratio equal
print(f"\nTraining rows: {len(X_train)}   Test rows: {len(X_test)}")

# WHY ACCURACY ALONE IS MISLEADING: if only ~7% default, a "model" that says
# "nobody ever defaults" is right 93% of the time but completely useless.
print(f"Useless baseline accuracy ('nobody defaults'): {1 - y_test.mean():.3f}")


# %% STEP 5: Build the three models
def make_pipeline(model, scale):
    """A pipeline = fill gaps -> (optional) scale numbers -> model, always in the same order."""
    steps = [("imputer", SimpleImputer(strategy="median"))]   # fill missing values with the median
    if scale:
        steps.append(("scaler", StandardScaler()))            # put all columns on a similar scale
    steps.append(("model", model))
    return Pipeline(steps)


# class_weight="balanced" tells the model that rare defaulters matter more,
# otherwise it would mostly ignore them.
models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), scale=True),
    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight="balanced",
                               random_state=RANDOM_STATE), scale=False),
    "Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20,
                               class_weight="balanced_subsample", n_jobs=-1,
                               random_state=RANDOM_STATE), scale=False),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, pipe in models.items():
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    print(f"{name:20s} 5-fold CV ROC-AUC: {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")


# %% STEP 6: Grade every model on the unseen test data
# "Positive" here means DEFAULT, so precision/recall are about catching defaulters.
results, probabilities = [], {}
for name, pipe in models.items():
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]      # probability of default
    probabilities[name] = y_prob
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),                          # overall % correct
        "Precision": precision_score(y_test, y_pred, zero_division=0),       # of those flagged, % truly defaulters
        "Recall": recall_score(y_test, y_pred, zero_division=0),             # of all defaulters, % we caught
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),               # balance of precision and recall
        "ROC-AUC": roc_auc_score(y_test, y_prob),                            # ranking quality (0.5 guessing, 1.0 perfect)
    })

results_df = pd.DataFrame(results).set_index("Model").round(3)
print("\n=== TEST SET RESULTS ===\n", results_df)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"))
best_name = results_df["ROC-AUC"].idxmax()
print("\nBest model by ROC-AUC:", best_name)

# The 0.5 cut-off is a choice. Moving it trades precision against recall.
print(f"\nPrecision vs recall for {best_name} at different cut-offs:")
for cutoff in [0.3, 0.5, 0.7]:
    flagged = (probabilities[best_name] >= cutoff).astype(int)
    print(f"  flag as risky if probability >= {cutoff}:  "
          f"precision={precision_score(y_test, flagged, zero_division=0):.3f}  "
          f"recall={recall_score(y_test, flagged, zero_division=0):.3f}")


# %% STEP 7: Charts
# 7a) ROC curves - the closer to the top-left corner, the better
plt.figure(figsize=(6, 5))
for name, prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Credit Scoring")
plt.legend(loc="lower right")
finish_plot("roc_curves.png")

# 7b) Confusion matrix of the best model
best_pipe = models[best_name]
cm = confusion_matrix(y_test, best_pipe.predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=["No default", "Default"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
finish_plot("confusion_matrix.png")

# 7c) Which features matter most? (Random Forest)
rf = models["Random Forest"]
names = rf[:-1].get_feature_names_out()
importance = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values()
importance.plot(kind="barh", figsize=(7, 5), color="teal")
plt.title("Feature Importance - Random Forest")
finish_plot("feature_importance.png")
print("\nTop 5 most important features:\n", importance.sort_values(ascending=False).head(5).round(3))


# %% STEP 8: Save the best model and try it on a few applicants
joblib.dump(best_pipe, os.path.join(OUTPUT_DIR, "best_credit_model.joblib"))

# Pick 2 people who really defaulted and 2 who did not
sample_idx = list(y_test[y_test == 1].sample(2, random_state=RANDOM_STATE).index) + \
             list(y_test[y_test == 0].sample(2, random_state=RANDOM_STATE).index)
for idx in sample_idx:
    prob = best_pipe.predict_proba(X_test.loc[[idx]])[0, 1]
    decision = "HIGH RISK - reject" if prob >= 0.5 else "LOW RISK - approve"
    actual = "defaulted" if y_test.loc[idx] == 1 else "repaid"
    print(f"Applicant #{idx}: chance of default = {prob:.0%} -> {decision} (actual: {actual})")

print("\nDone. Charts, results table and the saved model are in:", OUTPUT_DIR)
