import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1️⃣ Load Data
file_path = r"C:\Users\vaibh\Downloads\healthcare_dataset.csv\healthcare_dataset.csv"
df = pd.read_csv(file_path)

# 2️⃣ Select Required Columns
cols = [
    'Age', 'Gender', 'Blood Type', 'Medical Condition',
    'Billing Amount', 'Admission Type', 'Medication', 'Test Results'
]
df = df[cols]

# 3️⃣ Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Clip outliers
q1 = df['Billing Amount'].quantile(0.05)
q99 = df['Billing Amount'].quantile(0.95)
df['Billing Amount'] = df['Billing Amount'].clip(q1, q99)

# 4️⃣ Split X & y
X = df.drop('Test Results', axis=1)
y = df['Test Results']

# 5️⃣ Columns
num_cols = ['Age', 'Billing Amount']
cat_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Medication']

# 6️⃣ Preprocessor (FIXED)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# 7️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8️⃣ Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        class_weight='balanced',
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=250)
}

best_acc = 0
best_model = None
best_model_name = ""

print("\n--- Model Comparison (After Proper Cleaning) ---")

for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_model = pipe
        best_model_name = name

# 9️⃣ Result
print(f"\n🚀 Best Model Selected: {best_model_name} with {best_acc*100:.2f}% accuracy")

# 🔟 Save pipeline
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Training completed successfully | Model saved")
