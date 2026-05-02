import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("creditcard.csv")

print("Dataset Loaded Successfully ✅")

# -------------------------------
# 2. Class Distribution BEFORE SMOTE
# -------------------------------
plt.figure()
sns.countplot(x=df["Class"])
plt.title("Class Distribution (Before SMOTE)")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 3. Features & Target
# -------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------------
# 4. Scale Amount
# -------------------------------
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Apply SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n", y_train.value_counts())

# -------------------------------
# 7. Class Distribution AFTER SMOTE
# -------------------------------
plt.figure()
sns.countplot(x=y_train)
plt.title("Class Distribution (After SMOTE)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 8. Train Model (Random Forest)
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully ✅")

# -------------------------------
# 9. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 10. Evaluation
# -------------------------------
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 11. Confusion Matrix Graph
# -------------------------------
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()