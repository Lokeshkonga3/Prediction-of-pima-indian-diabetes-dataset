# Prediction-of-pima-indian-diabetes-dataset
import pandas as pd

# I uploaded dataset in my github page.So I Load dataset from my GitHub url.
url = "https://raw.githubusercontent.com/Lokeshkonga3/analyzing-pima-indian-diabetes-dataset-/main/diabetes.csv" 
df = pd.read_csv(url)
# Display first few rows
print(df.head())
df

# Check for missing values
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = df.drop("Outcome", axis=1) 
y = df["Outcome"]
# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

