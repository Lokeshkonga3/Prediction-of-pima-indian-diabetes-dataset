# Diabetes Prediction Model

## Project Overview
This project builds a **supervised learning model** to predict diabetes using the **Pima Indians Diabetes Dataset**. It applies multiple machine learning techniques, including:
- Logistic Regression
- Decision Trees
- Random Forest

The goal is to analyze and compare the performance of different models while optimizing accuracy through hyperparameter tuning.

## Dataset
- **Dataset Name:** Pima Indians Diabetes Dataset
- **Source:** [Dataset Link](https://github.com/Lokeshkonga3/analyzing-pima-indian-diabetes-dataset-/blob/main/diabetes.csv)
- **Features:** Includes medical attributes like glucose level, BMI, age, insulin levels, etc.

### **Programming Language**
- **Python** 

### **Data Handling & Processing**
- `pandas` - Data manipulation and preprocessing
- `numpy` - Numerical computations

### **Machine Learning**
- `scikit-learn` - Training models, feature scaling, evaluation metrics
  - `LogisticRegression` - Logistic regression classifier
  - `DecisionTreeClassifier` - Decision tree classifier
  - `RandomForestClassifier` - Random forest classifier
  - `train_test_split` - Splitting dataset into training & testing sets
  - `StandardScaler` - Feature scaling for better model performance
  - `accuracy_score`, `confusion_matrix`, `classification_report` - Model evaluation tools

### **Visualization**
- `matplotlib` - Basic plots and graphs
- `seaborn` - Advanced data visualization and statistical plots


## Installation & Usage 
Clone the repository and run the script:

```sh
git clone <your-repo-url>
cd diabetes-prediction
python model.py
pip install pandas numpy scikit-learn matplotlib


