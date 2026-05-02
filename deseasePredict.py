import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(1)

# Your provided code (preprocessing and feature scaling)
path = "C:/Users/Administrator/.cache/kagglehub/datasets/akshaydattatraykhare/diabetes-dataset/versions/1/diabetes.csv"
data = pd.read_csv(path)
data = data.drop_duplicates()

# Replace zeros with appropriate values
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())  # normal distribution
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())  # normal distribution
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].median())  # skewed distribution
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].median())  # skewed distribution
data['BMI'] = data['BMI'].replace(0, data['BMI'].median())  # skewed distribution

# Feature selection
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Scale features
cols = x_train.columns
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=cols)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=cols)

# 3. Train Models
def train_models(x_train_scaled, y_train):
    """Train Gradient Boosting, SVM, and Neural Network models"""
    models = {
        'Gradient Boosting': GradientBoostingClassifier(random_state=1),
        'SVM': SVC(probability=True, random_state=1),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=1)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        trained_models[name] = model
        print(f"{name} trained successfully.")
    
    return trained_models

# 4. Evaluate Models
def evaluate_models(models, x_test_scaled, y_test):
    """Evaluate models using F1 Score and ROC-AUC, and plot ROC curves"""
    results = {}
    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        # Predict and calculate metrics
        y_pred = model.predict(x_test_scaled)
        y_proba = model.predict_proba(x_test_scaled)[:, 1]
        
        # Store results
        results[name] = {
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'model': model
        }
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')
        
        print(f"\n{name} Results:\nF1 Score: {results[name]['f1_score']:.2f}\nROC AUC: {results[name]['roc_auc']:.2f}")
    
    # Plot ROC baseline
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('roc_curves.png')
    plt.close()
    
    return results

# 5. Generate Insights
def generate_insights(feature_names):
    """Provide actionable insights for healthcare professionals"""
    insights = f"""
    Actionable Insights for Diabetes Prevention:
    1. Monitor key risk factors: {', '.join(feature_names[:3])}.
    2. Prioritize regular glucose screening for early detection.
    3. Encourage lifestyle changes to manage BMI and blood pressure.
    4. Focus on high-risk groups, such as older patients or those with a family history of diabetes.
    """
    with open('diabetes_insights.txt', 'w') as f:
        f.write(insights)
    return insights

# Main Execution
if __name__ == '__main__':
    # Train models
    models = train_models(x_train_scaled, y_train)
    
    # Evaluate models
    results = evaluate_models(models, x_test_scaled, y_test)
    
    # Generate insights
    feature_names = cols.tolist()
    insights = generate_insights(feature_names)
    print("\nInsights saved to 'diabetes_insights.txt':\n", insights)