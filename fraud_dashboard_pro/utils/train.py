import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

def build_and_train(model, X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, 'predict_proba'):
        try:
            y_proba = pipeline.predict_proba(X_test)[:,1]
        except:
            y_proba = None
    elif hasattr(pipeline.named_steps['model'], 'decision_function'):
        scores = pipeline.decision_function(X_test)
        y_proba = 1/(1+np.exp(-scores))
    metrics = {
        'classification_report': classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'roc_auc': roc_auc_score(y_test, y_proba) if (y_proba is not None) else None
    }
    return pipeline, metrics
