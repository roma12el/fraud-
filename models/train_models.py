import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib
from pathlib import Path
from utils.preprocess import load_data, basic_preprocess

Path('models').mkdir(parents=True, exist_ok=True)

df = load_data('data/dataset.csv')
dfp = basic_preprocess(df, drop_cols=['contract_id'])

X = dfp.drop(columns=['target'])
y = dfp['target']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X,y)
joblib.dump(rf, 'models/model_rf.pkl')

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X,y)
joblib.dump(dt, 'models/model_dt.pkl')

svc = SVC(probability=True, gamma='scale', random_state=42)
svc.fit(X,y)
joblib.dump(svc, 'models/model_svc.pkl')

print('Saved models to models/*.pkl')
