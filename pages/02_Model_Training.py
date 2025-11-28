import streamlit as st
import pandas as pd
import joblib
from utils.preprocess import load_data, basic_preprocess
from utils.train import build_and_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import plotly.express as px

st.header('2 — Model Training & Evaluation (PRO)')

df = load_data('data/dataset.csv')
st.write('Dataset sample:')
st.dataframe(df.head())

# target selection (fixed to 'target' in sample)
target = st.selectbox('Target variable', options=df.columns.tolist(), index=df.columns.get_loc('target'))
features = [c for c in df.columns if c != target and c!='contract_id']

# sampling
sampling = st.selectbox('Sampling method', ['None', 'SMOTE', 'RandomOverSampler', 'RandomUnderSampler'])

# model choice
model_name = st.selectbox('Model', ['RandomForest', 'XGBoost', 'SVM', 'LogisticRegression'])
# hyperparams
if model_name == 'RandomForest':
    n_estimators = st.slider('n_estimators', 10, 500, 100)
    max_depth = st.slider('max_depth', 2, 30, 8)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
elif model_name == 'XGBoost':
    lr = st.slider('learning_rate', 0.01, 0.5, 0.1)
    model = XGBClassifier(learning_rate=lr, use_label_encoder=False, eval_metric='logloss', random_state=42)
elif model_name == 'SVM':
    C = st.slider('C', 0.1, 10.0, 1.0)
    model = SVC(C=C, probability=True, gamma='scale', random_state=42)
else:
    C = st.slider('C (LogReg)', 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C, max_iter=2000)

dfp = basic_preprocess(df, drop_cols=['contract_id'])

X = dfp.drop(columns=[target])
y = dfp[target]

if sampling != 'None':
    if sampling == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif sampling == 'RandomOverSampler':
        sampler = RandomOverSampler(random_state=42)
    else:
        sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    st.write('Resampled dataset size:', X_res.shape[0])
else:
    X_res, y_res = X, y

if st.button('Train & Evaluate'):
    pipeline, metrics = build_and_train(model, X_res, y_res)
    st.write('### Metrics')
    st.json(metrics)
    # save model
    joblib.dump(pipeline, 'models/trained_model.pkl')
    st.success('Model trained and saved as models/trained_model.pkl')

    # show feature importance if available
    try:
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            fi = pipeline.named_steps['model'].feature_importances_
            cols = X_res.columns.tolist()
            df_fi = pd.DataFrame({'feature': cols, 'importance': fi}).sort_values('importance', ascending=False)
            fig = px.bar(df_fi.head(20), x='feature', y='importance', title='Top feature importances')
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.write('Feature importance not available for this model:', e)
