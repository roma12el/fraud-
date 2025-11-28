import streamlit as st
import pandas as pd
import joblib
from utils.preprocess import load_data, basic_preprocess
import numpy as np

st.header('3 — Risk Assessment')

df = load_data('data/dataset.csv')
st.write('Select a contract to assess:')
contract = st.selectbox('Contract', df['contract_id'].tolist())

client = df[df['contract_id']==contract].iloc[0]
st.write('Client profile:')
st.json(client.to_dict())

# Load model
models = [f for f in ['models/trained_model.pkl','models/model_rf.pkl','models/model_dt.pkl','models/model_svc.pkl'] if os.path.exists(f)]
# But we can't use os directly; we'll try specific files
available = []
import os
for f in ['models/trained_model.pkl','models/model_rf.pkl','models/model_dt.pkl','models/model_svc.pkl']:
    if os.path.exists(f):
        available.append(f)

if not available:
    st.warning('No trained models found. Go to Model Training page and train a model (or run models/train_models.py).')
else:
    sel = st.selectbox('Select model file', available)
    if st.button('Estimate risk'):
        model = joblib.load(sel)
        dfp = basic_preprocess(df, drop_cols=['contract_id'])
        X = dfp[dfp['contract_id_'+client['contract_id']] if 'contract_id_'+client['contract_id'] in dfp.columns else dfp.columns].copy() if False else None
        # Simpler: compute features for this client
        client_df = pd.DataFrame([client]).reset_index(drop=True)
        client_proc = basic_preprocess(client_df, drop_cols=['contract_id'])
        # ensure columns match training
        model_cols = model.named_steps['scaler'] if 'scaler' in model.named_steps else None
        Xc = client_proc.reindex(columns=model.feature_names_in_, fill_value=0) if hasattr(model, 'feature_names_in_') else client_proc
        proba = None
        try:
            proba = model.predict_proba(Xc)[:,1][0]
        except:
            try:
                s = model.decision_function(Xc)
                proba = 1/(1+np.exp(-s))[0]
            except Exception as e:
                st.error('Unable to compute probability: ' + str(e))
        if proba is not None:
            st.metric('Estimated probability of high risk', f'{proba:.3f}')
