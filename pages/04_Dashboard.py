import streamlit as st
import pandas as pd
import plotly.express as px
from utils.preprocess import load_data

st.header('4 — Aggregated Dashboard')

df = load_data('data/dataset.csv')

col1, col2, col3 = st.columns(3)
col1.metric('Total clients', len(df))
col2.metric('Avg annual premium', f'{df["annual_premium"].mean():.2f}')
col3.metric('Mean target (risk rate)', f'{df["target"].mean():.2f}')

fig = px.histogram(df, x='target', title='Target distribution')
st.plotly_chart(fig, use_container_width=True)

st.write('Top risky clients')
st.write(df.sort_values('target', ascending=False).head(10))
