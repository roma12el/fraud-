import streamlit as st
import pandas as pd
import plotly.express as px
from utils.preprocess import load_data

st.header('1 — Data Exploration')

df = load_data('data/dataset.csv')
st.write('### Sample data', df)

# Filters
region = st.multiselect('Region', options=df['region'].unique(), default=df['region'].unique().tolist())
ptypes = st.multiselect('Policy type', options=df['policy_type'].unique(), default=df['policy_type'].unique().tolist())

df_f = df[df['region'].isin(region) & df['policy_type'].isin(ptypes)]

col1, col2 = st.columns(2)
fig1 = px.histogram(df_f, x='age', nbins=10, title='Age distribution')
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.box(df_f, x='policy_type', y='annual_premium', title='Premium by policy type')
col2.plotly_chart(fig2, use_container_width=True)

st.write('### Claims by region')
fig3 = px.bar(df_f.groupby('region').claim_history.sum().reset_index(), x='region', y='claim_history', title='Total claims by region')
st.plotly_chart(fig3, use_container_width=True)
