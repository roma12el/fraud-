import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Insurance Risk Dashboard', layout='wide')
st.title('Insurance Risk Dashboard - Minimal Example')

@st.cache_data
def load_data():
    return pd.read_csv('data/sample_insurance_portfolio.csv')

df = load_data()

st.write('### Sample data')
st.dataframe(df)

fig = px.histogram(df, x='risk_score', nbins=10, title='Distribution du risk_score')
st.plotly_chart(fig, use_container_width=True)
