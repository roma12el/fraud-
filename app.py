import streamlit as st
from pathlib import Path

st.set_page_config(page_title='Insurance Risk Dashboard PRO', layout='wide')
st.title('Insurance Risk Dashboard - PRO')

PAGES = {
    "Exploration": "pages/01_Exploration.py",
    "Model Training": "pages/02_Model_Training.py",
    "Risk Assessment": "pages/03_Risk_Assessment.py",
    "Dashboard": "pages/04_Dashboard.py",
}

choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
page_path = PAGES[choice]
with open(page_path, "r", encoding="utf-8") as f:
    code = f.read()
exec(compile(code, page_path, 'exec'), globals())
