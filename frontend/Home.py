import streamlit as st
from utils import ping, ping_database

st.set_page_config(
    page_title="Home",
    initial_sidebar_state="collapsed",
)

st.title("Home")

st.markdown("""
    Welcome to Visual Learning Aid for PSO! :wave:
    """)

st.subheader("Status Checks")
st.write(ping())
st.write(ping_database())
