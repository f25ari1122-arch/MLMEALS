
import streamlit as st
import sklearn
import pandas as pd
import numpy as np

st.title("SmartBite Diagnostic Test")

st.success("Streamlit is running!")

st.write("Scikit-learn:", sklearn.__version__)
st.write("Pandas:", pd.__version__)
st.write("NumPy:", np.__version__)

st.write("Testing application...")
