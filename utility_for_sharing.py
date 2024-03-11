import streamlit as st

@st.cache_data
def load_Initial_data():
    data = pd.read_csv('clouds_0.csv')
    return data

@st.cache_data
def load_cleaned_data():
    data = pd.read_csv('clouds_1.csv')
    return data

@st.cache_data
def load_enriched_data():
    data = pd.read_csv('clouds_2.csv')
    return data

@st.cache_data
def load_hist_data():
    data = pd.read_csv('training_history.csv')
    return data

if 'dataframe_1' not in st.session_state:
    st.session_state.dataframe_1 = None
    dataframe_1 = None
else:
    dataframe_1 = st.session_state.dataframe_1

if 'dataframe_2' not in st.session_state:
    st.session_state.dataframe_2 = None
    dataframe_2 = None
else:
    dataframe_2 = st.session_state.dataframe_2

if 'dataframe_4' not in st.session_state:
    st.session_state.dataframe_4 = None
    dataframe_4 = None
else:
    dataframe_4 = st.session_state.dataframe_4

if 'dataframe_11' not in st.session_state:
    st.session_state.dataframe_11 = None
    dataframe_11 = None
else:
    dataframe_11 = st.session_state.dataframe_11

if 'dataframe_21' not in st.session_state:
    st.session_state.dataframe_21 = None
    dataframe_21 = None
else:
    dataframe_21 = st.session_state.dataframe_21

if 'dataframe_31' not in st.session_state:
    st.session_state.dataframe_31 = None
    dataframe_31 = None
else:
    dataframe_31 = st.session_state.dataframe_31