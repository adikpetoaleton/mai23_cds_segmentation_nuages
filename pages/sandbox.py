import streamlit as st

st.set_page_config(layout='centered')




# Session is initialized
if 'data_session' not in st.session_state:
    st.session_state.data_session = None
    data_script = None
else:
    data_script = st.session_state.data_session

# Something happen
if st.button('True'):
    if st.session_state.data_session != True:
        st.session_state.data_session = True
        data_script = True

if st.button('False'):
    if st.session_state.data_session != False:
        st.session_state.data_session = False
        data_script = False

# Use variable
# st.markdown("#### Session is changed")
# st.session_state
st.write("data_script:", data_script)
