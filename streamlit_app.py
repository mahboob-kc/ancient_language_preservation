import streamlit as st

st.title("Sanskrit Text Processing App")
st.write("Use the sidebar to navigate through the steps.")
if st.button("Next ➝"):
    st.switch_page("pages/1_OCR.py")