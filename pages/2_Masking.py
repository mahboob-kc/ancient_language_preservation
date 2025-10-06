import streamlit as st
st.title("✏️ Manual Masking Step")

#display
st.subheader("Enter or Edit Sanskrit Text for Masking:")

#use OCR text if available, else use empty string
default_text = st.session_state.ocr_text if "ocr_text" in st.session_state and st.session_state.ocr_text else ""

#text area for manual masking
masked_text = st.text_area("Manually mask words (e.g., using [MASK]):", default_text, height=200)

st.session_state.masked_text = masked_text #store in session state for next step


if st.button("Next ➝"): #button
    st.switch_page("pages/3_IndicBERT.py")
