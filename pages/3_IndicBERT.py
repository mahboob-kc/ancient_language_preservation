import streamlit as st
from indicbert_predictor import predict_masked_words #custom module

st.title("🤖 IndicBERT Masked Word Prediction")
st.write("Enter a sentence with '[MASK]' and get predictions.")

#load masked text from previous page
masked_text = st.session_state.get("masked_text", "")

if not masked_text: #if no masked word it show an warning and stop execution
    st.warning("⚠️ No masked text found. Please go back and enter text with [MASK].")
    st.stop()

#input for editing
masked_text = st.text_area("Enter text with [MASK]:", value=masked_text, height=150)

#session var to store the result for storing the result(prediction)
if "masked_predictions" not in st.session_state:
    st.session_state.masked_predictions = None

#button to trigger prediction
if st.button("Predict Missing Words"):
    if "[MASK]" not in masked_text:
        st.warning("⚠️ Please enter a sentence with at least one [MASK] token.")
    else:
        with st.spinner("Predicting..."):
            predictions = predict_masked_words(masked_text)

        st.subheader("🔍 Predicted Words for Masked Text:")
        for mask_label, options in predictions.items():
            st.write(f"**{mask_label} Suggestions:**")
            for word, prob in options:
                st.write(f" - **{word}** ({prob:.2%})")  #show probability in percentage
        
        # Store predictions in session state for the next page
        st.session_state.masked_predictions = predictions

# Navigation button to 4_Translation.py (only if predictions exist)
if st.session_state.get("masked_predictions"):
    if st.button("Next ➝"):
        st.switch_page("pages/4_Translation.py")  # Ensure this path is correct!
