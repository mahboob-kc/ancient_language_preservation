import streamlit as st
from indicbert_predictor import predict_masked_words
from translation_model import translate_text

st.title("📜 Sanskrit Translation")

# Ensure masked sentence is retrieved from session state
if "masked_text" in st.session_state and st.session_state.masked_text:
    masked_sentence = st.session_state.masked_text
else:
    masked_sentence = st.text_area("Masked Sentence:", value="", height=150)

# Ensure predictions are available
if "masked_predictions" not in st.session_state or not st.session_state.masked_predictions:
    st.warning("⚠️ No predictions found. Please go back and enter text with [MASK].")
    st.stop()

st.subheader("Masked Sentence:")
st.write(masked_sentence)

# Replace all [MASK] words with best predictions
replaced_sentence = masked_sentence
for mask_label, options in st.session_state.masked_predictions.items():
    best_word = options[0][0]  # Pick highest probability word
    replaced_sentence = replaced_sentence.replace("[MASK]", best_word, 1)

st.subheader("🔍 Masked Words Replaced:")
st.write(replaced_sentence)

# Translate the updated sentence
if st.button("Translate to English"):
    translated_text = translate_text(replaced_sentence)
    st.subheader("📖 Translated English Text:")
    st.write(translated_text)
