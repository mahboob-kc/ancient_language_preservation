import streamlit as st #ui
import io
from PIL import Image #manipulate the uploaded image
from imageToText import get_text_from_image  # OCR function

st.title("📜 Sanskrit OCR & Translation")
st.write("Upload an image and extract Sanskrit text.")

if "ocr_text" not in st.session_state:  #initialize session state for OCR text
    st.session_state.ocr_text = ""

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Call OCR function
    with st.spinner("Extracting text..."): #show a loading spinner when ocr runs
        extracted_text = get_text_from_image(img_byte_arr)

  
    st.session_state.ocr_text = extracted_text   #store extracted text in session state

    #display Extracted Text
    st.subheader("Extracted Sanskrit Text:") 
    st.text_area("Extracted Sanskrit Text:", extracted_text, height=200, label_visibility="collapsed")

#always show Next button, no validation
if st.button("Next ➝"):
    st.switch_page("pages/2_Masking.py")
