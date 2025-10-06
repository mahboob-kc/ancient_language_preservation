#streamlit config 
import os
import io
from google.cloud import vision

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'insert your api here'

#initialize Vision API client
client = vision.ImageAnnotatorClient()

def get_text_from_image(image_bytes):
    #xtracts text from an image using Google Vision OCR API
    try:
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        if texts:
            return texts[0].description  # Extracted text
        else:
            return "No text detected."

    except Exception as e:
        return f"Error: {str(e)}"
