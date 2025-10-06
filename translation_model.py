from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator
import torch

# Load the user's fine-tuned MarianMT model
MODEL_NAME = r"fine_tuned_itihasa"  
try:
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model_available = True
except Exception as e:
    print(f"Warning: Could not load MarianMT model. Using Google Translate instead. Error: {e}")
    model_available = False

def translate_text(text):
    
   #Translates Sanskrit text to English using the fine-tuned MarianMT model.
   #If the MarianMT model is unavailable, it falls back to Google Translate.

    text = text.strip()
    if not text:
        return "⚠️ No input provided for translation."

    # If MarianMT model is available, use it
    if model_available:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs)

            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error with MarianMT model: {e}. Switching to Google Translate.")

    # If MarianMT fails or is unavailable, use Google Translate
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        return f"⚠️ Translation failed. Error: {e}"

