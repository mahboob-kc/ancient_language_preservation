import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load IndicBERT Model & Tokenizer
MODEL_NAME = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

def predict_masked_words(masked_text, top_k=5):
    """
    Predict the top-k words for the masked token(s) in the given text.

    Args:
    - masked_text (str): Input text with [MASK] token.
    - top_k (int): Number of predictions to return per mask.

    Returns:
    - dict: Dictionary mapping each masked token to its top predictions with probabilities.
    """
    inputs = tokenizer(masked_text, return_tensors="pt")
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    if mask_token_index.nelement() == 0:
        raise ValueError("No [MASK] token found in input text.")

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Extract the predictions for the masked token(s)
    mask_token_probs = probabilities[0, mask_token_index, :]
    top_k_probs, top_k_indices = torch.topk(mask_token_probs, top_k, dim=-1)

    predictions = {}
    for i in range(mask_token_index.shape[0]):  # Loop through multiple [MASK] tokens
        predicted_words = [tokenizer.decode([idx.item()]) for idx in top_k_indices[i]]
        predicted_probs = [prob.item() for prob in top_k_probs[i]]
        predictions[f"[MASK] {i+1}"] = list(zip(predicted_words, predicted_probs))

    return predictions  #  Now correctly handles multiple [MASK] tokens
