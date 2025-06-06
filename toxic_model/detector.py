import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import numpy as np
import re

# Load the English model and tokenizer
english_model_path = "./improved_toxic_detector_model"
try:
    english_tokenizer = DistilBertTokenizer.from_pretrained(english_model_path)
    english_model = DistilBertForSequenceClassification.from_pretrained(english_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    english_model.to(device)
    english_model.eval()
except Exception as e:
    print(f"Error loading English model or tokenizer: {e}")
    raise e

# Load the Marathi-Hindi model and tokenizer
marathi_hindi_model_path = "./marathi_hindi_toxic_detector_model"
best_model_path = "./Marathi_hindi_best_model.pt"
try:
    marathi_hindi_tokenizer = DistilBertTokenizer.from_pretrained(marathi_hindi_model_path)
    # Initialize model with explicit configuration to match sample.ipynb
    config = DistilBertConfig.from_pretrained(
        marathi_hindi_model_path,
        num_labels=6,
        problem_type="multi_label_classification",
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )
    marathi_hindi_model = DistilBertForSequenceClassification.from_pretrained(
        marathi_hindi_model_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    # Load the best model weights
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model weights not found at {best_model_path}")
    state_dict = torch.load(best_model_path, map_location=device)
    marathi_hindi_model.load_state_dict(state_dict, strict=False)
    marathi_hindi_model.to(device)
    marathi_hindi_model.eval()
    print(f"Successfully loaded Marathi-Hindi model with weights from {best_model_path}")
except Exception as e:
    print(f"Error loading Marathi-Hindi model or tokenizer: {e}")
    raise e

def score_comment(text):
    """
    Analyze the input text for toxicity and identify specific toxic words.
    
    Args:
        text (str): The text to analyze.
    
    Returns:
        dict: A dictionary containing:
            - is_toxic (bool): Whether the text is toxic.
            - toxic_words (list): List of toxic words for toxic chats (empty for non-toxic).
            - scores (dict): Toxicity scores for each category.
    """
    print(f"Processing English comment: {text!r}")
    print(f"Input ASCII: {[ord(c) for c in text]}")
    
    # Fallback: Redirect to Marathi-Hindi if 'murkha' is present
    if 'murkha' in text.lower():
        print("Redirecting to score_marathi_hindi_comment due to 'murkha' presence")
        return score_marathi_hindi_comment(text)
    
    # Check for asterisk-containing words
    has_asterisk = bool(re.search(r'\b[\w*]*\*[\w*]*\b', text))
    print(f"Contains asterisk: {has_asterisk}")
    
    # Tokenize the input text for overall scoring
    inputs = english_tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
    
    # Run inference for overall message
    with torch.no_grad():
        outputs = english_model(**inputs).logits
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Define toxicity categories
    categories = ['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']
    
    # Create scores dictionary
    scores = {cat: float(prob) for cat, prob in zip(categories, probs)}
    
    # Determine if the text is toxic (model threshold 0.5 or contains asterisk)
    is_toxic = has_asterisk or any(prob > 0.5 for prob in probs)
    print(f"Is toxic: {is_toxic}, scores: {scores}")
    
    # Extract toxic words for toxic messages
    toxic_words = []
    if is_toxic:
        # Split the text into words, preserving original case and special characters
        words = re.findall(r'\b[\w*]+\b', text)
        print(f"Extracted words: {words}")
        # Evaluate each word
        for word in words:
            # Add asterisk-containing words as toxic
            if '*' in word:
                toxic_words.append(word)
                print(f"Added asterisk-containing toxic word: {word}")
                continue
            # Score non-asterisk words
            word_inputs = english_tokenizer(word, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                word_outputs = english_model(**word_inputs).logits
            word_probs = torch.sigmoid(word_outputs).cpu().numpy()[0]
            # Consider the word toxic if any category score > 0.5
            if any(word_prob > 0.5 for word_prob in word_probs):
                toxic_words.append(word)
                print(f"Added model-detected toxic word: {word}, scores: {list(word_probs)}")
    
    # Remove duplicates while preserving order
    toxic_words = list(dict.fromkeys(toxic_words))
    print(f"Final toxic words: {toxic_words}")
    
    return {
        "is_toxic": is_toxic,
        "toxic_words": toxic_words,
        "scores": scores
    }

def score_marathi_hindi_comment(text):
    """
    Analyze Marathi-Hindi input text for toxicity and identify specific toxic words using the Marathi-Hindi model.
    
    Args:
        text (str): The text to analyze.
    
    Returns:
        dict: A dictionary containing:
            - is_toxic (bool): Whether the text is toxic.
            - toxic_words (list): List of toxic words for toxic chats (empty for non-toxic).
            - scores (dict): Toxicity scores for each category.
    """
    print(f"Processing Marathi-Hindi comment: {text!r}")
    print(f"Input ASCII: {[ord(c) for c in text]}")
    
    # Normalize input: strip whitespace and normalize spaces
    text = ' '.join(text.strip().split())
    print(f"Normalized text: {text!r}")
    
    # Check for asterisk-containing words
    has_asterisk = bool(re.search(r'\b[\w*]*\*[\w*]*\b', text))
    print(f"Contains asterisk: {has_asterisk}")
    
    # Tokenize the input text for overall scoring
    inputs = marathi_hindi_tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
    print(f"Tokenized input IDs: {inputs['input_ids'].tolist()}")
    print(f"Decoded tokens: {marathi_hindi_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    
    # Run inference for overall message
    with torch.no_grad():
        outputs = marathi_hindi_model(**inputs).logits
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    print(f"Raw logits: {outputs.cpu().numpy()[0].tolist()}")
    
    # Define toxicity categories for Marathi-Hindi model
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create scores dictionary
    scores = {cat: float(prob) for cat, prob in zip(categories, probs)}
    
    # Extract words for toxicity check
    words = re.findall(r'\b[\w\'-]+\b', text.lower())  # More permissive regex
    print(f"Extracted words: {words}")
    
    # Fallback: Check for known toxic words
    known_toxic_words = ['chutiya', 'abe', 'kutta', 'saala', 'bewakoof', 'bakwas', 'murkha']
    contains_known_toxic = any(word.lower() in known_toxic_words for word in words)
    print(f"Known toxic words check: {contains_known_toxic}")
    
    # Determine if the text is toxic (model threshold 0.5, asterisk, or known toxic word)
    is_toxic = has_asterisk or any(prob > 0.5 for prob in probs) or contains_known_toxic
    print(f"Is toxic: {is_toxic}, scores: {scores}")
    
    # Extract toxic words for toxic messages
    toxic_words = []
    if is_toxic:
        # Evaluate each word
        for word in words:
            # Skip common non-toxic words to avoid false positives
            if word in ['tu', 'ahe', 'kya', 'kar', 'raha', 'hai', 'ka', 'se']:
                print(f"Skipped non-toxic word: {word}")
                continue
            # Add asterisk-containing words as toxic
            if '*' in word:
                toxic_words.append(word)
                print(f"Added asterisk-containing toxic word: {word}")
                continue
            # Add known toxic words
            if word.lower() in known_toxic_words:
                toxic_words.append(word)
                print(f"Added known toxic word: {word}")
                continue
            # Score non-asterisk, non-known-toxic words
            word_inputs = marathi_hindi_tokenizer(word, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                word_outputs = marathi_hindi_model(**word_inputs).logits
            word_probs = torch.sigmoid(word_outputs).cpu().numpy()[0]
            # Consider the word toxic if any category score > 0.3
            if any(word_prob > 0.3 for word_prob in word_probs):
                toxic_words.append(word)
                print(f"Added model-detected toxic word: {word}, scores: {list(word_probs)}")
            else:
                print(f"Word not toxic: {word}, scores: {list(word_probs)}")
    
    # Remove duplicates while preserving order
    toxic_words = list(dict.fromkeys(toxic_words))
    print(f"Final toxic words: {toxic_words}")
    
    return {
        "is_toxic": is_toxic,
        "toxic_words": toxic_words,
        "scores": scores
    }