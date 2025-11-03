from datasets import load_dataset, Dataset
from collections import Counter
import numpy as np
import nltk
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import difflib
from langdetect import detect

# ✅ Ensure NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

print("Loading datasets...")

# Load multilingual toxic lexicon
try:
    multilingual_toxic_lexicon = load_dataset("textdetox/multilingual_toxic_lexicon")
    hi = list(multilingual_toxic_lexicon['hi'])
except Exception as e:
    print("⚠️ Could not load multilingual_toxic_lexicon from HuggingFace, using sample lexicon.")
    hi = ["बेवकूफ", "गंदी", "घृणित"]

# Load multilingual parade-tox dataset
try:
    dev = load_dataset("textdetox/multilingual_paradetox")
    hindi_sentence = list(dev['hi'])
except Exception as e:
    print("⚠️ Could not load multilingual_paradetox from HuggingFace, using sample sentences.")
    hindi_sentence = [{"toxic_sentence": "तुम बहुत बेवकूफ हो।", "neutral_sentence": "तुम समझदार हो।"}]

# Extract toxic and non-toxic sentences
toxic_sentences = [entry['toxic_sentence'] for entry in hindi_sentence]
non_toxic_sentences = [entry['neutral_sentence'] for entry in hindi_sentence]

# Tokenize the sentences
toxic_tokens = [word for sentence in toxic_sentences for word in nltk.word_tokenize(sentence)]
non_toxic_tokens = [word for sentence in non_toxic_sentences for word in nltk.word_tokenize(sentence)]

# Count the tokens
toxic_token_counts = Counter(toxic_tokens)
non_toxic_token_counts = Counter(non_toxic_tokens)

# Calculate log-odds ratio for each token and filter tokens
log_odds_ratios = {}
for token in set(toxic_tokens + non_toxic_tokens):
    toxic_freq = toxic_token_counts.get(token, 0)
    non_toxic_freq = non_toxic_token_counts.get(token, 0)
    log_odds_ratio = np.log((toxic_freq + 1) / (non_toxic_freq + 1))
    if log_odds_ratio > 0.5 and len(token) > 3:
        log_odds_ratios[token] = log_odds_ratio

# Sort tokens and store selected tokens
sorted_log_odds_ratios = sorted(log_odds_ratios.items(), key=lambda x: x[1], reverse=True)
selected_tokens = [token for token, _ in sorted_log_odds_ratios]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

# Initialize the unmasker pipeline
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=1)

# Function to compute similarity between two words
def compute_similarity(word1, word2):
    return difflib.SequenceMatcher(None, word1, word2).ratio()

# Function to remove words with 80% similarity from sentences and replace with <mask>
def mask_similar_words(hi, hindi_sentence, selected_tokens, threshold=0.8, max_unmask_attempts=3):
    masked_sentences = []

    for sentence_obj in hindi_sentence:
        sentence = sentence_obj['toxic_sentence']
        lang = detect(sentence)
        
        masked_sentence = sentence

        # ✅ Mask lexicon words safely (string or dict)
        for word_obj in hi:
            if isinstance(word_obj, dict) and 'text' in word_obj:
                word = word_obj['text']
            else:
                word = str(word_obj)

            for sentence_word in sentence.split():
                if difflib.SequenceMatcher(None, word, sentence_word).ratio() >= threshold:
                    masked_sentence = masked_sentence.replace(sentence_word, "<mask>").replace("<mask> <mask>", "<mask>")

        # Mask words from selected_tokens
        for selected_token in selected_tokens:
            for sentence_word in sentence.split():
                if difflib.SequenceMatcher(None, selected_token, sentence_word).ratio() >= threshold:
                    masked_sentence = masked_sentence.replace(sentence_word, "<mask>").replace("<mask> <mask>", "<mask>")

        # Apply unmasking using the pipeline
        try:
            umasked = unmasker(masked_sentence)[0]['sequence']
        except Exception:
            umasked = masked_sentence

        masked_sentences.append({'toxic_sentence': sentence, 'masked_sentence': umasked, 'language': lang})

    return masked_sentences

# ✅ Mask similar words in sentences
masked_sentences = mask_similar_words(hi, hindi_sentence, selected_tokens)
