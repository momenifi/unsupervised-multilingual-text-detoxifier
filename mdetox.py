from datasets import load_dataset
from collections import Counter
import numpy as np
import nltk
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import difflib
from langdetect import detect

from nltk.tokenize import word_tokenize

# Force use of old punkt only
import nltk.tokenize
if hasattr(nltk.tokenize, "_get_punkt_tokenizer"):
    def _get_punkt_tokenizer(language):
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        return PunktSentenceTokenizer()
    nltk.tokenize._get_punkt_tokenizer = _get_punkt_tokenizer


# Load the multilingual toxic lexicon dataset
multilingual_toxic_lexicon = load_dataset("textdetox/multilingual_toxic_lexicon")
hi = list(multilingual_toxic_lexicon['hi'])

# Load the multilingual parade-tox dataset
dev = load_dataset("textdetox/multilingual_paradetox")
hindi_sentence = list(dev['hi'])

# Download necessary resources for NLTK (run this only once)
nltk.download('punkt')

# Extract toxic and non-toxic sentences
toxic_sentences = [entry['toxic_sentence'] for entry in hindi_sentence]
non_toxic_sentences = [entry['neutral_sentence'] for entry in hindi_sentence]

# Tokenize the sentences
toxic_tokens = [word for sentence in toxic_sentences for word in nltk.word_tokenize(sentence)]
non_toxic_tokens = [word for sentence in non_toxic_sentences for word in nltk.word_tokenize(sentence)]

# Count the tokens
toxic_token_counts = Counter(toxic_tokens)
non_toxic_token_counts = Counter(non_toxic_tokens)

# Compute total number of tokens
total_toxic_tokens = sum(toxic_token_counts.values())
total_non_toxic_tokens = sum(non_toxic_token_counts.values())

# Calculate log-odds ratio for each token and filter tokens based on criteria
log_odds_ratios = {}
for token in set(toxic_tokens + non_toxic_tokens):
    toxic_freq = toxic_token_counts.get(token, 0)
    non_toxic_freq = non_toxic_token_counts.get(token, 0)
    log_odds_ratio = np.log((toxic_freq + 1) / (non_toxic_freq + 1))
    # Filter tokens with log-odds ratio above 0.5 and containing at least 3 letters
    if log_odds_ratio > 0.5 and len(token) > 3:
        log_odds_ratios[token] = log_odds_ratio

# Sort tokens by log-odds ratio
sorted_log_odds_ratios = sorted(log_odds_ratios.items(), key=lambda x: x[1], reverse=True)

# Store selected tokens in a list
selected_tokens = [token for token, log_odds_ratio in sorted_log_odds_ratios]

# Load model and tokenizer directly
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

# Initialize the unmasker pipeline
unmasker = pipeline('fill-mask', model='xlm-roberta-large', top_k=1)

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
        for word_obj in hi:
            word = word_obj['text']
         
            # Compare similarity between the word and each word in the sentence
            for sentence_word in sentence.split():
                similarity = difflib.SequenceMatcher(None, word, sentence_word).ratio()
                if similarity >= threshold:
                    masked_sentence = masked_sentence.replace(sentence_word, '<mask>').replace('<mask> <mask>', '<mask>')
                    masked_sentence = masked_sentence.strip('<mask>')
        
        # Compare the similarity of each element of selected_tokens with each word in the sentence
        for selected_token in selected_tokens:
            for sentence_word in sentence.split():
                similarity = difflib.SequenceMatcher(None, selected_token, sentence_word).ratio()
                if similarity >= threshold:
                    masked_sentence = masked_sentence.replace(sentence_word, '<mask>').replace('<mask> <mask>', '<mask>')
                    masked_sentence = masked_sentence.strip('<mask>')
                    
        # Remove <mask> from the beginning or end of the sentence
        #masked_sentence = masked_sentence.strip('<mask>')
        try:
            umasked=unmasker(masked_sentence)[0]['sequence']
        except:
            umasked= masked_sentence
            
        # Remove <mask> from the beginning or end of the sentence
        #umasked = umasked.strip('<mask>')

            
        masked_sentences.append({'toxic_sentence': sentence, 'masked_sentence': umasked, 'language': lang})

    return masked_sentences
	
	
# Mask similar words in sentences
masked_sentences = mask_similar_words(hi, hindi_sentence,selected_tokens)

