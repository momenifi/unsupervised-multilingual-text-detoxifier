import nltk
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from langdetect import detect
from textblob import TextBlob
from difflib import SequenceMatcher

# ✅ Ensure NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# -----------------------------
# Load small demo dataset (to avoid OOM)
# -----------------------------
print("Loading dataset...")
try:
    dataset = load_dataset("s-nlp/Multilingual-Toxic-Comment-Classification")
except:
    print("⚠️ Could not load dataset from HuggingFace, using local sample instead.")
    data = {"comment_text": [
        "You are so stupid!",
        "What a beautiful day!",
        "I hate everything.",
        "This is a wonderful idea.",
        "You are such a nice person."
    ]}
    dataset = Dataset.from_dict(data)
    dataset = {"train": dataset}

# Use only a small subset to save memory
sentences = dataset["train"]["comment_text"][:5]
print(f"Loaded {len(sentences)} example sentences.")

# -----------------------------
# Tokenization check
# -----------------------------
print("Tokenizing sentences...")
tokens = []
for sentence in sentences:
    try:
        tokens.extend(nltk.word_tokenize(sentence))
    except:
        print(f"Failed tokenization: {sentence}")
print(f"Collected {len(tokens)} tokens.")

# -----------------------------
# Language detection & sentiment check
# -----------------------------
print("Detecting languages and sentiment...")
for s in sentences:
    try:
        lang = detect(s)
        sentiment = TextBlob(s).sentiment.polarity
        print(f"'{s}' → language: {lang}, sentiment: {sentiment}")
    except:
        print(f"Could not process: {s}")

# -----------------------------
# Load model and tokenizer
# -----------------------------
print("\nLoading model and tokenizer...")
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# -----------------------------
# Masking function
# -----------------------------
def mask_similar_words(sentences, toxic_words=None, threshold=0.8):
    """
    Mask toxic words in any input sentences.
    sentences: list of str or dicts with key 'toxic_sentence'
    toxic_words: list of strings or dicts with 'text'
    """
    # Normalize toxic_words to strings
    if toxic_words and isinstance(toxic_words[0], dict) and "text" in toxic_words[0]:
        toxic_words = [w["text"] for w in toxic_words]

    # Normalize sentences
    normalized_sentences = []
    for s in sentences:
        if isinstance(s, str):
            normalized_sentences.append({'toxic_sentence': s})
        elif isinstance(s, dict) and 'toxic_sentence' in s:
            normalized_sentences.append(s)

    masked_sentences = []
    for sent_obj in normalized_sentences:
        sentence = sent_obj['toxic_sentence']
        lang = detect(sentence)
        masked_sentence = sentence

        words = nltk.word_tokenize(sentence)

        # Mask words similar to toxic_words
        if toxic_words:
            for toxic_word in toxic_words:
                for w in words:
                    if SequenceMatcher(None, toxic_word, w).ratio() >= threshold:
                        masked_sentence = masked_sentence.replace(w, '<mask>').replace('<mask> <mask>', '<mask>')

        # Predict masked words
        try:
            unmasked_sentence = unmasker(masked_sentence)[0]['sequence']
        except:
            unmasked_sentence = masked_sentence

        masked_sentences.append({
            'toxic_sentence': sentence,
            'masked_sentence': unmasked_sentence,
            'language': lang
        })

    return masked_sentences

# -----------------------------
# Example toxic words (small sample)
# -----------------------------
hi_lexicon = ["stupid", "hate", "ugly", "shit"]

# -----------------------------
# Run the masking function
# -----------------------------
masked_sentences = mask_similar_words(sentences, toxic_words=hi_lexicon)

# -----------------------------
# Show results
# -----------------------------
for entry in masked_sentences:
    print(f"Original: {entry['toxic_sentence']}")
    print(f"Masked / Detoxified: {entry['masked_sentence']}")
    print(f"Language: {entry['language']}\n")
