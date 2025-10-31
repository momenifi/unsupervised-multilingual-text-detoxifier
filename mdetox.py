import nltk
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from langdetect import detect
from textblob import TextBlob
from tqdm import tqdm

# ✅ Sicherstellen, dass NLTK-Modelle da sind
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

print("Loading dataset...")
try:
    # Versuch, das echte Datenset von HuggingFace zu laden
    dataset = load_dataset("s-nlp/Multilingual-Toxic-Comment-Classification")
except Exception as e:
    print("⚠️ Could not load dataset from HuggingFace, using local sample instead.")
    data = {
        "comment_text": [
            "You are so stupid!",
            "What a beautiful day!",
            "I hate everything.",
            "This is a wonderful idea.",
            "You are such a nice person."
        ]
    }
    dataset = Dataset.from_dict(data)
    dataset = {"train": dataset}

# 🧠 Verwende nur einen kleinen Teil (spart Speicher)
toxic_sentences = dataset["train"]["comment_text"][:5]
print(f"Loaded {len(toxic_sentences)} example sentences.")

# ✅ Tokenisierung (robust)
print("Tokenizing sentences...")
toxic_tokens = []
for sentence in toxic_sentences:
    try:
        words = nltk.word_tokenize(sentence)
        toxic_tokens.extend(words)
    except Exception as e:
        print(f"Tokenization failed for: {sentence}")

print(f"Collected {len(toxic_tokens)} tokens.")

# ✅ Language detection test
print("Detecting languages...")
for s in toxic_sentences:
    try:
        lang = detect(s)
        print(f"'{s}' → language: {lang}")
    except:
        print(f"Could not detect language for: {s}")

# ✅ Text sentiment (sanity check)
print("\nSentiment analysis:")
for s in toxic_sentences:
    sentiment = TextBlob(s).sentiment.polarity
    print(f"{s} → sentiment: {sentiment}")

# ✅ Lade Modell
print("\nLoading model and tokenizer...")
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# ✅ Pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# ✅ Beispiel: Detoxify Satz
print("\nDetoxifying example:")
text = "You are so stupid!"
detox_text = text.replace("stupid", fill_mask("You are so [MASK]!")[0]["token_str"])
print(f"Original: {text}")
print(f"Detoxified: {detox_text}")

print("\n✅ Script completed successfully.")
