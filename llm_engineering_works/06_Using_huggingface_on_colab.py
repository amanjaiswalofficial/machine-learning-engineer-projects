from transformers import pipeline, AutoTokenizer
from huggingface_hub import login

# Login to Hugging Face (assuming token is set up)
hf_token = "YOUR_HF_TOKEN"
login(hf_token, add_to_git_credential=True)

# ---------------------
# 1. Pipelines Examples
# ---------------------

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I'm super excited to be on the way to LLM mastery!")
print("Sentiment:", result)

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
result = ner("Barack Obama was the 44th president of the United States.")
print("NER:", result)

# Translation
translator = pipeline("translation_en_to_fr")
text_en = "The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API."
result = translator(text_en)
print("Translation:", result[0]['translation_text'])

# Text Generation
generator = pipeline("text-generation")
prompt = "If there's one thing I want you to remember about using HuggingFace pipelines, it's"
result = generator(prompt, max_new_tokens=30)
print("Generated Text:", result[0]['generated_text'])

# ------------------------
# 2. Tokenizer Examples
# ------------------------

# Load tokenizer for Meta LLaMA
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)
text = "I am excited to show Tokenizers in action to my LLM engineers"
tokens = tokenizer.encode(text)
print("Tokens:", tokens)
print("Decoded:", tokenizer.decode(tokens))
print("Batch Decoded:", tokenizer.batch_decode(tokens))

# Tokenizers for other models
PHI3_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
QWEN2_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
STARCODER2_MODEL_NAME = "bigcode/starcoder2-3b"

phi3_tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_NAME)
qwen2_tokenizer = AutoTokenizer.from_pretrained(QWEN2_MODEL_NAME)
starcoder2_tokenizer = AutoTokenizer.from_pretrained(STARCODER2_MODEL_NAME, trust_remote_code=True)

# Encode and decode with different tokenizers
print("\nEncoding/Decoding with PHI-3:")
print("Encoded:", phi3_tokenizer.encode(text))
print("Decoded:", phi3_tokenizer.batch_decode(tokens))

print("\nEncoding/Decoding with Qwen2:")
print("Encoded:", qwen2_tokenizer.encode(text))
print("Decoded:", qwen2_tokenizer.batch_decode(tokens))

# Tokenize code with Starcoder2
code = """
def hello_world(person):
  print("Hello", person)
"""
code_tokens = starcoder2_tokenizer.encode(code)
print("\nCode Tokenization with Starcoder2:")
for token in code_tokens:
    print(f"{token} = {starcoder2_tokenizer.decode(token)}")

# ------------------------
# 3. Chat Formatting Example
# ------------------------

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists."}
]

print("\nPHI-3 Chat Template:")
print(phi3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

print("\nQwen2 Chat Template:")
print(qwen2_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

# ------------------------
# 4. BONUS: Zero-shot classification (extra example)
# ------------------------

zero_shot_classifier = pipeline("zero-shot-classification")
sequence = "Transformers is a great tool for deep learning"
labels = ["machine learning", "cooking", "travel"]
result = zero_shot_classifier(sequence, candidate_labels=labels)
print("\nZero-shot Classification:")
print(result)
