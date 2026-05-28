from transformers import AutoTokenizer
from pathlib import Path
import subprocess
import sys

import spacy

# ------------------------------------------------
# Configuration
# ------------------------------------------------

INPUT_FILE = "/content/input.txt"

# Transformer model
TRANSFORMER_MODEL = "xlm-roberta-base"

# spaCy sentence segmentation model
SPACY_MODEL = "xx_sent_ud_sm"

MAX_TOKENS = 512

# ------------------------------------------------
# Load transformer tokenizer
# ------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    TRANSFORMER_MODEL
)

# ------------------------------------------------
# Load spaCy model
# ------------------------------------------------

try:
    nlp = spacy.load(SPACY_MODEL)

except OSError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "spacy",
        "download",
        SPACY_MODEL
    ])
    nlp = spacy.load(SPACY_MODEL)

# Add sentence boundary detection
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# -----------------------------
# Read text file
# -----------------------------

input_path = Path(INPUT_FILE)

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------
# Sentence segmentation
# -----------------------------

doc = nlp(text)
sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# -----------------------------
# Create token-safe chunks
# -----------------------------

chunks = []
current_chunk = []
current_length = 0

for sentence in sentences:

    sentence_tokens = tokenizer.encode(
        sentence,
        add_special_tokens=False
    )

    sentence_length = len(sentence_tokens)

    # Handle extremely long single sentences
    if sentence_length > MAX_TOKENS:

        # Save current chunk first
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        # Split oversized sentence by tokens
        for i in range(0, sentence_length, MAX_TOKENS):
            token_slice = sentence_tokens[i:i + MAX_TOKENS]
            chunk_text = tokenizer.decode(
                token_slice,
                skip_special_tokens=True
            )
            chunks.append(chunk_text)

        continue

    # Normal chunking
    if current_length + sentence_length <= MAX_TOKENS:
        current_chunk.append(sentence)
        current_length += sentence_length
    else:
        chunks.append(" ".join(current_chunk))

        current_chunk = [sentence]
        current_length = sentence_length

# Add final chunk
if current_chunk:
    chunks.append(" ".join(current_chunk))

# -----------------------------
# Save chunks as txt files
# -----------------------------

base_name = input_path.stem
extension = input_path.suffix

for idx, chunk in enumerate(chunks, start=1):

    output_path = (
        input_path.parent
        / f"{base_name}_{idx:04d}{extension}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(chunk)

print(f"Created {len(chunks)} chunk files.")