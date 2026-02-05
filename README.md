# eduseg_de
This repository contains the XLM-RoBERTa model for discourse segmentation used in our paper:

**Discourse Segmentation of German Text with Pretrained Language Models** -- 
Steffen Frenzel, Maximilian Krupop, Manfred Stede, 2026, Journal for Language Technology and Computational Linguistics (JLCL)

The repository contains the exact model used in the paper, along with configuration and tokenizer files.

## Table of Contents
* Canonical GitHub Version
* Hugging Face Mirror
* Loading the Model
* Model Details
* Citing

### Canonical GitHub Version
The canonical version of the model, linked to our paper, is frozen under the GitHub tag:
`eduseg_de-v1.0`

Model files are located in: `\model`

This includes:
* `model.safetensors` — the model weights
* `.json` files — model configuration and tokenizer metadata
* `.model` — SentencePiece tokenizer model

Always refer to the `paper-v1.0` tag for paper results.

### Hugging Face Mirror
For convenience, the same model is mirrored on Hugging Face. The Hugging Face repository is bit-identical to the GitHub tag `paper-v1.0`.
GitHub remains the canonical reference for reproducibility.

### Loading the model
```
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Using GitHub
MODEL_PATH_GH = "https://github.com/discourse_lab/eduseg_de/raw/paper-v1.0/model/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_GH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH_GH)

# Using Hugging Face
MODEL_PATH_HF = "sfrenzel/eduseg_de"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_HF)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH_HF)
```

* Requires `transformers >= 4.25`
* `.safetensors` is used for the model weights

### Model details
* Architecture: XLM-RoBERTa
* Task: Discourse segmentation (token classification)
* Framework: Hugging Face Transformers
* Tokenizer: SentencePiece (included in repo)

### Using the model
The model accepts plain text (`.txt`) as input. The output will be `.txt`as well, segments are separated by newline.
You can use the scripts provided in `\code` for finetuning and inference.

### Citing
If you use this model, please cite our paper:
```
@article{Frenzel_Krupop_Stede_2026, title={Discourse Segmentation of German Text with Pretrained Language Models}, volume={39}, url={https://jlcl.org/article/view/306}, DOI={10.21248/jlcl.39.2026.306}, number={1}, journal={Journal for Language Technology and Computational Linguistics}, author={Frenzel, Steffen and Krupop, Maximilian and Stede, Manfred}, year={2026}, month={Feb.}, pages={1–31} }
```











