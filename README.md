# 💬 Mental Health Chatbot — Mistral 7B Fine-Tuning with QLoRA

> **Fine-tuning Mistral 7B Instruct on 7,000 mental health conversations with BERTScore evaluation and RAG-style context injection**

---

## 📊 Results at a Glance

| Metric | Base Mistral 7B | Fine-Tuned Model |
|---|---|---|
| BERTScore Precision | 0.7642 | 0.8493 | +8.5pp |
| BERTScore Recall | 0.8404 | 0.8592 | +1.9pp |
| **BERTScore F1 ✅** | **0.8003** | **0.8541** | **+5.4pp** |
| Response safety | Generic | Context-grounded, empathetic | RAG pipeline |

> Full numeric results available in `/results/bert_scores_base.csv` and `/results/bert_scores_finetuned.csv`

---

## 🧠 Project Overview

This project was completed as part of the **DataBytes x Deakin University AI Capstone** (2025). I served as **overall project lead** across a team of 8, coordinating all fine-tuning, evaluation, and pipeline engineering work.

The objective was to fine-tune an open-source LLM to produce safe, empathetic, context-aware responses for mental health conversations — going beyond generic chatbot responses toward genuinely supportive dialogue.

---

## 🗂️ Dataset

**AugESC** — an augmented emotional support conversation dataset from Tsinghua University.

| Split | Size |
|---|---|
| Training | 7,000 conversations |
| Validation | 1,000 conversations |
| Test | 3,000 conversations |

Each record is a multi-turn dialogue between a user and a supportive assistant, formatted for Mistral instruction-style prompting:

```
<s>[INST] You are a supportive mental health assistant.
User: I've been feeling overwhelmed lately...
Assistant: I hear you... [/INST]
```

---

## 🏗️ Architecture & Pipeline

```
AugESC Dataset (Hugging Face)
       │
       ▼
  Data Formatting
  (Mistral instruction format + system context)
       │
       ├──→ Train split (7K)
       ├──→ Validation split (1K)
       └──→ Test split (3K)
                │
                ▼
  ┌────────────────────────────────────────┐
  │     Mistral 7B Instruct v0.2           │
  │  (mistralai/Mistral-7B-Instruct-v0.2)  │
  └────────────────────────────────────────┘
                │
  Baseline benchmark (BERTScore F1 on 100 samples)
                │
                ▼
  ┌────────────────────────────────────────┐
  │      QLoRA Fine-Tuning                 │
  │  LoRA rank=64, alpha=16, NF4 4-bit     │
  │  SFTTrainer (TRL) + cosine LR          │
  │  Gradient checkpointing enabled        │
  └────────────────────────────────────────┘
                │
  Post-tuning BERTScore evaluation
                │
                ▼
  ┌────────────────────────────────────────┐
  │      RAG-Style Prompt Pipeline         │
  │  Mental health context → Mistral →     │
  │  Grounded, safe response               │
  └────────────────────────────────────────┘
```

---

## ⚙️ Fine-Tuning Configuration

```python
# Quantization — 4-bit NF4 via BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# LoRA configuration
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Training arguments
train_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=2,
    fp16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200
)
```

---

## 🔁 RAG-Style Prompt Pipeline

Before each model interaction, structured mental health context is injected into the prompt to ground the model's responses:

```python
prompt = (
    "<s>[INST] You are a supportive mental health assistant. "
    "Respond with empathy and encouragement.\n"
    "User: {user_message} [/INST]"
)
```

This approach — injecting context at inference time — reduces hallucination and ensures responses remain within appropriate mental health support boundaries, without requiring full model retraining for each new context type.

---

## 📈 Evaluation

Evaluation uses **BERTScore** — a semantic similarity metric that compares model-generated responses against reference responses using contextual embeddings from a pretrained BERT model.

```python
from bert_score import score

P, R, F1 = score(generated, references, lang="en")

print(f"BERTScore Precision: {P.mean().item():.4f}")
print(f"BERTScore Recall:    {R.mean().item():.4f}")
print(f"BERTScore F1:        {F1.mean().item():.4f}")  # Main metric
```

BERTScore F1 is the primary evaluation metric — it measures how semantically similar the generated response is to the reference, capturing meaning rather than just token overlap (unlike BLEU/ROUGE).

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Base model | Mistral 7B Instruct v0.2 (mistralai/Mistral-7B-Instruct-v0.2) |
| Fine-tuning | PEFT, QLoRA (rank 64), BitsAndBytes (4-bit NF4) |
| Training framework | TRL (SFTTrainer), Hugging Face Transformers |
| Data processing | Hugging Face Datasets, Pandas |
| Evaluation | BERTScore, sentence-transformers |
| Context pipeline | RAG-style prompt injection |
| Compute | Google Colab (GPU), Google Drive |
| Version control | Git, GitHub |
| Project management | Agile / Scrum — team of 8 |

---

## 🗃️ Repository Structure

```
mental-health-chatbot/
├── notebooks/
│   └── Mental_Health_Chatbot.ipynb     # Full pipeline: load → benchmark → fine-tune → RAG → evaluate
├── data/
│   └── README.md                        # AugESC dataset download instructions
├── results/
│   ├── bert_scores_base.csv             # BERTScore results — base model
│   └── bert_scores_finetuned.csv        # BERTScore results — fine-tuned model
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/hashaamkhan/mental-health-chatbot.git
cd mental-health-chatbot
```

**2. Install dependencies**
```bash
pip install transformers datasets accelerate peft trl bitsandbytes bert-score sentence-transformers evaluate nltk
```

**3. Add Hugging Face token**
```python
from huggingface_hub import login
login()  # Mistral 7B requires acceptance of terms on Hugging Face
```

**4. Run the notebook**

Open `notebooks/Mental_Health_Chatbot.ipynb` in Google Colab (GPU runtime recommended — A100 preferred for training, T4 for inference).

---

## ⚠️ Responsible AI Note

This project is a research prototype built for academic and portfolio purposes. It is **not** intended as a replacement for professional mental health support. If you or someone you know is in distress, please contact a licensed mental health professional or a crisis helpline.

---

## 👥 Team

Led by **Hashaam Khan** (overall project lead) — team of 8, DataBytes x Deakin University AI Capstone.

Methodology: Agile Scrum — weekly standups, sprint retrospectives, stakeholder presentations.

---

## 📬 Contact

**Hashaam Khan**
- 📧 hashaamkhan542@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/hashaamkhan)
- 🐙 [GitHub](https://github.com/hashaamkhan)

> *Master of Data Science (Distinction) · Deakin University · Melbourne, AU*
> *Open to data science and AI/ML roles · 485 Graduate Visa · Full work rights*
