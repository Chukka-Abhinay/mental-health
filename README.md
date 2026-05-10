# Mental Health Sentiment Detector

A real-time mental health sentiment analysis web application powered by a fine-tuned **RoBERTa + BiLSTM** model. It classifies social media text into five mental health categories with confidence scores.

---

## Live Demo

> Deployed on Render — [your-app-url.onrender.com](https://your-app-url.onrender.com)

---

## Features

- **5-class classification** — Anxiety, Depression, Normal, Stress, Suicidal
- **Confidence scores** with bar chart visualization for all categories
- **Batch analysis** — upload a CSV of texts and analyse all at once
- **Session history** — view all past analyses in the current session
- **Sample texts** — load pre-written examples to try the model instantly
- **Crisis alerts** — prominent warning shown for suicidal signals
- **Fully offline inference** — no external API calls, model runs locally

---

## Mental Health Categories

| Category | Description |
|---|---|
| 😰 **Anxiety** | Excessive worry, fear, racing thoughts, panic |
| 😔 **Depression** | Hopelessness, loss of interest, low energy, emptiness |
| 😊 **Normal** | Neutral or positive emotional state |
| 😤 **Stress** | Overwhelm from work, deadlines, or life pressures |
| 🆘 **Suicidal** | Crisis signals, self-harm ideation, hopelessness |

---

## How to Use

### Single Text Analysis
1. Open the app and go to the **Single Analysis** tab
2. Either:
   - Type or paste any social media post into the text box, **or**
   - Use the **"Load a sample text"** dropdown to auto-fill an example
3. Click **"Analyse Sentiment"**
4. View the detected category, confidence score, and score breakdown chart

### Batch Analysis (CSV)
1. Go to the **Batch Analysis** tab
2. Upload a `.csv` file with a column containing text (e.g. `text`, `post`, `content`)
3. Select which column to analyse
4. Click **"Analyse All"** — results are shown in a table with category and confidence
5. Download results as CSV

### Session History
- Every analysis is saved to the **History** tab automatically
- View all past inputs and results from the current session
- Export history as CSV

---

## Model Architecture

```
Text Input
    │
    ▼
RoBERTa Tokenizer (vocab: 50,265, max length: 512)
    │
    ▼
RoBERTa-base Encoder (12 transformer layers, hidden: 768)
    │
    ▼
BiLSTM Layer (hidden: 384, bidirectional → output: 768)
    │
    ▼
Linear Classifier (768 → 5)
    │
    ▼
Softmax → [anxiety, depression, normal, stress, suicidal]
```

**Model size:** 489 MB | **Framework:** PyTorch + HuggingFace Transformers

---

## Run Locally

### Prerequisites
- Python 3.10+
- `model.safetensors` placed in the project root (not included in repo — see below)

### Setup

```bash
git clone https://github.com/rohank073/mental-health-chatbot.git
cd mental-health-chatbot

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Download the model

Set your Hugging Face token and run:

```bash
set HF_MODEL_REPO=rohank073/mental-health-bert
set HF_TOKEN=your_hf_token_here
python startup.py
```

### Run the app

```bash
streamlit run streamlit_mental_health_bert.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
├── streamlit_mental_health_bert.py   # Main Streamlit UI
├── mental_health_classifier.py       # Model loading & inference
├── startup.py                        # Downloads model at build time
├── tokenizer.json                    # RoBERTa tokenizer vocab
├── tokenizer_config.json             # Tokenizer configuration
├── requirements.txt                  # Python dependencies
├── render.yaml                       # Render deployment config
└── .streamlit/
    └── config.toml                   # Streamlit settings
```

---

## Deployment

The app is deployed on **Render** using `render.yaml`. The model is hosted on **Hugging Face Hub** and downloaded automatically during the build phase via `startup.py`.

### Environment Variables (set in Render dashboard)

| Variable | Value |
|---|---|
| `HF_MODEL_REPO` | `rohank073/mental-health-bert` |
| `HF_TOKEN` | Your Hugging Face access token |

---

## Performance

Evaluated on 1,000 real-world statements:

| Category | Accuracy |
|---|---|
| Normal | 97% |
| Stress | 57% |
| Depression | 40% |
| Anxiety | 33% |
| Suicidal | 18% |
| **Overall** | **48%** |

> Low accuracy on minority classes is due to class imbalance in the training data.

---

## Tech Stack

- **Model:** RoBERTa-base + BiLSTM (PyTorch)
- **UI:** Streamlit + Plotly
- **Model Hub:** Hugging Face Hub
- **Hosting:** Render

---

## Disclaimer

This tool is for **research and educational purposes only**. It is not a substitute for professional mental health diagnosis or treatment. If you or someone you know is in crisis, please contact a mental health professional or crisis helpline immediately.
