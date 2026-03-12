import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file when running locally

# ── API Key ───────────────────────────────────────────────────────────────────
# Locally  : put GROQ_API_KEY=your_key_here in the .env file
# Streamlit : add GROQ_API_KEY in App Settings → Secrets
groq_api_key = os.getenv("GROQ_API_KEY", "")

# ── Model paths ───────────────────────────────────────────────────────────────
# Both model files must be present inside the /models folder in the project root.
# Folder structure:
#   asd_agent_v2/
#   ├── models/
#   │   ├── xgboost_asd_model.pkl
#   │   └── asd_classifier_model/       ← BERT folder (unpacked)
#   │       ├── config.json
#   │       ├── pytorch_model.bin
#   │       └── vocab.txt  (etc.)
#   ├── config.py
#   ├── agent.py
#   ├── app.py
#   ├── requirements.txt
#   └── .env                            ← local only, never commit to GitHub

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR        = os.path.join(BASE_DIR, "models")

XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_asd_model.pkl")
BERT_MODEL_PATH    = os.path.join(MODELS_DIR, "asd_classifier_model")
