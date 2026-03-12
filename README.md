# ASD Detection AI Agent v2

LangGraph-based ASD screening agent with Streamlit interface.

## Folder Structure

```
asd_agent_v2/
├── models/                        ← put your model files here (not on GitHub)
│   ├── xgboost_asd_model.pkl
│   └── asd_classifier_model/      ← unpacked BERT folder
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt  (etc.)
├── config.py                      ← paths + API key loader
├── agent.py                       ← full LangGraph agent
├── app.py                         ← Streamlit UI
├── requirements.txt
├── .env.example                   ← copy to .env and add your key
├── .gitignore
└── README.md
```

## Local Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env
# open .env and set GROQ_API_KEY=your_key_here

# 4. Place models in the /models folder (see structure above)

# 5. Run
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this folder to GitHub (models folder is gitignored — too large)
2. Host models on Google Drive or Hugging Face Hub and download at runtime, OR use Streamlit Cloud's file storage
3. In Streamlit Cloud → App Settings → Secrets, add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy — any push to main branch auto-redeploys

## What's New in v2

- **Confidence scores** — both models now show full probability breakdown (ASD% vs Non-ASD%)
- **Input relevance gate** — LLM filters irrelevant inputs before BERT runs
- **Post-result flow** — after any prediction, agent asks if user wants another assessment
- **Robust JSON parsing** — regex fallback prevents supervisor crashes
- **Clean separation** — agent logic in agent.py, UI in app.py, config in config.py
