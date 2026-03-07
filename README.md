# SmartFit (Streamlit)

A simple fitness chatbot + workout progress tracker.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Gemini API key

This app does **not** accept API keys via the UI.

Use one of these options:

1. Streamlit secrets (recommended)

- Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
- Set `GEMINI_API_KEY` in `.streamlit/secrets.toml` (paste the key inside the quotes)

2. Environment variable

- Set `GEMINI_API_KEY` in your shell/user environment
