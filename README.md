# AI Career Coach - Streamlit

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app accepts a PDF CV and 3 evaluation questions, then uses:
- PyPDFLoader
- HuggingFace Embeddings
- FAISS
- Mistral-Nemo-Instruct-2407
- Structured JSON parsing

## Hugging Face authentication

The original notebook uses `login()` interactively. For Streamlit/deployment, use a Hugging Face access token as an environment variable if the model requires authentication:

Windows PowerShell:
```powershell
$env:HF_TOKEN="YOUR_TOKEN"
streamlit run app.py
```

Linux/macOS:
```bash
export HF_TOKEN="YOUR_TOKEN"
streamlit run app.py
```

If needed, add:
```python
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))
```
before loading the model.

## Important

Mistral-Nemo is a large model. A GPU is strongly recommended.
