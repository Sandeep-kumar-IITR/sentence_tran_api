# Sentence Similarity Search API

A small Django API that ranks a supplied list of sentences by semantic similarity to a query. It uses a Sentence Transformers model to embed the query and candidate strings, then ranks candidates by cosine similarity.

## Endpoint

`POST /search/`

Example request:

```json
{
  "query": "machine learning engineer",
  "strings": [
    "I build backend services in Python",
    "I work on machine learning models",
    "I enjoy cooking"
  ]
}
```

The response contains the query and the top matching strings:

```json
{
  "query": "machine learning engineer",
  "results": [
    "I work on machine learning models",
    "I build backend services in Python"
  ]
}
```

The current view selects up to three results using cosine similarity. The model is loaded once when the Django view module is imported.

## Stack

- Python
- Django
- sentence-transformers
- PyTorch

## Run locally

```bash
git clone https://github.com/Sandeep-kumar-IITR/sentence_tran_api.git
cd sentence_tran_api
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python manage.py runserver
```

Send a JSON POST request to `http://127.0.0.1:8000/search/`.

## Current limitations

- The endpoint expects valid JSON and does not yet return structured validation errors.
- The top-k value is fixed at three; handle candidate lists shorter than three explicitly.
- The view is CSRF-exempt. Add appropriate authentication, request limits and CORS policy before exposing it publicly.
- The model download and inference cost should be considered when deploying.

## Author

Sandeep Kumar · [GitHub](https://github.com/Sandeep-kumar-IITR)
