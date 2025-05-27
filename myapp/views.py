from django.shortcuts import render
from django.http import JsonResponse
from sentence_transformers import SentenceTransformer, util
import json
from django.views.decorators.csrf import csrf_exempt

# Load the model once (global)
model = SentenceTransformer('all-MiniLM-L6-v2')


@csrf_exempt
def search_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("query", "")
        string_list = data.get("strings", [])

        results = []

        if query and string_list:
            # Encode the query and string list
            query_embedding = model.encode(query, convert_to_tensor=True)
            corpus_embeddings = model.encode(string_list, convert_to_tensor=True)

            # Compute cosine similarities
            similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]

            # Get top 3 most similar strings
            top_results = similarities.topk(k=3)
            results = [string_list[idx] for idx in top_results[1]]

        return JsonResponse({"query": query, "results": results})

    return JsonResponse({"error": "Invalid request method"}, status=400)
