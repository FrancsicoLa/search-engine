from flask import Flask, render_template, request, jsonify
from search_engine import SearchEngine

app = Flask(__name__)

# Cargar el motor al iniciar
print("Loading search engine...")
engine = SearchEngine("corpus.json")
print("Ready!")

@app.route("/")
def index():
    stats = engine.stats()
    return render_template("index.html", stats=stats)

@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": [], "time": 0, "query_terms": []})

    results, elapsed, terms = engine.search(query)
    return jsonify({
        "results": results,
        "time": elapsed,
        "query_terms": terms
    })

@app.route("/autocomplete")
def autocomplete():
    prefix = request.args.get("q", "")
    suggestions = engine.autocomplete(prefix)
    return jsonify({"suggestions": suggestions})

@app.route("/index-view")
def index_view():
    # Mostrar las primeras 50 entradas del índice invertido
    index_data = {}
    for term, postings in list(engine.index.items())[:50]:
        index_data[term] = postings
    return jsonify({
        "index": index_data,
        "vocab_size": len(engine.vocab),
        "total_docs": len(engine.documents)
    })

if __name__ == "__main__":
    app.run(debug=True)
