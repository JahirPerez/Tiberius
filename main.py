from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

text = """
Apollo carried 300 jars of wine on July 10.
Hermes arrived in Alexandria on August 3.
Zephyr departed from Corinth with olive oil on July 15.
The crew of Apollo reported calm seas and favorable winds.
Hermes transported textiles and spices from Carthage.
Zephyr encountered rough seas near Crete.
Apollo docked in Rome on July 12.
Hermes left Alexandria on August 5.
Zephyr returned to Corinth on July 20.
Apollo was inspected by the port authority on July 11.
"""

chunks = [line.strip() for line in text.split('\n') if line.strip()]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def search(query):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), 1)
    return chunks[indices[0][0]]

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = None
    if request.method == "POST":
        query = request.form["query"]
        answer = search(query)
    return render_template("chat.html", answer=answer)

if __name__ == "__main__":
    app.run()
