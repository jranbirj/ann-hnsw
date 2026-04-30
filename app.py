from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from hnsw import HNSW

app = Flask(__name__, static_folder="static")
CORS(app)

graph = HNSW(M=6, ef_construction=50, seed=42)

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    global graph
    data = request.get_json(silent=True) or {}
    M = data.get("M", 6)
    ef = data.get("ef_construction", 50)
    seed = data.get("seed", 42)
    graph = HNSW(M=M, ef_construction=ef, seed=seed)
    return jsonify({"status": "reset"})


@app.route("/insert", methods=["POST"])
def insert():
    data = request.get_json()
    vector = data["vector"]
    node_id = graph.insert(vector)
    node = graph.nodes[node_id]
    layer = max(node.neighbors.keys()) if node.neighbors else 0
    return jsonify({"node_id": node_id, "layer": layer, "graph": graph.get_graph_state()})


@app.route("/build", methods=["POST"])
def build():
    data = request.get_json(silent=True) or {}
    n = data.get("n", 50)
    seed = data.get("seed", 42)
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2)).tolist()
    for p in points:
        graph.insert(p)
    return jsonify({"graph": graph.get_graph_state()})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    vector = data["vector"]
    k = data.get("k", 5)
    ef = data.get("ef", 50)
    result = graph.query(vector, k=k, ef=ef)
    return jsonify({
        "results": [
            {"id": nid, "vector": graph.nodes[nid].vector, "distance": dist}
            for dist, nid in result["results"]
        ],
        "traversal": result["traversal"],
        "query_vector": vector,
    })


@app.route("/graph", methods=["GET"])
def get_graph():
    return jsonify(graph.get_graph_state())


if __name__ == "__main__":
    app.run(debug=True, port=5001)