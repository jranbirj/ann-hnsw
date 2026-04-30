# Core HNSW logic

import math
import random
import heapq
from dataclasses import dataclass, field
from typing import Optional

@dataclass

# id for integer index, 
# vector for the actual point, 
# and neighbor mapping.
class Node: 
    id: int
    vector: list[float]
    neighbors: dict = field(default_factory=dict)

# HNSW implementation
class HNSW: 
    def __init__(self, M=16, ef_construction=200, seed=None):
        """
        M is a parameter for sparsity, specifically max number of neighbors a node can have.
        ef_construction controls set / pool of neighbors to consider to pick best M for graph building. 
        Base refers to bottom dense layer. 
        mL is the layer assignment logic, which uses an exponential decay distribution 
        As layer index increases, assignment probability decreases. 
        Pseudorandom seed for reproducibility. 
        """
        self.M = M
        self.base = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1 / math.log(M)
        self.nodes: list[Node] = []
        self.entry_point: Optional[int] = None
        self.max_layer = 0
        self._rng = random.Random(seed)

    def _distance(self, a: list[float], b: list[float]) -> float:
        """Euclidean distance between two vectors."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _get_layer(self) -> int:
        """
        Sample the insertion layer using exponential decay:
            l = floor( -ln(uniform(0,1)) * mL )
        Higher layers are exponentially less likely.
        """
        return int(-math.log(self._rng.random()) * self.mL)

    def _search_layer(
        self,
        query: list[float],
        entry_id: int,
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """
        Greedy beam search within a single layer.
        Returns a list of (distance, node_id) pairs sorted nearest-first.
        """
        entry_dist = self._distance(query, self.nodes[entry_id].vector)

        # candidates : min-heap — always expand the closest unexplored node
        # found      : max-heap — tracks the ef best seen so far (negated dist)
        # visited    : avoid re-processing nodes
        candidates = [(entry_dist, entry_id)]
        found = [(-entry_dist, entry_id)]
        visited = {entry_id}

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            worst_found = -found[0][0]

            # If the closest unexplored node is already further than our
            # worst kept candidate, no further improvement is possible.
            if c_dist > worst_found:
                break

            for nb_id in self.nodes[c_id].neighbors.get(layer, []):
                if nb_id in visited:
                    continue
                visited.add(nb_id)
                nb_dist = self._distance(query, self.nodes[nb_id].vector)
                worst_found = -found[0][0]

                if nb_dist < worst_found or len(found) < ef:
                    heapq.heappush(candidates, (nb_dist, nb_id))
                    heapq.heappush(found, (-nb_dist, nb_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted((-d, nid) for d, nid in found)

    def _select_neighbors(
        self,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """
        Keep the M closest candidates.
        candidates : list of (distance, node_id), nearest-first.
        """
        return [nid for _, nid in candidates[:M]]

    def _max_neighbors(self, layer: int) -> int:
        """Layer-0 uses 2*M (self.base); all other layers use M."""
        return self.base if layer == 0 else self.M

    def insert(self, vector: list[float]) -> int:
        """
        Insert a new vector into the index.
        Returns the id assigned to the new node.
        """
        node_id = len(self.nodes)
        node = Node(id=node_id, vector=vector)
        self.nodes.append(node)

        insert_layer = self._get_layer()

        # First node: just become the entry point
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = insert_layer
            return node_id

        ep = self.entry_point

        # Greedy descent from max_layer down to insert_layer+1 (ef=1)
        for layer in range(self.max_layer, insert_layer, -1):
            results = self._search_layer(vector, ep, ef=1, layer=layer)
            ep = results[0][1]

        # From insert_layer down to 0: beam search + connect edges
        for layer in range(min(insert_layer, self.max_layer), -1, -1):
            candidates = self._search_layer(
                vector, ep, ef=self.ef_construction, layer=layer
            )
            M_layer = self._max_neighbors(layer)
            neighbors = self._select_neighbors(candidates, M_layer)

            # Wire bidirectional edges
            node.neighbors[layer] = neighbors
            for nb_id in neighbors:
                nb_node = self.nodes[nb_id]
                nb_neighbors = nb_node.neighbors.get(layer, [])
                nb_neighbors.append(node_id)

                # Prune neighbour list if it exceeds the layer limit
                if len(nb_neighbors) > M_layer:
                    nb_dists = [
                        (self._distance(nb_node.vector, self.nodes[x].vector), x)
                        for x in nb_neighbors
                    ]
                    nb_neighbors = self._select_neighbors(
                        sorted(nb_dists), M_layer
                    )

                nb_node.neighbors[layer] = nb_neighbors

            ep = candidates[0][1]

        # Promote entry point if new node sits higher
        if insert_layer > self.max_layer:
            self.max_layer = insert_layer
            self.entry_point = node_id

        return node_id

    def query(self, vector: list[float], k: int = 5, ef: int = None) -> list[tuple[float, int]]:
        """
        Find the k approximate nearest neighbours of vector.
        ef : candidate pool size (defaults to max(k, ef_construction)).
        Returns a list of (distance, node_id) sorted nearest-first.
        """
        if not self.nodes:
            return []

        if ef is None:
            ef = max(k, self.ef_construction)

        ep = self.entry_point

        # Greedy descent through upper layers
        for layer in range(self.max_layer, 0, -1):
            results = self._search_layer(vector, ep, ef=1, layer=layer)
            ep = results[0][1]

        # Full beam search at layer 0
        candidates = self._search_layer(vector, ep, ef=ef, layer=0)

        return candidates[:k]


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import random as rnd

    DIM = 32
    N = 500
    K = 5
    rnd.seed(42)

    index = HNSW(M=16, ef_construction=200, seed=42)

    vecs = [[rnd.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
    for v in vecs:
        index.insert(v)

    print(f"Inserted {N} vectors | layers built: 0..{index.max_layer}")

    results = index.query(vecs[0], k=K)
    print(f"\nTop-{K} neighbours of node 0:")
    for dist, nid in results:
        print(f"  node {nid:4d}  dist={dist:.4f}")

    exact = sorted(
        (math.sqrt(sum((a - b) ** 2 for a, b in zip(vecs[0], vecs[i]))), i)
        for i in range(N)
    )[:K]
    hnsw_ids = {nid for _, nid in results}
    exact_ids = {nid for _, nid in exact}
    recall = len(hnsw_ids & exact_ids) / K
    print(f"\nRecall@{K}: {recall:.2f}  (HNSW ∩ exact = {hnsw_ids & exact_ids})")
