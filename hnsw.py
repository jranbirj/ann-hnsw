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


