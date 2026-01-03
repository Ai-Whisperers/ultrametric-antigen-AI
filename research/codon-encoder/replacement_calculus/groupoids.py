"""Global Optima as Groupoids.

Global optima are NOT single groups—they are GROUPOIDS:
- Multiple local coordinate systems
- Partial symmetries
- Context-dependent transitions

This is why biology survives mutation: it operates on groupoids.
Civilization keeps pretending it's a group (single truth), which is fragile.

Key insight: Local minima become COORDINATES, not traps.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import heapq

import numpy as np

from .groups import LocalMinimum
from .morphisms import Morphism, is_valid_morphism, compose_morphisms
from .invariants import InvariantTuple


@dataclass
class Groupoid:
    """A groupoid of local minima with valid morphisms.

    A groupoid generalizes groups to have:
    - Multiple identity elements (one per object)
    - Partial composition (only when source/target match)
    - Multiple coordinate systems

    Attributes:
        objects: Local minima as groupoid objects
        morphisms: Valid morphisms between objects
        name: Identifier for this groupoid
    """
    name: str = "unnamed"
    objects: List[LocalMinimum] = field(default_factory=list)
    morphisms: Dict[Tuple[int, int], List[Morphism]] = field(default_factory=lambda: defaultdict(list))

    # Index for fast lookup
    _object_index: Dict[str, int] = field(default_factory=dict, repr=False)

    def add_object(self, local_min: LocalMinimum) -> int:
        """Add a local minimum as a groupoid object.

        Returns:
            Index of the added object
        """
        idx = len(self.objects)
        self.objects.append(local_min)
        self._object_index[local_min.name] = idx
        return idx

    def add_morphism(self, source_idx: int, target_idx: int, morphism: Morphism) -> bool:
        """Add a morphism between two objects.

        Only adds if morphism is valid.

        Returns:
            True if morphism was added
        """
        is_valid, reason = is_valid_morphism(morphism)
        if is_valid:
            self.morphisms[(source_idx, target_idx)].append(morphism)
            return True
        return False

    def get_object(self, idx: int) -> Optional[LocalMinimum]:
        """Get object by index."""
        if 0 <= idx < len(self.objects):
            return self.objects[idx]
        return None

    def get_object_by_name(self, name: str) -> Optional[LocalMinimum]:
        """Get object by name."""
        idx = self._object_index.get(name)
        if idx is not None:
            return self.objects[idx]
        return None

    def get_morphisms(self, source_idx: int, target_idx: int) -> List[Morphism]:
        """Get all morphisms from source to target."""
        return self.morphisms.get((source_idx, target_idx), [])

    def has_morphism(self, source_idx: int, target_idx: int) -> bool:
        """Check if any morphism exists between two objects."""
        return len(self.get_morphisms(source_idx, target_idx)) > 0

    def n_objects(self) -> int:
        """Number of objects in groupoid."""
        return len(self.objects)

    def n_morphisms(self) -> int:
        """Total number of morphisms."""
        return sum(len(ms) for ms in self.morphisms.values())

    def adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix (1 if morphism exists, 0 otherwise)."""
        n = self.n_objects()
        adj = np.zeros((n, n), dtype=int)
        for (i, j), ms in self.morphisms.items():
            if len(ms) > 0:
                adj[i, j] = 1
        return adj

    def cost_matrix(self) -> np.ndarray:
        """Get matrix of minimum morphism costs."""
        n = self.n_objects()
        costs = np.full((n, n), np.inf)
        np.fill_diagonal(costs, 0)

        for (i, j), ms in self.morphisms.items():
            if ms:
                min_cost = min(m.cost for m in ms)
                costs[i, j] = min_cost

        return costs

    def invariant_matrix(self) -> np.ndarray:
        """Get matrix of invariant tuples for each object."""
        result = []
        for obj in self.objects:
            I = obj.invariant_tuple()
            result.append([I.valuation, I.redundancy, I.symmetry_rank])
        return np.array(result)

    def connected_components(self) -> List[Set[int]]:
        """Find connected components (ignoring morphism direction)."""
        n = self.n_objects()
        visited = [False] * n
        components = []

        def dfs(start: int) -> Set[int]:
            component = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.add(node)

                # Add neighbors (both directions)
                for (i, j) in self.morphisms.keys():
                    if i == node and not visited[j]:
                        stack.append(j)
                    elif j == node and not visited[i]:
                        stack.append(i)

            return component

        for i in range(n):
            if not visited[i]:
                comp = dfs(i)
                if comp:
                    components.append(comp)

        return components

    def is_connected(self) -> bool:
        """Check if groupoid is connected."""
        return len(self.connected_components()) <= 1

    def maximal_elements(self) -> List[int]:
        """Find maximal elements (no outgoing morphisms to better elements)."""
        n = self.n_objects()
        maximal = []

        for i in range(n):
            I_i = self.objects[i].invariant_tuple()
            is_maximal = True

            for j in range(n):
                if i == j:
                    continue
                if self.has_morphism(i, j):
                    I_j = self.objects[j].invariant_tuple()
                    if I_j > I_i:
                        is_maximal = False
                        break

            if is_maximal:
                maximal.append(i)

        return maximal

    def minimal_elements(self) -> List[int]:
        """Find minimal elements (no incoming morphisms from worse elements)."""
        n = self.n_objects()
        minimal = []

        for i in range(n):
            I_i = self.objects[i].invariant_tuple()
            is_minimal = True

            for j in range(n):
                if i == j:
                    continue
                if self.has_morphism(j, i):
                    I_j = self.objects[j].invariant_tuple()
                    if I_j < I_i:
                        is_minimal = False
                        break

            if is_minimal:
                minimal.append(i)

        return minimal


def find_escape_path(
    groupoid: Groupoid,
    source_idx: int,
    target_idx: int,
) -> Optional[List[Morphism]]:
    """Find the lowest-cost path of morphisms from source to target.

    Uses Dijkstra's algorithm on morphism costs.

    Args:
        groupoid: The groupoid to search
        source_idx: Starting object index
        target_idx: Goal object index

    Returns:
        List of morphisms forming the path, or None if no path exists
    """
    n = groupoid.n_objects()

    # Priority queue: (cost, tiebreaker, current_idx, path)
    # Tiebreaker ensures deterministic ordering when costs are equal
    counter = 0
    heap = [(0.0, counter, source_idx, [])]
    visited = set()

    while heap:
        cost, _, current, path = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)

        if current == target_idx:
            return path

        # Explore outgoing morphisms
        for (src, tgt), morphisms in groupoid.morphisms.items():
            if src == current and tgt not in visited:
                for morphism in morphisms:
                    new_cost = cost + morphism.cost
                    new_path = path + [morphism]
                    counter += 1
                    heapq.heappush(heap, (new_cost, counter, tgt, new_path))

    return None


def find_all_escape_paths(
    groupoid: Groupoid,
    source_idx: int,
    max_length: int = 10,
) -> Dict[int, List[Morphism]]:
    """Find escape paths from source to all reachable objects.

    Args:
        groupoid: The groupoid
        source_idx: Starting object
        max_length: Maximum path length

    Returns:
        Dictionary mapping target index to path
    """
    paths = {}

    for target_idx in range(groupoid.n_objects()):
        if target_idx != source_idx:
            path = find_escape_path(groupoid, source_idx, target_idx)
            if path and len(path) <= max_length:
                paths[target_idx] = path

    return paths


def construct_groupoid_from_minima(
    minima: List[LocalMinimum],
    p: int = 3,
    max_morphisms_per_pair: int = 3,
) -> Groupoid:
    """Construct a groupoid from a list of local minima.

    Automatically finds valid morphisms between all pairs.

    Args:
        minima: List of local minima
        p: Prime for valuation
        max_morphisms_per_pair: Maximum morphisms to keep per pair

    Returns:
        Constructed groupoid
    """
    from .morphisms import find_morphisms_between

    groupoid = Groupoid(name="constructed")

    # Add all minima as objects
    for minimum in minima:
        groupoid.add_object(minimum)

    # Find morphisms between all pairs
    n = len(minima)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            morphisms = find_morphisms_between(
                minima[i], minima[j], p, max_morphisms_per_pair
            )

            for morphism in morphisms:
                groupoid.add_morphism(i, j, morphism)

    return groupoid


def analyze_groupoid_structure(groupoid: Groupoid) -> Dict:
    """Analyze the structure of a groupoid.

    Returns summary statistics and structural properties.
    """
    n_objects = groupoid.n_objects()
    n_morphisms = groupoid.n_morphisms()
    components = groupoid.connected_components()
    maximal = groupoid.maximal_elements()
    minimal = groupoid.minimal_elements()

    # Invariant statistics
    invariants = groupoid.invariant_matrix()
    mean_valuation = np.mean(invariants[:, 0]) if len(invariants) > 0 else 0
    mean_redundancy = np.mean(invariants[:, 1]) if len(invariants) > 0 else 0
    mean_symmetry = np.mean(invariants[:, 2]) if len(invariants) > 0 else 0

    # Cost statistics
    cost_matrix = groupoid.cost_matrix()
    finite_costs = cost_matrix[np.isfinite(cost_matrix) & (cost_matrix > 0)]
    mean_cost = np.mean(finite_costs) if len(finite_costs) > 0 else 0

    return {
        "n_objects": n_objects,
        "n_morphisms": n_morphisms,
        "n_components": len(components),
        "is_connected": groupoid.is_connected(),
        "n_maximal": len(maximal),
        "n_minimal": len(minimal),
        "maximal_indices": maximal,
        "minimal_indices": minimal,
        "mean_valuation": mean_valuation,
        "mean_redundancy": mean_redundancy,
        "mean_symmetry": mean_symmetry,
        "mean_morphism_cost": mean_cost,
    }
