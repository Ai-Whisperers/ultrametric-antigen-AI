"""
Resistance Pathway Graph Analysis for HIV Drug Resistance.

This module implements graph-based analysis of HIV drug resistance evolution,
modeling mutational pathways as directed graphs where nodes are resistance
states and edges are mutations with associated fitness costs.

Key features:
1. Mutation pathway graph construction from Stanford HIVDB data
2. Shortest path analysis for resistance acquisition
3. Bottleneck detection in resistance evolution
4. Integration with hyperbolic geometry for edge weights

Based on papers:
- Rhee et al. 2010: Resistance pathway patterns
- Theys et al. 2018: Fitness landscapes
- Zanini et al. 2015: Evolutionary dynamics

Author: Research Team
Date: December 2025
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MutationNode:
    """Represents a resistance state (set of mutations)."""

    mutations: frozenset  # Immutable set of mutations at this node
    resistance_level: float  # Fold-change resistance
    frequency: int  # How often observed in dataset
    fitness_cost: float  # Estimated replication fitness cost
    drug_class: str = ""


@dataclass
class MutationEdge:
    """Represents a transition between resistance states."""

    source: frozenset  # Source mutation set
    target: frozenset  # Target mutation set
    mutation_added: str  # The mutation that was added
    weight: float  # Edge weight (fitness cost / selection pressure)
    hyperbolic_distance: float  # Geometric distance of the transition
    frequency: int  # How often this transition observed


@dataclass
class ResistancePathway:
    """A complete pathway from wild-type to resistance."""

    mutations: list[str]  # Ordered list of mutations acquired
    total_fitness_cost: float
    total_hyperbolic_distance: float
    final_resistance: float  # Fold-change at end
    pathway_probability: float  # Estimated probability of this path
    bottleneck_mutations: list[str]  # Rate-limiting steps


class ResistanceGraph:
    """
    Graph representation of resistance evolution.

    Nodes are resistance states (mutation sets).
    Edges are single-mutation transitions.
    """

    def __init__(self, drug_class: str = ""):
        self.drug_class = drug_class
        self.nodes: dict[frozenset, MutationNode] = {}
        self.edges: dict[tuple[frozenset, frozenset], MutationEdge] = {}
        self.adjacency: dict[frozenset, list[frozenset]] = defaultdict(list)
        self.reverse_adjacency: dict[frozenset, list[frozenset]] = defaultdict(list)

    def add_node(self, mutations: frozenset, resistance: float, frequency: int = 1,
                 fitness_cost: float = 0.0):
        """Add a resistance state node to the graph."""
        if mutations not in self.nodes:
            self.nodes[mutations] = MutationNode(
                mutations=mutations,
                resistance_level=resistance,
                frequency=frequency,
                fitness_cost=fitness_cost,
                drug_class=self.drug_class,
            )
        else:
            # Update frequency
            self.nodes[mutations].frequency += frequency

    def add_edge(self, source: frozenset, target: frozenset, mutation: str,
                 weight: float = 1.0, hyperbolic_distance: float = 0.0):
        """Add a transition edge between resistance states."""
        edge_key = (source, target)

        if edge_key not in self.edges:
            self.edges[edge_key] = MutationEdge(
                source=source,
                target=target,
                mutation_added=mutation,
                weight=weight,
                hyperbolic_distance=hyperbolic_distance,
                frequency=1,
            )
            self.adjacency[source].append(target)
            self.reverse_adjacency[target].append(source)
        else:
            self.edges[edge_key].frequency += 1

    def get_neighbors(self, node: frozenset) -> list[frozenset]:
        """Get all nodes reachable from this node."""
        return self.adjacency.get(node, [])

    def get_predecessors(self, node: frozenset) -> list[frozenset]:
        """Get all nodes that lead to this node."""
        return self.reverse_adjacency.get(node, [])

    def get_edge(self, source: frozenset, target: frozenset) -> Optional[MutationEdge]:
        """Get edge between two nodes."""
        return self.edges.get((source, target))


# Known primary and accessory mutations by drug class
PRIMARY_MUTATIONS = {
    "PI": {"D30N", "V32I", "M46I", "M46L", "I47V", "I47A", "G48V", "I50V", "I50L",
           "I54V", "I54L", "I54M", "L76V", "V82A", "V82F", "V82T", "V82S", "I84V",
           "N88S", "L90M"},
    "NRTI": {"M41L", "K65R", "K65N", "D67N", "K70R", "K70E", "L74V", "L74I",
             "Y115F", "Q151M", "M184V", "M184I", "L210W", "T215Y", "T215F", "K219Q"},
    "NNRTI": {"K103N", "K103S", "V106A", "V106M", "V108I", "E138K", "E138A",
              "E138G", "Y181C", "Y181I", "Y188L", "Y188C", "G190A", "G190S",
              "H221Y", "P225H", "M230L"},
    "INSTI": {"T66I", "T66K", "E92Q", "G118R", "F121Y", "G140S", "G140A",
              "Q148H", "Q148K", "Q148R", "N155H", "R263K"},
}

ACCESSORY_MUTATIONS = {
    "PI": {"L10I", "L10V", "L10F", "V11I", "K20T", "K20R", "L24I", "L33F",
           "E35G", "M36I", "K43T", "F53L", "Q58E", "A71V", "A71T", "G73S",
           "L89V", "I93L"},
    "NRTI": {"E44D", "V75I", "F77L", "F116Y", "V118I", "E203K", "H208Y",
             "D218E", "K219E", "K219N", "K219R"},
    "NNRTI": {"A98G", "K101E", "K101P", "V179D", "V179F", "V179T"},
    "INSTI": {"L74M", "L74I", "Q95K", "T97A", "Y143R", "Y143C", "S147G",
              "S153Y", "E157Q", "G163R"},
}


def parse_mutation_list(mutation_str: str) -> set[str]:
    """
    Parse a mutation list string into a set of mutations.

    Args:
        mutation_str: Comma-separated mutation list like "D30N,M46I,L90M"

    Returns:
        Set of mutation strings
    """
    if not mutation_str or mutation_str == "NA" or pd.isna(mutation_str):
        return set()

    mutations = set()
    for part in mutation_str.replace(" ", "").split(","):
        part = part.strip()
        if part and part != "NA":
            mutations.add(part)

    return mutations


def build_resistance_graph(
    stanford_data: pd.DataFrame,
    drug_class: str,
    drug_column: str,
    resistance_threshold: float = 3.0,
) -> ResistanceGraph:
    """
    Build a resistance evolution graph from Stanford HIVDB data.

    Args:
        stanford_data: DataFrame with columns ['CompMutList', drug_column, ...]
        drug_class: Drug class name (PI, NRTI, NNRTI, INSTI)
        drug_column: Column name for resistance fold-change values
        resistance_threshold: Minimum fold-change to consider resistant

    Returns:
        ResistanceGraph object
    """
    graph = ResistanceGraph(drug_class=drug_class)

    # Add wild-type node (empty mutation set)
    wild_type = frozenset()
    graph.add_node(wild_type, resistance=1.0, frequency=1, fitness_cost=0.0)

    # Process each record
    for _, row in stanford_data.iterrows():
        mutation_str = row.get("CompMutList", "")
        resistance = row.get(drug_column, 1.0)

        if pd.isna(resistance) or resistance == "NA":
            continue

        try:
            resistance = float(resistance)
        except (ValueError, TypeError):
            continue

        mutations = parse_mutation_list(mutation_str)
        if not mutations:
            continue

        mutation_set = frozenset(mutations)

        # Estimate fitness cost based on mutation count and types
        fitness_cost = estimate_mutation_fitness(mutations, drug_class)

        # Add node for this mutation set
        graph.add_node(mutation_set, resistance=resistance, fitness_cost=fitness_cost)

        # Add edges from all possible predecessor states
        for mut in mutations:
            predecessor_muts = mutations - {mut}
            predecessor_set = frozenset(predecessor_muts)

            # Estimate edge weight (transition cost)
            is_primary = mut in PRIMARY_MUTATIONS.get(drug_class, set())
            edge_weight = 0.5 if is_primary else 1.0  # Primary mutations are favored

            # If predecessor exists or is wild-type, add edge
            if predecessor_set in graph.nodes or len(predecessor_muts) == 0:
                graph.add_edge(
                    source=predecessor_set if predecessor_muts else wild_type,
                    target=mutation_set,
                    mutation=mut,
                    weight=edge_weight,
                )

    return graph


def estimate_mutation_fitness(mutations: set[str], drug_class: str) -> float:
    """
    Estimate total fitness cost of a mutation set.

    Uses literature-based costs for known mutations.
    """
    total_cost = 0.0

    # Known fitness costs (replication capacity reduction)
    KNOWN_COSTS = {
        # PI mutations
        "D30N": 0.15,
        "M46I": 0.05,
        "M46L": 0.08,
        "I50V": 0.12,
        "I84V": 0.10,
        "L90M": 0.08,
        # NRTI mutations
        "M184V": 0.20,
        "K65R": 0.15,
        "T215Y": 0.05,
        "T215F": 0.08,
        "K70R": 0.10,
        # NNRTI mutations
        "K103N": 0.02,
        "Y181C": 0.05,
        "G190A": 0.10,
        # INSTI mutations
        "Q148H": 0.15,
        "N155H": 0.10,
        "R263K": 0.20,
    }

    for mut in mutations:
        if mut in KNOWN_COSTS:
            total_cost += KNOWN_COSTS[mut]
        elif mut in PRIMARY_MUTATIONS.get(drug_class, set()):
            total_cost += 0.10  # Default primary mutation cost
        elif mut in ACCESSORY_MUTATIONS.get(drug_class, set()):
            total_cost += 0.03  # Default accessory mutation cost
        else:
            total_cost += 0.05  # Unknown mutation

    return total_cost


def find_shortest_path_to_resistance(
    graph: ResistanceGraph,
    target_resistance: float = 10.0,
    weight_type: str = "fitness",
) -> list[ResistancePathway]:
    """
    Find shortest paths from wild-type to high resistance.

    Uses Dijkstra's algorithm with specified weight type.

    Args:
        graph: ResistanceGraph object
        target_resistance: Minimum resistance level to consider
        weight_type: 'fitness' (minimize fitness cost) or 'mutations' (minimize count)

    Returns:
        List of ResistancePathway objects
    """
    import heapq

    wild_type = frozenset()

    # Find all target nodes (resistance >= threshold)
    target_nodes = [
        node for node, data in graph.nodes.items()
        if data.resistance_level >= target_resistance
    ]

    if not target_nodes:
        return []

    # Dijkstra's algorithm
    distances = {wild_type: 0.0}
    predecessors = {wild_type: (None, None)}  # (prev_node, mutation)
    pq = [(0.0, wild_type)]
    visited = set()

    while pq:
        dist, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        for neighbor in graph.get_neighbors(current):
            edge = graph.get_edge(current, neighbor)
            if edge is None:
                continue

            if weight_type == "fitness":
                new_dist = dist + edge.weight
            else:  # mutations
                new_dist = dist + 1

            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = (current, edge.mutation_added)
                heapq.heappush(pq, (new_dist, neighbor))

    # Reconstruct paths to target nodes
    pathways = []

    for target in target_nodes:
        if target not in predecessors:
            continue

        # Backtrack to find path
        mutations = []
        current = target
        while predecessors[current][0] is not None:
            prev_node, mutation = predecessors[current]
            mutations.append(mutation)
            current = prev_node

        mutations.reverse()

        if not mutations:
            continue

        # Calculate pathway metrics
        total_fitness = graph.nodes[target].fitness_cost
        resistance = graph.nodes[target].resistance_level

        # Identify bottlenecks (high-cost transitions)
        bottlenecks = []
        temp_set = frozenset()
        for mut in mutations:
            edge = graph.get_edge(temp_set, frozenset(temp_set | {mut}))
            if edge and edge.weight > 0.7:  # High weight = bottleneck
                bottlenecks.append(mut)
            temp_set = frozenset(temp_set | {mut})

        pathway = ResistancePathway(
            mutations=mutations,
            total_fitness_cost=total_fitness,
            total_hyperbolic_distance=distances.get(target, 0),
            final_resistance=resistance,
            pathway_probability=np.exp(-total_fitness),  # Rough estimate
            bottleneck_mutations=bottlenecks,
        )
        pathways.append(pathway)

    # Sort by pathway probability
    pathways.sort(key=lambda p: p.pathway_probability, reverse=True)

    return pathways


def identify_resistance_bottlenecks(graph: ResistanceGraph) -> list[dict]:
    """
    Identify bottleneck mutations that are required for high-level resistance.

    A bottleneck is a mutation that appears in most high-resistance pathways
    and has high fitness cost.

    Returns:
        List of bottleneck mutation dictionaries with metrics
    """
    # Find all high-resistance nodes
    high_res_nodes = [
        node for node, data in graph.nodes.items()
        if data.resistance_level >= 10.0
    ]

    if not high_res_nodes:
        return []

    # Count mutation frequency in high-resistance states
    mutation_counts = defaultdict(int)
    total_high_res = len(high_res_nodes)

    for node in high_res_nodes:
        for mut in node:
            mutation_counts[mut] += 1

    # Identify bottlenecks
    bottlenecks = []
    primary_muts = PRIMARY_MUTATIONS.get(graph.drug_class, set())

    for mut, count in mutation_counts.items():
        frequency = count / total_high_res

        # Calculate average resistance when this mutation is present
        present_resistance = []
        absent_resistance = []

        for node, data in graph.nodes.items():
            if mut in node:
                present_resistance.append(data.resistance_level)
            else:
                absent_resistance.append(data.resistance_level)

        avg_with = np.mean(present_resistance) if present_resistance else 0
        avg_without = np.mean(absent_resistance) if absent_resistance else 0
        resistance_impact = avg_with - avg_without

        is_primary = mut in primary_muts

        # Bottleneck score combines frequency, impact, and primary status
        bottleneck_score = frequency * (1 + resistance_impact / 10) * (2 if is_primary else 1)

        if frequency > 0.5:  # Present in >50% of high-res cases
            bottlenecks.append({
                "mutation": mut,
                "frequency_in_resistant": frequency,
                "resistance_impact": resistance_impact,
                "is_primary": is_primary,
                "bottleneck_score": bottleneck_score,
            })

    # Sort by bottleneck score
    bottlenecks.sort(key=lambda x: x["bottleneck_score"], reverse=True)

    return bottlenecks


def calculate_pathway_diversity(graph: ResistanceGraph) -> dict:
    """
    Calculate diversity of resistance pathways.

    Measures:
    1. Number of distinct pathways
    2. Average pathway length
    3. Pathway convergence (how many lead to same endpoint)
    """
    wild_type = frozenset()

    # Find all high-resistance endpoints
    endpoints = [
        node for node, data in graph.nodes.items()
        if data.resistance_level >= 10.0
    ]

    # Count paths using DFS
    def count_paths(start: frozenset, end: frozenset, visited: set) -> int:
        if start == end:
            return 1
        if start in visited:
            return 0

        visited.add(start)
        total = 0
        for neighbor in graph.get_neighbors(start):
            total += count_paths(neighbor, end, visited.copy())
        return total

    pathway_counts = {}
    for endpoint in endpoints:
        n_paths = count_paths(wild_type, endpoint, set())
        pathway_counts[endpoint] = n_paths

    total_paths = sum(pathway_counts.values())
    avg_path_length = np.mean([len(ep) for ep in endpoints]) if endpoints else 0

    # Calculate convergence (entropy-based)
    if total_paths > 0:
        probs = np.array(list(pathway_counts.values())) / total_paths
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
    else:
        entropy = 0

    return {
        "total_pathways": total_paths,
        "distinct_endpoints": len(endpoints),
        "average_path_length": avg_path_length,
        "pathway_entropy": entropy,
        "most_common_endpoint": max(pathway_counts, key=pathway_counts.get, default=None),
    }


def integrate_hyperbolic_distances(
    graph: ResistanceGraph,
    hyperbolic_encoder,
    sequence_data: dict[frozenset, str],
) -> ResistanceGraph:
    """
    Add hyperbolic distance information to graph edges.

    Args:
        graph: ResistanceGraph to update
        hyperbolic_encoder: Function to encode sequence to hyperbolic space
        sequence_data: Mapping from mutation sets to sequences

    Returns:
        Updated ResistanceGraph
    """
    for (source, target), edge in graph.edges.items():
        # Get sequences for source and target
        source_seq = sequence_data.get(source, "")
        target_seq = sequence_data.get(target, "")

        if source_seq and target_seq:
            # Calculate hyperbolic distance
            try:
                source_emb = hyperbolic_encoder(source_seq)
                target_emb = hyperbolic_encoder(target_seq)
                distance = np.linalg.norm(target_emb - source_emb)
                edge.hyperbolic_distance = distance
            except Exception:
                edge.hyperbolic_distance = 0.0

    return graph


def visualize_graph_structure(graph: ResistanceGraph) -> str:
    """
    Generate ASCII visualization of graph structure.

    Returns:
        String representation of the graph
    """
    lines = [
        "=" * 60,
        f"RESISTANCE PATHWAY GRAPH: {graph.drug_class}",
        "=" * 60,
        f"Nodes: {len(graph.nodes)}",
        f"Edges: {len(graph.edges)}",
        "",
        "KEY NODES (by resistance level):",
        "-" * 40,
    ]

    # Sort nodes by resistance
    sorted_nodes = sorted(
        graph.nodes.items(),
        key=lambda x: x[1].resistance_level,
        reverse=True
    )[:10]

    for muts, node in sorted_nodes:
        mut_str = ",".join(sorted(muts)) if muts else "WILD-TYPE"
        lines.append(
            f"  {mut_str[:30]:30} | Resistance: {node.resistance_level:>8.1f}x | "
            f"Fitness cost: {node.fitness_cost:.2f}"
        )

    lines.extend([
        "",
        "MOST FREQUENT TRANSITIONS:",
        "-" * 40,
    ])

    # Sort edges by frequency
    sorted_edges = sorted(
        graph.edges.items(),
        key=lambda x: x[1].frequency,
        reverse=True
    )[:10]

    for (src, tgt), edge in sorted_edges:
        src_str = ",".join(sorted(src)) if src else "WT"
        lines.append(
            f"  {src_str[:20]:20} → +{edge.mutation_added:8} | "
            f"Freq: {edge.frequency:4} | Weight: {edge.weight:.2f}"
        )

    lines.append("=" * 60)

    return "\n".join(lines)


def export_to_networkx(graph: ResistanceGraph):
    """
    Export ResistanceGraph to NetworkX format for visualization.

    Returns:
        NetworkX DiGraph object (if networkx is available)
    """
    try:
        import networkx as nx

        G = nx.DiGraph()

        # Add nodes
        for mutations, node in graph.nodes.items():
            label = ",".join(sorted(mutations)) if mutations else "WT"
            G.add_node(
                label,
                resistance=node.resistance_level,
                fitness_cost=node.fitness_cost,
                frequency=node.frequency,
            )

        # Add edges
        for (src, tgt), edge in graph.edges.items():
            src_label = ",".join(sorted(src)) if src else "WT"
            tgt_label = ",".join(sorted(tgt)) if tgt else "WT"
            G.add_edge(
                src_label,
                tgt_label,
                mutation=edge.mutation_added,
                weight=edge.weight,
                frequency=edge.frequency,
            )

        return G
    except ImportError:
        print("NetworkX not available. Install with: pip install networkx")
        return None


def generate_pathway_report(graph: ResistanceGraph, drug_name: str = "") -> str:
    """
    Generate comprehensive pathway analysis report.

    Args:
        graph: ResistanceGraph object
        drug_name: Specific drug name for report title

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        f"RESISTANCE PATHWAY ANALYSIS: {drug_name or graph.drug_class}",
        "=" * 70,
        "",
    ]

    # Graph statistics
    lines.extend([
        "GRAPH STATISTICS:",
        f"  Total resistance states: {len(graph.nodes)}",
        f"  Total transitions: {len(graph.edges)}",
        "",
    ])

    # Find bottlenecks
    bottlenecks = identify_resistance_bottlenecks(graph)
    lines.extend([
        "BOTTLENECK MUTATIONS:",
        "-" * 40,
    ])

    for i, bn in enumerate(bottlenecks[:5], 1):
        primary_str = "PRIMARY" if bn["is_primary"] else "ACCESSORY"
        lines.append(
            f"  {i}. {bn['mutation']:8} | {primary_str:10} | "
            f"Freq: {bn['frequency_in_resistant']:.1%} | "
            f"Impact: +{bn['resistance_impact']:.1f}x"
        )

    # Find shortest paths
    pathways = find_shortest_path_to_resistance(graph, target_resistance=10.0)

    lines.extend([
        "",
        "SHORTEST PATHWAYS TO HIGH RESISTANCE (≥10x):",
        "-" * 40,
    ])

    for i, path in enumerate(pathways[:5], 1):
        mut_str = " → ".join(path.mutations)
        lines.append(
            f"  {i}. {mut_str}"
        )
        lines.append(
            f"     Final resistance: {path.final_resistance:.1f}x | "
            f"Fitness cost: {path.total_fitness_cost:.2f} | "
            f"P(path): {path.pathway_probability:.3f}"
        )

    # Pathway diversity
    diversity = calculate_pathway_diversity(graph)
    lines.extend([
        "",
        "PATHWAY DIVERSITY:",
        f"  Total distinct pathways: {diversity['total_pathways']}",
        f"  Distinct endpoints: {diversity['distinct_endpoints']}",
        f"  Average path length: {diversity['average_path_length']:.1f} mutations",
        f"  Pathway entropy: {diversity['pathway_entropy']:.2f} bits",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("Testing Resistance Pathway Graph Module")
    print("=" * 50)

    # Create synthetic Stanford HIVDB-like data
    np.random.seed(42)

    # Common PI mutation combinations
    mutation_combos = [
        ("", 1.0),
        ("M46I", 2.5),
        ("L90M", 3.0),
        ("M46I,L90M", 8.0),
        ("D30N", 5.0),
        ("D30N,M46I", 12.0),
        ("D30N,L90M", 15.0),
        ("D30N,M46I,L90M", 45.0),
        ("I84V", 4.0),
        ("I84V,L90M", 10.0),
        ("M46I,I84V", 9.0),
        ("M46I,I84V,L90M", 35.0),
        ("V82A", 3.5),
        ("V82A,M46I", 8.5),
        ("V82A,L90M", 11.0),
        ("V82A,M46I,L90M", 40.0),
    ]

    test_data = []
    for muts, resistance in mutation_combos:
        # Add multiple observations with some noise
        for _ in range(np.random.randint(5, 20)):
            test_data.append({
                "CompMutList": muts,
                "LPV": resistance * np.random.uniform(0.8, 1.2),
            })

    df = pd.DataFrame(test_data)

    # Build graph
    print("\nBuilding resistance graph...")
    graph = build_resistance_graph(df, drug_class="PI", drug_column="LPV")

    # Visualize structure
    print("\n" + visualize_graph_structure(graph))

    # Find bottlenecks
    print("\nIdentifying bottleneck mutations...")
    bottlenecks = identify_resistance_bottlenecks(graph)
    print(f"Found {len(bottlenecks)} bottleneck mutations")

    for bn in bottlenecks[:3]:
        print(f"  - {bn['mutation']}: score = {bn['bottleneck_score']:.2f}")

    # Find shortest paths
    print("\nFinding shortest paths to resistance...")
    pathways = find_shortest_path_to_resistance(graph, target_resistance=10.0)
    print(f"Found {len(pathways)} pathways to ≥10x resistance")

    for i, path in enumerate(pathways[:3], 1):
        print(f"  {i}. {' → '.join(path.mutations)} "
              f"(resistance: {path.final_resistance:.1f}x)")

    # Calculate diversity
    print("\nCalculating pathway diversity...")
    diversity = calculate_pathway_diversity(graph)
    print(f"  Total pathways: {diversity['total_pathways']}")
    print(f"  Pathway entropy: {diversity['pathway_entropy']:.2f} bits")

    # Generate full report
    print("\n" + generate_pathway_report(graph, drug_name="Lopinavir (LPV)"))

    # Test NetworkX export
    print("\nTesting NetworkX export...")
    nx_graph = export_to_networkx(graph)
    if nx_graph:
        print(f"  Exported to NetworkX: {nx_graph.number_of_nodes()} nodes, "
              f"{nx_graph.number_of_edges()} edges")

    print("\n" + "=" * 50)
    print("Module testing complete!")
