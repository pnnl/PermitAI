import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Set
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from collections import defaultdict
from scipy.spatial.distance import cdist
from topic_merge.predictor_types import LabeledCluster
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import pacmap


def generate_pacmap(embed, n_components=32, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, distance='euclidean'):
    reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio, distance=distance)
    return reducer.fit_transform(embed, init='pca')


def farthest_point_sampling(embeddings, N, metric='euclidean'):
    num_embeddings = embeddings.shape[0]
    if N >= num_embeddings:
        return np.arange(num_embeddings)

    if metric == 'cosine':
        distance_func = cosine_distances
    elif metric == 'euclidean':
        distance_func = euclidean_distances
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'.")

    mean_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    distances_from_mean = distance_func(embeddings, mean_embedding).flatten()
    farthest_idx = np.argmax(distances_from_mean)

    selected_indices = [farthest_idx]
    min_distances = distances_from_mean.copy()
    min_distances[farthest_idx] = 0

    for _ in range(N - 1):
        current_farthest_point = embeddings[selected_indices[-1]].reshape(1, -1)
        distances_to_current = distance_func(embeddings, current_farthest_point).flatten()
        min_distances = np.minimum(min_distances, distances_to_current)
        new_farthest_idx = np.argmax(min_distances)
        selected_indices.append(new_farthest_idx)
        min_distances[new_farthest_idx] = 0

    return np.array(selected_indices)


class UnionFind:
    def __init__(self, elements: List[int]):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> int:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                return root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                return root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                return root_i
        return root_i


def calculate_graph_edges(
    embeddings,
    labels,
    topk=3,
    distance_metric='euclidean',
    distance_threshold=None,
    graph_type='augmented_mst'
):
    """
    Calculates graph edges based on the specified graph_type.

    'rnn': Reciprocal Nearest Neighbor graph (often disconnected).
    'mst': Minimum Spanning Tree (guaranteed connected).
    'augmented_mst': Starts with the RNN graph and adds the cheapest MST edges to connect components.
    """
    if not isinstance(embeddings, np.ndarray): embeddings = np.array(embeddings)
    if not isinstance(labels, np.ndarray): labels = np.array(labels)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must match number of labels.")

    grouped_embeddings = defaultdict(list)
    for emb, label in zip(embeddings, labels):
        if label != -1: grouped_embeddings[label].append(emb)

    unique_labels = sorted(list(grouped_embeddings.keys()))
    if len(unique_labels) < 2:
        return unique_labels, np.array([]), {}, []

    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    centroids_array = np.array([np.mean(grouped_embeddings[label], axis=0) for label in unique_labels])
    distance_matrix = cdist(centroids_array, centroids_array, metric=distance_metric)

    if graph_type == 'mst':
        mst_sparse = minimum_spanning_tree(distance_matrix)
        mst_edges = []
        rows, cols = mst_sparse.nonzero()
        for i in range(len(rows)):
            u, v = unique_labels[rows[i]], unique_labels[cols[i]]
            weight = mst_sparse[rows[i], cols[i]]
            mst_edges.append((u, v, weight))
        return unique_labels, distance_matrix, {}, mst_edges

    topk_closest_clusters = {}
    for i, current_label in enumerate(unique_labels):
        distances_from_current = distance_matrix[i, :]
        potential_closest = []
        for j, other_label in enumerate(unique_labels):
            if i == j: continue
            dist = distances_from_current[j]
            if distance_threshold is None or dist < distance_threshold:
                potential_closest.append((dist, other_label))
        potential_closest.sort(key=lambda x: x[0])
        topk_closest_clusters[current_label] = potential_closest[:topk]

    neighbor_sets = {lbl: {n_id for _, n_id in ns} for lbl, ns in topk_closest_clusters.items()}
    rnn_edges = []
    processed_pairs = set()
    for cluster_a, neighbors_a in topk_closest_clusters.items():
        for _, cluster_b in neighbors_a:
            if cluster_b in neighbor_sets and cluster_a in neighbor_sets[cluster_b]:
                edge_pair = tuple(sorted((cluster_a, cluster_b)))
                if edge_pair not in processed_pairs:
                    dist = distance_matrix[label_to_index[edge_pair[0]], label_to_index[edge_pair[1]]]
                    rnn_edges.append((edge_pair[0], edge_pair[1], dist))
                    processed_pairs.add(edge_pair)

    if graph_type == 'rnn':
        return unique_labels, distance_matrix, topk_closest_clusters, rnn_edges

    elif graph_type == 'augmented_mst':
        adj_list = defaultdict(list)
        for u, v, _ in rnn_edges:
            adj_list[u].append(v)
            adj_list[v].append(u)

        visited = set()
        component_map = {}
        component_id_counter = 0

        def dfs(start_node, comp_id):
            stack = [start_node]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component_map[node] = comp_id
                    for neighbor in adj_list.get(node, []):
                        if neighbor not in visited:
                            stack.append(neighbor)

        for node in unique_labels:
            if node not in visited:
                dfs(node, component_id_counter)
                component_id_counter += 1

        if component_id_counter <= 1:
            return unique_labels, distance_matrix, {}, rnn_edges

        augmented_edges = list(rnn_edges)
        bridge_candidates = []
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                u, v = unique_labels[i], unique_labels[j]
                if component_map[u] != component_map[v]:
                    dist = distance_matrix[i, j]
                    bridge_candidates.append((dist, u, v))

        bridge_candidates.sort()
        for dist, u, v in bridge_candidates:
            comp_u = component_map[u]
            comp_v = component_map[v]
            if comp_u != comp_v:
                augmented_edges.append((u, v, dist))
                for node, comp_id in component_map.items():
                    if comp_id == comp_v:
                        component_map[node] = comp_u
                component_id_counter -= 1
                if component_id_counter == 1:
                    break

        return unique_labels, distance_matrix, {}, augmented_edges

    else:
        raise ValueError(f"Unknown graph_type: '{graph_type}'. Must be 'rnn', 'mst', or 'augmented_mst'.")


def get_context_clusters_from_embeddings(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
    cluster_id_a: int,
    cluster_id_b: int,
    num_local: int,
    num_global: int,
    metric: str = 'euclidean'
):
    if embeddings.shape[0] != cluster_assignments.shape[0]:
        raise ValueError("Embeddings and cluster_assignments must have the same number of documents.")

    unique_cluster_ids = np.unique(cluster_assignments)
    centroids_list = []
    id_to_idx = {}
    for i, cluster_id in enumerate(unique_cluster_ids):
        cluster_embeddings = embeddings[cluster_assignments == cluster_id]
        centroids_list.append(np.mean(cluster_embeddings, axis=0))
        id_to_idx[cluster_id] = i

    centroids_matrix = np.array(centroids_list)

    if cluster_id_a not in id_to_idx or cluster_id_b not in id_to_idx:
        raise ValueError("cluster_id_a or cluster_id_b not found in cluster_assignments.")

    if metric == 'cosine':
        dist_matrix = cosine_distances(centroids_matrix)
    elif metric == 'euclidean':
        dist_matrix = euclidean_distances(centroids_matrix)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'.")

    idx_a = id_to_idx[cluster_id_a]
    idx_b = id_to_idx[cluster_id_b]

    other_cluster_ids = [cid for cid in unique_cluster_ids if cid not in [cluster_id_a, cluster_id_b]]
    distances = []
    for other_id in other_cluster_ids:
        idx_other = id_to_idx[other_id]
        min_dist = min(dist_matrix[idx_a, idx_other], dist_matrix[idx_b, idx_other])
        distances.append((min_dist, other_id))

    distances.sort(key=lambda x: x[0])
    local_cluster_ids = {tup[1] for tup in distances[:num_local]}

    excluded_ids = {cluster_id_a, cluster_id_b}.union(local_cluster_ids)
    global_candidate_pool = [cid for cid in unique_cluster_ids if cid not in excluded_ids]

    if not global_candidate_pool or num_global <= 0:
        return local_cluster_ids, set()

    if len(global_candidate_pool) <= num_global:
        return local_cluster_ids, set(global_candidate_pool)

    reference_indices = {id_to_idx[cid] for cid in excluded_ids}
    candidate_indices = [id_to_idx[cid] for cid in global_candidate_pool]

    sampled_indices = set()
    min_dists = np.full(len(candidate_indices), np.inf)

    for ref_idx in reference_indices:
        for i, cand_idx in enumerate(candidate_indices):
            min_dists[i] = min(min_dists[i], dist_matrix[ref_idx, cand_idx])

    for _ in range(num_global):
        best = np.argmax(min_dists)
        sampled_indices.add(candidate_indices[best])
        new_ref_idx = candidate_indices[best]
        for i, cand_idx in enumerate(candidate_indices):
            min_dists[i] = min(min_dists[i], dist_matrix[new_ref_idx, cand_idx])
        min_dists[best] = -1

    idx_to_id = {v: k for k, v in id_to_idx.items()}
    global_cluster_ids = {idx_to_id[idx] for idx in sampled_indices}

    return local_cluster_ids, global_cluster_ids


class AggregatedClusterInfo:
    """
    Tracks state for a super-cluster (a group of merged original clusters).
    Maintains the current synthesized label and all original IDs merged into it.
    After each merge, relabels via farthest point sampling.
    """
    def __init__(self,
                 original_id: int,
                 text=None,
                 embeddings=None,
                 labeled_points=None,
                 batch_size=8,
                 initial_label: LabeledCluster = None,
                 distance_metric='euclidean'
                 ):
        self.member_original_ids: Set[int] = {original_id}
        self.text = text
        self.embeddings = embeddings
        self.labeled_points = labeled_points
        self.batch_size = batch_size
        self.distance_metric = distance_metric
        self.current_label: LabeledCluster = initial_label

    def add_original_members(self, other_member_ids: Set[int]):
        self.member_original_ids.update(other_member_ids)

    def to_labeled_cluster(self) -> LabeledCluster:
        return self.current_label

    def get_representative_texts(self) -> List[str]:
        mask = np.isin(np.array(self.labeled_points), np.array(list(self.member_original_ids)))
        if mask.sum() == 0:
            return []
        num_to_sample = min(self.batch_size, self.embeddings[mask].shape[0])
        points = farthest_point_sampling(self.embeddings[mask], num_to_sample, metric=self.distance_metric)
        return [self.text[mask].iloc[point].text[:1024] for point in points]

    def relabel(self, cluster_labeler, cluster_A, cluster_B, high_level_topics) -> LabeledCluster:
        representative_texts = self.get_representative_texts()
        if not representative_texts:
            return self.current_label
        cluster_label_output = cluster_labeler(texts=representative_texts, cluster_A=cluster_A, cluster_B=cluster_B, high_level_topic_examples=high_level_topics)
        self.current_label = LabeledCluster(
            cluster_label=cluster_label_output['label'],
            cluster_description=cluster_label_output['label_description']
        )
        return self.current_label

    def to_context_cluster(self, A_original_ids, B_original_ids) -> LabeledCluster:
        return self.current_label

    def get_unique_id_for_state(self) -> frozenset:
        return frozenset(sorted(list(self.member_original_ids)))

    def get_original_ids(self):
        return self.member_original_ids
