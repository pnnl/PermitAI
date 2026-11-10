import dspy
from topic_merge.signatures import get_cluster_labeler, get_cluster_merger, get_cluster_relabeler
from topic_merge.llm_utils import get_client
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import os
import pandas as pd
from google.genai import types
from typing import List, Dict, Tuple
import pickle
import copy
from tqdm.auto import tqdm
import time
from sklearn.metrics import silhouette_score
from topic_merge.predictor_types import LabeledCluster
from topic_merge.metrics import get_metrics
from topic_merge.utils import UnionFind, AggregatedClusterInfo, get_context_clusters_from_embeddings, calculate_graph_edges
from topic_merge.utils import farthest_point_sampling, generate_pacmap

BILLS_EXAMPLE_TOPICS = ['Public Lands', 'Domestic Commerce']
WIKI_EXAMPLE_TOPICS = ['Sports', 'Media and drama', 'Video Games']


class EmbedderModule(dspy.Module):
    def __init__(self, model_name="gemini-embedding-001", api_key=None):
        self.embedding_model = get_client(api_key).models
        self.model_name = model_name

    def forward(self, texts, task_type):
        llm_embeddings = []
        for i in tqdm(range(0, len(texts), 100), desc="Generating Embeddings"):
            batch = texts[i:i+100]
            wait_time = 2
            last_exc = None
            for _ in range(3):
                try:
                    resp = self.embedding_model.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=256,
                        )
                    )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    time.sleep(wait_time)
                    wait_time *= 2
            if last_exc is not None:
                raise last_exc
            for embedding_obj in resp.embeddings:
                llm_embeddings.append(np.array(embedding_obj.values, dtype=np.float32))

        llm_embeddings = np.vstack(llm_embeddings)
        llm_embeddings /= np.linalg.norm(llm_embeddings, axis=1, keepdims=True)
        return llm_embeddings


class ClusterModule(dspy.Module):
    def __init__(self, num_clusters=None, metric='euclidean', linkage='ward', distance_threshold=None):
        self.cluster_algorithm = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage=linkage,
            metric=metric,
            distance_threshold=distance_threshold,
        )

    def forward(self, embedding):
        clusters = self.cluster_algorithm.fit_predict(embedding)
        return clusters


class ClusterMerger(dspy.Module):
    def __init__(self, lm, cluster_merging_prompt, cluster_labeling_prompt, cluster_distance_metric, topk, batch_size=8, dataset='bills', reasoning_head='predict'):
        super().__init__()
        self.cluster_merger = get_cluster_merger(cluster_merging_prompt)
        self.cluster_labeler = get_cluster_relabeler(cluster_labeling_prompt, reasoning_head=reasoning_head)
        self.cluster_distance_metric = cluster_distance_metric
        self.topk = topk
        self.lm = lm
        self.batch_size = batch_size
        if dataset == 'bills':
            self.high_level_topics = BILLS_EXAMPLE_TOPICS
        elif dataset == 'wiki':
            self.high_level_topics = WIKI_EXAMPLE_TOPICS

    def forward(
        self,
        labeled_points: np.array,
        text,
        embeddings,
        cluster_text_mapping,
        cluster_threshold,
        topk
    ) -> Tuple[Dict[int, List[int]], np.ndarray]:
        def purity_local_silhouette(embeddings: np.ndarray, labels: np.ndarray, cluster_id_A: int, cluster_id_B: int) -> float:
            indices_A = np.where(labels == cluster_id_A)[0]
            indices_B = np.where(labels == cluster_id_B)[0]
            subset_indices = np.concatenate([indices_A, indices_B])
            if len(indices_A) < 2 or len(indices_B) < 2:
                return 0.0
            subset_embeddings = embeddings[subset_indices]
            subset_labels = np.zeros(len(subset_indices), dtype=int)
            subset_labels[len(indices_A):] = 1
            separation_score = silhouette_score(subset_embeddings, subset_labels, metric='cosine')
            return (1.0 - separation_score) / 2.0

        active_cluster_ids = np.unique(labeled_points).tolist()
        uf = UnionFind(active_cluster_ids)
        next_uf = copy.deepcopy(uf)

        current_aggregated_info: Dict[int, AggregatedClusterInfo] = {}
        for cluster_id in active_cluster_ids:
            initial_label = cluster_text_mapping.get(cluster_id, None)
            current_aggregated_info[cluster_id] = AggregatedClusterInfo(
                original_id=cluster_id,
                text=text,
                embeddings=embeddings,
                labeled_points=labeled_points,
                batch_size=self.batch_size,
                initial_label=initial_label,
                distance_metric=self.cluster_distance_metric
            )

        next_aggregated_info = copy.deepcopy(current_aggregated_info)
        llm_decision_cache: Dict[frozenset[frozenset[int]], bool] = {}

        iteration_flag = True
        clusters = labeled_points
        while iteration_flag:
            iteration_flag = False
            _, D1, k_membership_dict, k_membership = calculate_graph_edges(
                embeddings, clusters, topk=topk,
                distance_metric=self.cluster_distance_metric,
                distance_threshold=cluster_threshold, graph_type='augmented_mst'
            )
            k_membership.sort(key=lambda x: x[-1])

            merge_candidates = []
            for pair in k_membership:
                A = pair[0]
                B = pair[1]

                root_A_current = uf.find(A)
                root_B_current = uf.find(B)
                assert A == root_A_current
                assert B == root_B_current

                candidate_key = frozenset([root_A_current, root_B_current])
                if any(frozenset([c['root_A'], c['root_B']]) == candidate_key for c in merge_candidates):
                    continue

                local_context_clusters = []
                global_context_clusters = []
                local_cluster_ids, global_cluster_ids = get_context_clusters_from_embeddings(
                    embeddings, clusters, root_A_current, root_B_current, 3, 5, metric=self.cluster_distance_metric
                )
                for id in local_cluster_ids:
                    context_label = current_aggregated_info[id].to_context_cluster(
                        current_aggregated_info[A].get_original_ids(),
                        current_aggregated_info[B].get_original_ids()
                    )
                    local_context_clusters.append(context_label)
                for id in global_cluster_ids:
                    context_label = current_aggregated_info[id].to_context_cluster(
                        current_aggregated_info[A].get_original_ids(),
                        current_aggregated_info[B].get_original_ids()
                    )
                    global_context_clusters.append(context_label)

                if root_A_current not in current_aggregated_info or root_B_current not in current_aggregated_info:
                    continue
                if root_A_current == root_B_current:
                    continue

                aggregated_A_info = current_aggregated_info[root_A_current]
                aggregated_B_info = current_aggregated_info[root_B_current]
                state_A_id = aggregated_A_info.get_unique_id_for_state()
                state_B_id = aggregated_B_info.get_unique_id_for_state()
                context_state_id = frozenset(
                    current_aggregated_info[cid].get_unique_id_for_state()
                    for cid in list(local_cluster_ids)
                    if cid in current_aggregated_info
                )

                cache_key = (frozenset([state_A_id, state_B_id]), context_state_id)
                should_merge = False
                if cache_key in llm_decision_cache:
                    should_merge = llm_decision_cache[cache_key]
                else:
                    cluster_A_label = aggregated_A_info.to_labeled_cluster()
                    cluster_B_label = aggregated_B_info.to_labeled_cluster()

                    try:
                        is_merge_output = self.cluster_merger(
                            cluster_A=cluster_A_label,
                            cluster_B=cluster_B_label,
                            local_context_clusters=local_context_clusters,
                            global_context_clusters=global_context_clusters,
                            high_level_topic_examples=self.high_level_topics
                        )
                        should_merge = bool(is_merge_output.is_merge)
                        if should_merge:
                            is_merge_output = self.cluster_merger(
                                cluster_A=cluster_A_label,
                                cluster_B=cluster_B_label,
                                local_context_clusters=local_context_clusters,
                                global_context_clusters=global_context_clusters,
                                high_level_topic_examples=self.high_level_topics
                            )
                            should_merge = bool(is_merge_output.is_merge)
                        llm_decision_cache[cache_key] = should_merge
                    except Exception as e:
                        should_merge = False

                if should_merge:
                    purity_score = purity_local_silhouette(embeddings, clusters, root_A_current, root_B_current)
                    merge_candidates.append({
                        "root_A": root_A_current,
                        "root_B": root_B_current,
                        "score": purity_score
                    })

            if not merge_candidates:
                iteration_flag = False
            else:
                merge_candidates.sort(key=lambda x: x['score'], reverse=True)
                merges_to_execute = []
                clusters_already_used = set()
                iteration_flag = True

                for candidate in merge_candidates:
                    root_A = candidate['root_A']
                    root_B = candidate['root_B']
                    if root_A not in clusters_already_used and root_B not in clusters_already_used:
                        merges_to_execute.append((root_A, root_B))
                        clusters_already_used.add(root_A)
                        clusters_already_used.add(root_B)

            if not iteration_flag:
                break

            merged_roots_to_relabel = {}

            for root_A_to_merge, root_B_to_merge in merges_to_execute:
                current_merged_root_A = next_uf.find(root_A_to_merge)
                current_merged_root_B = next_uf.find(root_B_to_merge)
                if current_merged_root_A == current_merged_root_B:
                    continue

                label_A = next_aggregated_info[current_merged_root_A].current_label
                label_B = next_aggregated_info[current_merged_root_B].current_label

                new_root = next_uf.union(current_merged_root_A, current_merged_root_B)
                absorbed_root = current_merged_root_A if new_root == current_merged_root_B else current_merged_root_B

                if new_root in next_aggregated_info and absorbed_root in next_aggregated_info:
                    new_root_agg_info = next_aggregated_info[new_root]
                    absorbed_agg_info = next_aggregated_info[absorbed_root]
                    new_root_agg_info.add_original_members(absorbed_agg_info.member_original_ids)
                    del next_aggregated_info[absorbed_root]
                    merged_roots_to_relabel[new_root] = (label_A, label_B)

            for root_id, (cluster_A, cluster_B) in merged_roots_to_relabel.items():
                if root_id in next_aggregated_info:
                    next_aggregated_info[root_id].relabel(self.cluster_labeler, cluster_A, cluster_B, self.high_level_topics)

            uf = copy.deepcopy(next_uf)
            current_aggregated_info = copy.deepcopy(next_aggregated_info)

            final_roots_for_cluster_relabeling = sorted([r for r in uf.parent if uf.parent[r] == r])
            root_to_new_label_map = {root: i for i, root in enumerate(final_roots_for_cluster_relabeling)}
            new_clusters_array = np.full(len(labeled_points), -1, dtype=int)
            for i, initial_id_of_point in enumerate(labeled_points):
                if initial_id_of_point != -1:
                    current_super_cluster_root = uf.find(initial_id_of_point)
                    if current_super_cluster_root in root_to_new_label_map:
                        new_clusters_array[i] = current_super_cluster_root
            clusters = new_clusters_array

        final_point_labels_raw = clusters
        final_roots = sorted([r for r in uf.parent if uf.parent[r] == r])
        final_root_to_new_label_map = {root: i for i, root in enumerate(final_roots)}
        final_point_labels = np.full(len(labeled_points), -1, dtype=int)
        for i, uf_root_id in enumerate(final_point_labels_raw):
            if uf_root_id != -1:
                final_point_labels[i] = final_root_to_new_label_map.get(uf_root_id, -1)

        cluster_label_mapping: Dict[int, Dict[str, str]] = {}
        for root_id, new_label in final_root_to_new_label_map.items():
            if root_id in current_aggregated_info:
                agg_info = current_aggregated_info[root_id]
                labeled_cluster = agg_info.current_label
                cluster_label_mapping[new_label] = {
                    "label": labeled_cluster.cluster_label if labeled_cluster else None,
                    "description": labeled_cluster.cluster_description if labeled_cluster and hasattr(labeled_cluster, 'cluster_description') else None,
                }
            else:
                cluster_label_mapping[new_label] = {"label": None, "description": None}

        return cluster_label_mapping, final_point_labels


class ClusterLabeler(dspy.Module):
    def __init__(self, lm, prompt, batch_size=8, cluster_distance_metric='euclidean', dataset='bills', reasoning_head='predict'):
        self.cluster_labeler = get_cluster_labeler(prompt, reasoning_head)
        self.batch_size = batch_size
        self.cluster_distance_metric = cluster_distance_metric
        self.lm = lm
        if dataset == 'bills':
            self.high_level_topics = BILLS_EXAMPLE_TOPICS
        elif dataset == 'wiki':
            self.high_level_topics = WIKI_EXAMPLE_TOPICS

    def forward(self, texts, labeled_points, embeddings):
        cluster_id_mapping = defaultdict(list)
        for cluster_id in np.unique(labeled_points):
            mask = np.array(labeled_points) == cluster_id
            num_to_sample = min(self.batch_size, embeddings[mask].shape[0])
            points = farthest_point_sampling(embeddings[mask], num_to_sample, metric=self.cluster_distance_metric)
            for point in points:
                cluster_id_mapping[cluster_id].append(texts[mask].iloc[point].text)

        cluster_label_mapping = defaultdict()
        for cluster_id in cluster_id_mapping.keys():
            text_input = [text[:1024] for text in cluster_id_mapping[cluster_id]]
            cluster_label = self.cluster_labeler(texts=text_input, high_level_topic_examples=self.high_level_topics)
            cluster_label_mapping[cluster_id] = LabeledCluster(cluster_label=cluster_label['label'], cluster_description=cluster_label['label_description'])
        return cluster_label_mapping, cluster_id_mapping


class ClusterPipeline(dspy.Module):
    def __init__(self, lm, args):
        self.dataset_path = args.dataset_path.lower()
        from topic_merge.prompts import CLUSTER_LABELING_PROMPT, CLUSTER_MERGING_PROMPT, CLUSTER_RELABELING_PROMPT

        self.cluster_labeler = ClusterLabeler(lm, prompt=CLUSTER_LABELING_PROMPT, batch_size=args.labeler_batch_size, cluster_distance_metric=args.cluster_distance_metric, dataset=args.dataset_name, reasoning_head=args.reasoning_head)
        self.cluster_merger = ClusterMerger(lm, cluster_merging_prompt=CLUSTER_MERGING_PROMPT, cluster_labeling_prompt=CLUSTER_RELABELING_PROMPT, cluster_distance_metric=args.cluster_distance_metric, topk=args.cluster_topk, batch_size=args.labeler_batch_size, dataset=args.dataset_name, reasoning_head=args.reasoning_head)
        self.embedder_module = EmbedderModule(model_name=args.embedding_model_name, api_key=args.api_key)
        use_threshold = args.point_distance_threshold is not None
        self.cluster_module = ClusterModule(
            metric=args.point_distance_metric,
            num_clusters=None if use_threshold else args.point_num_clusters,
            linkage='complete',
            distance_threshold=args.point_distance_threshold,
        )
        self.dataset_out_path = args.dataset_out_path
        self.label_dataset_out_path = args.label_out_path
        self.dataset_name = args.dataset_name
        self.load_stage = args.load_stage

    def save_df(self, df, stage):
        path_list = self.dataset_out_path.split('/')
        path_list.insert(-1, stage)
        path = '/'.join(path_list)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)

    def load_df(self, stage):
        path_list = self.dataset_out_path.split('/')
        path_list.insert(-1, stage)
        path = '/'.join(path_list)
        return pd.read_parquet(path)

    def save_json(self, dictionary, stage):
        path_list = self.label_dataset_out_path.split('/')
        path_list.insert(-1, stage)
        path = '/'.join(path_list)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(dictionary, f)

    def load_json(self, stage):
        path_list = self.label_dataset_out_path.split('/')
        path_list.insert(-1, stage)
        path = '/'.join(path_list)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def forward(self, dataset):
        if self.load_stage >= 1:
            dataset = self.load_df(stage='first')
            llm_embeddings_full = np.array(list(dataset['llm_embeddings_full']))
        else:
            llm_embeddings_full = self.embedder_module(dataset['text'].to_list(), task_type="CLUSTERING")
            dataset['llm_embeddings_full'] = list(llm_embeddings_full)
            self.save_df(dataset, stage='first')

        if self.load_stage >= 2:
            dataset = self.load_df(stage='second')
            llm_embeddings = np.array(list(dataset['llm_embeddings']))
            clusters = np.array(list(dataset['initial_clusters']))
        else:
            llm_embeddings = generate_pacmap(llm_embeddings_full, n_components=64, distance='angular')
            clusters = self.cluster_module(llm_embeddings)
            dataset['llm_embeddings'] = list(llm_embeddings)
            dataset['initial_clusters'] = clusters
            self.save_df(dataset, stage='second')

        get_metrics(list(dataset['topic']), clusters)

        if self.load_stage >= 3:
            cluster_label_mapping = self.load_json(stage='third')
        else:
            cluster_label_mapping, _ = self.cluster_labeler(dataset[["text"]], clusters, llm_embeddings_full)
            self.save_json(cluster_label_mapping, stage='third')

        cluster_sub_label_map = {}
        for id in cluster_label_mapping:
            cluster_sub_label_map[id] = cluster_label_mapping[id].cluster_label
        dataset['label_ids'] = dataset['initial_clusters'].map(cluster_sub_label_map)
        get_metrics(list(dataset['topic']), clusters)

        if self.load_stage >= 4:
            dataset = self.load_df(stage='fourth')
            cluster_label_mapping = self.load_json(stage='fourth')
            final_clusters = np.array(list(dataset['final_clusters']))
        else:
            cluster_label_mapping, final_clusters = self.cluster_merger(clusters, dataset[["text"]], llm_embeddings_full, cluster_label_mapping, cluster_threshold=100, topk=5)
            dataset['final_clusters'] = final_clusters
            self.save_df(dataset, stage='fourth')
            self.save_json(cluster_label_mapping, stage='fourth')

        get_metrics(list(dataset['topic']), final_clusters)
        return dataset
