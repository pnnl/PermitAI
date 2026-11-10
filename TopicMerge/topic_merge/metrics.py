from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from collections import Counter


def purity_score(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    unique_predicted_clusters = np.unique(predicted_labels)
    total_correct_hits = 0
    for cluster_id in unique_predicted_clusters:
        corresponding_true_labels = true_labels[predicted_labels == cluster_id]
        if corresponding_true_labels.size > 0:
            counts = np.bincount(corresponding_true_labels)
            total_correct_hits += np.max(counts)
    return total_correct_hits / len(true_labels)


def calculate_f_measure_amigo_et_al(true_labels, predicted_clusters):
    true_labels = np.asarray(true_labels)
    predicted_clusters = np.asarray(predicted_clusters)

    if len(true_labels) != len(predicted_clusters):
        raise ValueError("true_labels and predicted_clusters must have the same length.")
    if len(true_labels) == 0:
        return 0.0

    N = len(true_labels)
    unique_true_classes = np.unique(true_labels)
    unique_predicted_clusters = np.unique(predicted_clusters)
    sum_weighted_max_f_per_class = 0.0

    for class_id in unique_true_classes:
        true_class_mask = (true_labels == class_id)
        size_Li = np.sum(true_class_mask)
        if size_Li == 0:
            continue
        max_f_for_current_class = 0.0
        for cluster_id in unique_predicted_clusters:
            predicted_cluster_mask = (predicted_clusters == cluster_id)
            size_Cj = np.sum(predicted_cluster_mask)
            intersection_size = np.sum(true_class_mask & predicted_cluster_mask)
            precision = intersection_size / size_Cj if size_Cj > 0 else 0.0
            recall = intersection_size / size_Li if size_Li > 0 else 0.0
            f_score_Li_Cj = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            max_f_for_current_class = max(max_f_for_current_class, f_score_Li_Cj)
        sum_weighted_max_f_per_class += (size_Li / N) * max_f_for_current_class

    return sum_weighted_max_f_per_class


def get_metrics(topic_list, predicted):
    topic_set = set(topic_list)
    topic_map = {topic: i for i, topic in enumerate(topic_set)}
    num_labels = np.array([topic_map[topic] for topic in topic_list])

    mask = predicted != -1
    filtered_num_labels = num_labels[mask]
    filtered_predicted = predicted[mask]
    NMI = normalized_mutual_info_score(filtered_num_labels, filtered_predicted)
    ARI = adjusted_rand_score(filtered_num_labels, filtered_predicted)
    P = purity_score(filtered_num_labels, filtered_predicted)
    P1 = calculate_f_measure_amigo_et_al(filtered_num_labels, filtered_predicted)

    print(f"NMI: {NMI}, ARI: {ARI}, Purity: {P}, P1: {P1}")
    return {"NMI": NMI, "ARI": ARI, "P": P, "P1": P1}
