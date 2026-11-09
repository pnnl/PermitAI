import dspy
from topic_merge.predictor_types import LabeledCluster, SuperLabeledCluster
from topic_merge.llm_utils import Extractor

class DocumentClusterLabeler(dspy.Signature):
    texts: list[str] = dspy.InputField(desc="A list of summaries of texts.")
    high_level_topic_examples: list[str] = dspy.InputField(desc="Examples of ground-truth cluster labels.")
    label: str = dspy.OutputField(desc="The topic label of texts")
    label_description: str = dspy.OutputField(desc="The description of the topic label for the texts.")

def get_cluster_labeler(prompt, reasoning_head):
    DocumentClusterLabeler.__doc__ = prompt
    
    refine = Extractor(signature=DocumentClusterLabeler, reasoning_head=reasoning_head, metadata_name='labeler',)
    return refine


class DocumentClusterReLabeler(dspy.Signature):
    texts: list[str] = dspy.InputField(desc="A list of summaries of texts.")
    cluster_A: LabeledCluster = dspy.InputField(desc="Cluster A.")
    cluster_B: LabeledCluster = dspy.InputField(desc="Cluster B.")
    high_level_topic_examples: list[str] = dspy.InputField(desc="Examples of ground-truth cluster labels.")
    label: str = dspy.OutputField(desc="The topic label of texts")
    label_description: str = dspy.OutputField(desc="The description of the topic label for the texts.")

def get_cluster_relabeler(prompt, reasoning_head):
    DocumentClusterReLabeler.__doc__ = prompt
    
    refine = Extractor(signature=DocumentClusterReLabeler, reasoning_head=reasoning_head, metadata_name='labeler')
    return refine

class DocumentMerger(dspy.Signature):
    cluster_A: LabeledCluster = dspy.InputField(desc="Cluster A.")
    cluster_B: LabeledCluster = dspy.InputField(desc="Cluster B.")
    local_context_clusters: list[LabeledCluster] = dspy.InputField(desc="A collection of clusters that are near cluster A and B")
    global_context_clusters: list[LabeledCluster] = dspy.InputField(desc="A collection of clusters that are far from cluster A and B")
    high_level_topic_examples: list[str] = dspy.InputField(desc="Examples of ground-truth cluster labels.")
    reasoning: str = dspy.OutputField(desc="The reasoning for the merging choice.")
    is_merge: bool = dspy.OutputField(desc="The merging choice.")

def get_cluster_merger(prompt):
    DocumentMerger.__doc__ = prompt
    
    refine = Extractor(signature=DocumentMerger, reasoning_head='predict', metadata_name='merger')
    return refine