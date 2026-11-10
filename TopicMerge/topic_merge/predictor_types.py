from pydantic import BaseModel, Field

class Cluster(BaseModel):
    cluster_points: list[str] = Field(
        "The bill summaries in the cluster."
    )

class LabeledCluster(BaseModel):
    cluster_label: str = Field(
        "The cluster label."
    )
    cluster_description: str = Field(
        "The cluster label description."
    )


class SuperLabeledCluster(BaseModel):
    sub_clusters: list[LabeledCluster] = Field(
        "The sub clusters that have been merged into the super cluster."
    )



class ContextualCluster(BaseModel):
    """Holds a context cluster along with its distances to the primary clusters A and B."""
    cluster: SuperLabeledCluster 
    distance_to_A: float
    distance_to_B: float