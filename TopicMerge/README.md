# TopicMerge: A Divide and Conquer Approach to Topic Modeling

**Matthew Raffel<sup>1,2</sup>, Lizhong Chen<sup>1</sup>, Sai Munikoti<sup>2</sup>, Sameera Horawalavithana<sup>2</sup>**  
<sup>1</sup>Oregon State University &nbsp;|&nbsp; <sup>2</sup>Pacific Northwest National Laboratory

> *TopicMerge is a multi-stage pipeline that bridges efficient embedding-based clustering with LLM-guided semantic merging to achieve state-of-the-art topic modeling while reducing required LLM calls by over 92% compared to document-level methods.*

The paper for TopicMerge can be found at https://aclanthology.org/2026.findings-emnlp.2248/.

---

## Overview

TopicMerge combines the global structural awareness of clustering with the local semantic reasoning of an LLM. The pipeline runs in three stages:

1. **Initial Clustering** — Documents are embedded (Gemini embedding model), normalized, and dimensionality-reduced with PaCMAP. Agglomerative clustering then deliberately *over-segments* the corpus into a large set of fine-grained, semantically pure micro-clusters. The over-segmentation ensures the LLM receives unambiguous concepts to evaluate in later stages.

2. **Initial Labeling** — A dedicated LLM module samples representative documents from each cluster using Farthest Point Sampling (FPS) and generates a concise label and description for each micro-cluster. These semantic representations serve as the primary input for the merging stage.

3. **LLM-Guided Merging** — An iterative algorithm consolidates micro-clusters into broader topics:
   - An augmented Reciprocal Nearest Neighbor (RNN) graph identifies plausible merge candidates efficiently, avoiding quadratic pairwise comparison.
   - For each candidate pair, an LLM evaluates the merge decision with rich context: local neighboring clusters, global representative clusters, and domain-level topic examples. Merges require two-fold LLM agreement (self-consistency).
   - Approved merges are ranked by a local silhouette-based purity score and executed as a maximal independent set (no cluster participates in more than one merge per iteration).
   - Each merged cluster is immediately relabeled by the LLM. The process repeats until no new merges are approved.

---

## Results

TopicMerge achieves state-of-the-art or competitive performance on both benchmark datasets while requiring far fewer LLM calls than document-level methods.

### Clustering quality

| Model | Bills P1 | Bills ARI | Bills NMI | Wiki P1 | Wiki ARI | Wiki NMI |
|---|---|---|---|---|---|---|
| LDA | 0.52 | 0.31 | 0.42 | 0.72 | 0.63 | 0.71 |
| NMF | 0.24 | 0.04 | 0.15 | 0.58 | 0.38 | 0.55 |
| CTM | 0.37 | 0.19 | 0.32 | 0.61 | 0.46 | 0.62 |
| FASTopic | 0.40 | 0.30 | 0.33 | 0.65 | 0.56 | 0.60 |
| EDTM | **0.58** | — | — | 0.65 | — | — |
| BERTopic | 0.41 | 0.18 | 0.37 | 0.62 | 0.50 | 0.59 |
| PromptTopic | 0.36 | 0.17 | 0.28 | 0.62 | 0.46 | 0.59 |
| TopicGPT | *0.57* | *0.40* | 0.49 | **0.74** | 0.60 | 0.70 |
| LiSA | *0.57* | *0.40* | 0.51 | **0.74** | *0.66* | 0.71 |
| **TopicMerge** (Ours) w/ 2.5 Flash-Lite | *0.57* | **0.42** | **0.53** | 0.72 | **0.71** | *0.73* |
| **TopicMerge** (Ours) w/ 3.1 Flash-Lite | *0.57* | **0.42** | *0.52* | *0.73* | **0.71** | **0.74** |

Best result in each column is **bold**; second-best is *italic*.

### LLM prompt efficiency

| Dataset | PromptTopic | TopicGPT | LiSA | TopicMerge |
|---|---|---|---|---|
| Wiki (15,242 docs) | 15,242+ | 15,242+ | 15,242+ | **562** |
| Bills (8,024 docs) | 8,024+ | 8,024+ | 8,024+ | **645** |

---

## Installation

```bash
pip install -r requirements.txt
```

A Google AI Studio API key is required. Either pass it via `--api_key` or set the `GOOGLE_API_KEY` environment variable.

---

## Usage

```bash
python -m topic_merge.main \
  --dataset_path       /path/to/dataset.jsonl \
  --dataset_out_path   /path/to/output.pq \
  --label_out_path     /path/to/labels.pkl \
  --api_key            YOUR_GOOGLE_API_KEY \
  --model_name         gemini-3.1-flash-lite \
  --embedding_model_name gemini-embedding-001 \
  --point_num_clusters 80 \
  --point_distance_metric cosine \
  --cluster_distance_metric cosine \
  --load_stage -1
```

### Key arguments

| Argument | Paper default | Description |
|---|---|---|
| `--api_key` | env `GOOGLE_API_KEY` | Google AI Studio API key |
| `--model_name` | `gemini-3.1-flash-lite` | LLM for labeling and merging (Table 1 setting) |
| `--embedding_model_name` | `gemini-embedding-001` | Embedding model (256-dim) |
| `--temperature` | `1.0` | Sampling temperature (Table 1 Gemini 3.1 setting; the 2.5 primary experiments used 0.0) |
| `--reasoning_head` | `predict` | `predict` or `cot` |
| `--point_num_clusters` | `80` | Initial cluster count |
| `--point_distance_metric` | `cosine` | Distance metric for initial clustering |
| `--cluster_distance_metric` | `cosine` | Distance metric for merge stage |
| `--labeler_batch_size` | `32` | Documents sampled per cluster (FPS) |
| `--cluster_topk` | `5` | Neighbors in the RNN graph |
| `--load_stage` | `-1` | Resume from stage N (1–4); `-1` runs all |

### Supported datasets

| Dataset | Expected columns | Path pattern |
|---|---|---|
| Bills (8,024 U.S. Congressional bill summaries, 21 topics) | `summary`, `topic` | path contains `bills` |
| Wiki (15,242 Wikipedia articles, 15 topics) | `text`, `supercategory` | path contains `wiki` |

Both datasets are derived from [ahoho/topics](https://github.com/ahoho/topics).

---

## Evaluation

After stages 2, 3, and 4 the pipeline prints NMI, ARI, Purity, and P1 (Amigó et al.) against the ground-truth `topic` column. Metrics can also be called independently:

```python
from topic_merge.metrics import get_metrics
get_metrics(true_labels, predicted_labels)
```

---

## Module reference

| Module | Contents |
|---|---|
| `model.py` | `EmbedderModule`, `ClusterModule`, `ClusterLabeler`, `ClusterMerger`, `ClusterPipeline` |
| `utils.py` | `UnionFind`, `AggregatedClusterInfo`, augmented MST / RNN graph, PaCMAP wrapper, FPS |
| `metrics.py` | `get_metrics`, `purity_score`, `calculate_f_measure_amigo_et_al` |
| `llm_utils.py` | Google AI Studio client factory, `get_lm`, `Extractor` |
| `signatures.py` | DSPy signatures for labeling and merging |
| `prompts.py` | System prompts for labeling, relabeling, and merging |
| `predictor_types.py` | Pydantic output types (`Cluster`, `LabeledCluster`, etc.) |

---

## Citation

```bibtex
@inproceedings{raffel-etal-2026-topicmerge,
    title = "{T}opic{M}erge: A Divide and Conquer Approach to Topic Modeling",
    author = "Raffel, Matthew  and
      Chen, Lizhong  and
      Munikoti, Sai  and
      Horawalavithana, Sameera",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.2248/",
    doi = "10.18653/2026.findings-emnlp.2248",
    pages = "48120--48135",
    ISBN = "979-8-89176-462-0",
    abstract = "Recent advancements in topic modeling have split into two main paths: efficient embedding-based methods that offer scalability but are often outperformed on standard clustering metrics, and Large Language Model (LLM) approaches that achieve state-of-the-art clustering performance but at the cost of prohibitive computational expense and poor scalability. This paper introduces *TopicMerge*, a novel multi-stage pipeline that bridges these approaches to achieve both state-of-the-art clustering performance and computational efficiency. TopicMerge first employs an efficient clustering pipeline to deliberately over-segment a document corpus into a large set of fine-grained, semantically pure micro-clusters, which are then given initial semantic labels. It then introduces a novel, iterative merging algorithm where an LLM, guided by local neighborhood context, arbitrates on consolidating candidate pairs. These approved merges are then selectively executed and semantically relabeled by the LLM, hierarchically building broader, more coherent topics. Extensive experiments on the Bills and Wiki datasets show that TopicMerge achieves state-of-the-art clustering performance while reducing required LLM calls by over 92{\%} compared to document-level methods, offering a practical and scalable solution."
}
```

---

## DISCLAIMER
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                PACIFIC NORTHWEST NATIONAL LABORATORY
                             operated by
                               BATTELLE
                               for the
                  UNITED STATES DEPARTMENT OF ENERGY
                   under Contract DE-AC05-76RL01830

---

## LICENSE
Copyright Battelle Memorial Institute 2026
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.