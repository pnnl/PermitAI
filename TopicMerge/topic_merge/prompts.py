CLUSTER_RELABELING_PROMPT = """
You are an expert content analyst and text categorizer.

This cluster was formed by merging `cluster_A` and `cluster_B`. These prior labels indicate the full scope of content in this cluster. Use them to understand the breadth of what this cluster contains, but do NOT simply concatenate them. Your job is to synthesize a single unified label that naturally encompasses both, guided by the representative `texts` provided. You may generalize, refine, or completely rename if the texts suggest a more accurate characterization.

**Your Goal:** Propose a concise, descriptive topic label that accurately reflects the **dominant and most central theme** of the combined cluster. This label should capture the essence of the *majority* of the texts, allowing for the possibility that a small percentage might be outliers.

**Granularity Target:** The `high_level_topic_examples` are real topic labels from this dataset (though not exhaustive). Your label should be *at least* as specific as these examples — never broader. These represent the coarsest acceptable level; prefer more specific labels when the texts support it. If the cluster's dominant theme clearly falls under one of these examples, use that example as your label.

Each label must adhere to the following requirements:
1.  **Concise:** Be 1-3 words long.
2.  **Optimal Granularity & Specificity:** Represent the *most specific and relevant conceptual category* that accurately describes the primary topic covered by a majority of the provided texts. Your label must never be broader than the examples in `high_level_topic_examples` — if your label would subsume multiple of those examples, it is too broad.
3.  **Unified Coverage:** The label must naturally encompass the content from both `cluster_A` and `cluster_B`. Do not pick a label that only covers one source cluster while ignoring the other.
4.  **Singular Concept:** Each label should describe a single primary conceptual category, not a combination (e.g., "Medieval Architecture" is good; "Places and Institutions" is not).
5.  **Subject Matter, Not Format:** The label must describe *what the texts are about*, not how the texts are structured. Labels like "Biographies," "Lists," "Descriptions," "Events," or "Individuals" describe formats or containers — instead, identify the substantive domain (e.g., "American Politicians" instead of "Biographies"; "Gothic Cathedrals" instead of "Descriptions of Buildings").
6.  **Predictive of Domain:** Ask yourself: "If someone saw only this label, would they correctly predict the single academic field or subject area these texts belong to?" If the label could plausibly point to multiple unrelated fields, it is too vague.
7.  **Informed Synthesis:** Use `cluster_A` and `cluster_B` labels to understand scope, but let the representative `texts` ground your final decision. If the texts reveal that both source clusters are actually about the same specific thing, prefer that specific label over a forced generalization.

Topic Label Description Requirements:
1.  **Concise:** Be 1-2 sentences long.
2.  **Scope:** Clearly define the scope of the topic label, explaining how the content from both source clusters coheres under this single label.
3.  **General & Conceptual:** Focus on defining the label as a concept. Avoid using specific individuals, document identifiers, names, events, or narrow sub-topics from the provided texts as examples.
4.  **Tightly Focused:** The description must reinforce the same core domain as the label. Do not mention secondary themes or tangential content that only applies to a minority of texts.

Do not include:
*   Specific individuals, organizations, document identifiers, project names, or specific titles from the provided texts.
*   Overly obscure specialized jargon, unless that jargon *is* the precise conceptual category itself.
*   Secondary or tangential themes that only apply to outlier texts.
*   Any conversational remarks or information beyond the requested label and its description.
"""

CLUSTER_LABELING_PROMPT = f"""
You are an expert content analyst and text categorizer.

Your goal is to propose a concise, descriptive topic label that accurately reflects the **dominant and most central theme** of the cluster. This label should capture the essence of the *majority* of the summaries, allowing for the possibility that a small percentage of summaries might be outliers that do not perfectly align.
**Granularity Target:** The following are examples of the level of specificity your labels should aim for or exceed: `high_level_topic_examples`. Your label should be *at least* as specific as these examples — never broader. These represent the coarsest acceptable level; prefer more specific labels when the texts support it. If the cluster's dominant theme clearly falls under one of these examples, use that example as your label.

Each label must adhere to the following requirements:
1.  **Concise:** Be 1-3 words long.
2.  **Optimal Granularity & Specificity:** Represent the *most specific and relevant conceptual category* that accurately describes the primary topic covered by a majority of the provided summaries. Prioritize specificity for the core theme over attempting to encompass every single outlier. Your label must never be broader than the examples in the granularity target above — if your label would subsume multiple of those examples, it is too broad.
3.  **Applicable to Dominant Theme:** The label must accurately and logically apply to the *dominant theme* represented by the majority of summaries in the batch. It is acceptable if a small percentage of outlier summaries do not perfectly fit this label.
4.  **Singular Concept:** Each label should describe a single primary conceptual category, not a combination (e.g., "Medieval Architecture" is good; "Places and Institutions" is not).
5.  **Subject Matter, Not Format:** The label must describe *what the texts are about*, not how the texts are structured. Labels like "Biographies," "Lists," "Descriptions," "Events," or "Individuals" describe formats or containers — instead, identify the substantive domain (e.g., "American Politicians" instead of "Biographies and Individuals"; "Gothic Cathedrals" instead of "Descriptions of Buildings").
6.  **Predictive of Domain:** Ask yourself: "If someone saw only this label, would they correctly predict the single academic field or subject area these texts belong to?" If the label could plausibly point to multiple unrelated fields, it is too vague or misleading.

Topic Label Description Requirements:
1.  **Concise:** Be 1-2 sentences long.
2.  **Scope:** Clearly define the scope of the topic label, explaining the specific conceptual area or field of activity it represents. This description should reflect the primary theme identified for the majority of the cluster members.
3.  **General & Conceptual:** Focus on defining the label as a concept. Avoid using specific individuals, document identifiers, names, events, or narrow sub-topics *from the provided summaries* as examples within the description itself.
4.  **Tightly Focused:** The description must reinforce the same core domain as the label. Do not mention secondary themes, tangential content, or adjacent fields that only apply to a minority of texts — this creates misleading associations.

Do not include:
*   Specific individuals, organizations, document identifiers, project names, or specific titles from the provided text *as examples or illustrative details within your description*.
*   Overly obscure specialized jargon or technical terms, unless that jargon *is* the precise conceptual category itself.
*   Secondary or tangential themes that only apply to outlier texts in the cluster.
*   Any conversational remarks or information beyond the requested label and its description.
"""



CLUSTER_MERGING_PROMPT = """
You are deciding whether `Cluster_A` and `Cluster_B` should be merged into a single cluster.

**Goal:** Consolidate fragments of the same topic while keeping genuinely different topics apart.

**Calibration:**
Look at `high_level_topic_examples` — these are real topic labels from this dataset (though not exhaustive). They define the TARGET granularity level your clusters should reach. Your merged clusters should be roughly at this level of breadth. Use `local_context_clusters` and `global_context_clusters` to understand the broader landscape of topics in this dataset.

**MERGE if any of these clearly apply:**
1. The clusters are about the same core topic or one is a clearly a subset/specialization of the other.
2. The clusters are redundant splits of the same concept.
3. Both clusters are sub-topics that would naturally fall under a single entry at the granularity level shown by `high_level_topic_examples`.
4. Both clusters share the same subject domain AND the same type of content.

**Do NOT merge if any of these clearly apply:**
- The clusters would belong to DIFFERENT entries at the granularity level shown by `high_level_topic_examples`.
- The clusters represent fundamentally different real-world domains (e.g., biology vs. economics).
- One cluster is coherent and the other is noise — merging would dilute it.

**Final Decision:** Return True for a merge if the evidence for it is compelling and False otherwise.
"""