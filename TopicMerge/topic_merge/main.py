from topic_merge.model import ClusterPipeline
import os
import pandas as pd
from topic_merge.llm_utils import get_lm
import dspy
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process metadata documents to generate a clustered dataset using a language model."
    )
    parser.add_argument('--dataset_path', type=str, required=True,
        help="Path to the input JSONL file containing metadata documents.")
    parser.add_argument('--api_key', type=str, default=os.environ.get('GOOGLE_API_KEY'),
        help="Google AI Studio API key (or set GOOGLE_API_KEY env var).")
    parser.add_argument('--model_name', type=str, default="gemini-3.1-flash-lite",
        help="Gemini model name.")
    parser.add_argument('--embedding_model_name', type=str, default="gemini-embedding-001",
        help="Gemini embedding model name.")
    parser.add_argument('--temperature', type=float, default=1.0,
        help="Sampling temperature for the language model.")
    parser.add_argument('--reasoning_head', type=str, default="predict",
        choices=["cot", "predict"],
        help="The reasoning head.")
    parser.add_argument('--dataset_out_path', type=str, required=True,
        help="Path to the output Parquet file for the labeled dataset.")
    parser.add_argument('--label_out_path', type=str, required=True,
        help="Path to the output pickle file for the label mapping.")
    parser.add_argument('--labeler_batch_size', type=int, default=32,
        help="The batch size of the labeler.")
    parser.add_argument('--cluster_topk', type=int, default=5,
        help="The topk values to check for the merger.")
    parser.add_argument('--point_distance_threshold', type=float, default=None,
        help="The distance threshold for initial clustering. When set, it replaces --point_num_clusters.")
    parser.add_argument('--point_distance_metric', type=str, default="cosine")
    parser.add_argument('--point_num_clusters', type=int, default=80,
        help="Initial cluster count (ignored if --point_distance_threshold is set).")
    parser.add_argument('--cluster_distance_threshold', type=float, default=0.5,
        help="The distance threshold for the merger.")
    parser.add_argument('--cluster_distance_metric', type=str, default="cosine")
    parser.add_argument('--load_stage', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("Provide --api_key or set the GOOGLE_API_KEY environment variable.")

    dataset = pd.read_json(args.dataset_path, lines=True)

    if 'bills' in args.dataset_path:
        dataset = dataset.rename(columns={'summary': 'text'})
        args.dataset_name = 'bills'
    elif 'wiki' in args.dataset_path:
        dataset = dataset.rename(columns={'supercategory': 'topic'})
        args.dataset_name = 'wiki'

    lm = get_lm(model=args.model_name, api_key=args.api_key, temperature=args.temperature)
    dspy.settings.configure(lm=lm)

    cluster_pipeline = ClusterPipeline(lm, args)
    cluster_pipeline(dataset=dataset)


if __name__ == "__main__":
    main()
