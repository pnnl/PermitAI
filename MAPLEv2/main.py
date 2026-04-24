def main():

    # Argument parser
    import sys
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configPath",
        help = "path to the configuration file"
    )
    parser.add_argument(
        "-e", "--env", 
        help = "path to the .env file containing the API keys and endpoints",
        default = ".env"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.configPath)

    # Logger setup
    from utils import LogManager
    LogManager.initialize(conf["output"]["logfile"])
    logger = LogManager.get_logger("main")
    logger.info("Application started")

    # Load the benchmark data or benchmark questions
    from utils import load_benchmark_entries
    benchmark = load_benchmark_entries(conf)
    benchmark = benchmark

    # Load the .env file to access credentials for frontier models
    from dotenv import load_dotenv
    load_dotenv(args.env)

    # Check for batch job submission request in config
    batch_job_models = [
        "gemini-2.5-pro", "gemini-2.5-flash",
        "gemini-2.0-flash"
        ]
    
    if "batch_job" in conf:
        model_name = conf["model"]["name"]
        if model_name not in batch_job_models:
            logger.error(f"Requested model {model_name} does not support batch job execution")
            raise NotImplementedError(f"Requested model {model_name} does not support batch job execution")

        else:
            from llm_handlers import GoogleGenAIHandler
            llm_client = GoogleGenAIHandler(model_name)
    
    
    else:
        # Connect to the LLM client
        provider = conf["model"]["provider"]
        model_name = conf["model"]["name"]
        max_tokens = conf["model"]["max_tokens"] if "max_tokens" in conf["model"] else 20000

        try:
            adaptor_path = conf["model"]["adaptor"]
        except:
            adaptor_path = None

        if provider.lower() == "azure":
            from llm_handlers import AzureOpenAIHandler
            llm_client = AzureOpenAIHandler(model_name, token_limit=max_tokens)
        elif provider.lower() == "aws":
            from llm_handlers import AWSBedrockHandler
            llm_client = AWSBedrockHandler(model_name, token_limit=max_tokens)
        elif provider.lower() == "vertex":
            from llm_handlers import VertexAIHandler
            llm_client = VertexAIHandler(model_name, token_limit=max_tokens)
        elif provider.lower() == "google":
            from llm_handlers import GoogleGenAIHandler
            llm_client = GoogleGenAIHandler(model_name, token_limit=max_tokens)
        elif provider.lower() == "huggingface":
            from llm_handlers import HuggingFaceHandler
            llm_client = HuggingFaceHandler(model_path=model_name, adaptor_path = adaptor_path)
        else:
            raise NotImplementedError(f"Unsupported provider: {provider}")
        

    # Result files
    import os
    from pathlib import Path
    eval_config = conf["evaluation"]
    
    # update prompt file path for NEPABench
    if conf.get('nepabench_directory', None):
        eval_config["prompt_file"] = os.path.join(conf.get('nepabench_directory'), eval_config['prompt_file'])

    continue_from_previous = eval_config["continue_previous"] if 'continue_previous' in eval_config else True
    output_dir = conf["output"]["directory"]
    dir = Path(output_dir)
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)
        os.chmod(dir, 0o777)
    
    # Response evaluator module
    if conf["task"] == "question_answer":
        from evaluation import QAEvaluator
        eval_client = QAEvaluator(llm_client, eval_config["prompt_file"])
        context_type = eval_config["eval_kwargs"]["context_type"]
        if conf.get('nepabench_directory', None):
            eval_config['eval_kwargs']['nepabench_directory'] = conf.get('nepabench_directory')
        response_json_path = os.path.join(output_dir, f"responses_{context_type}.json")
        scores_json_path = os.path.join(output_dir, f"scores_{context_type}.json")
    elif conf["task"] == "information_extraction":
        from evaluation import InformationExtractor
        eval_client = InformationExtractor(llm_client, eval_config["prompt_file"])
        response_json_path = os.path.join(output_dir, f"responses.json")
        scores_json_path = os.path.join(output_dir, f"scores.json")
    elif conf["task"] == "tribe_extraction":
        from evaluation import TribeExtractor
        eval_client = TribeExtractor(llm_client, eval_config["prompt_file"])
        response_json_path = os.path.join(output_dir, f"responses.json")
        scores_json_path = os.path.join(output_dir, f"scores.json")
    elif conf["task"] == "structured_extraction":
        from evaluation import StructuredExtractor
        eval_client = StructuredExtractor(llm_client, eval_config["prompt_file"])
        response_json_path = os.path.join(output_dir, f"responses.json")
        scores_json_path = os.path.join(output_dir, f"scores.json")
    elif conf["task"] == "bin_assignment":
        from evaluation import CommentBinAssigner
        eval_client = CommentBinAssigner(llm_client, eval_config['prompt_file'])
        response_json_path = os.path.join(output_dir, f"responses.json")
        scores_json_path = os.path.join(output_dir, f"scores.json")
    elif conf["task"] == "comment_classification":
        from evaluation import CommentClassifier
        eval_client = CommentClassifier(llm_client, eval_config['prompt_file'])
        response_json_path = os.path.join(output_dir, f"responses.json")
        scores_json_path = os.path.join(output_dir, f"scores.json")
    elif conf["task"] == "bin_summarization":
        from evaluation import BinSummarizer
        eval_client = BinSummarizer(llm_client, eval_config['prompt_file'])
        response_json_path = os.path.join(output_dir, f"responses.json")
        scores_json_path = os.path.join(output_dir, f"scores.json")
    elif conf["task"] == "map_classification":
        from evaluation import MapClassifier
        eval_client = MapClassifier(llm_client, eval_config['prompt_file'])
        response_json_path = os.path.join(output_dir, f"responses.json")
        scores_json_path = os.path.join(output_dir, f"scores.json")
    elif conf["task"] == "comment_delineation":
        from evaluation import CommentDelineator
        eval_client = CommentDelineator(llm_client, eval_config['prompt_file'])
        response_json_path = os.path.join(output_dir, f"responses.json")
        scores_json_path = os.path.join(output_dir, f"scores.json")
    else:
        logger.error(f"Unknown task type {conf['task']} in configuration file")
        raise NotImplementedError(f"Unknown task type {conf['task']} in configuration file")
    
    # Check if batch job submission is requested
    if "batch_job" in conf:
        eval_client.execute_gemini_batch_job(
            benchmark, 
            output_path=response_json_path,
            continue_from_previous=continue_from_previous,
            **eval_config["eval_kwargs"],
            **conf["batch_job"]
            )
    
    else:
        eval_client.evaluate_batch(
            benchmark, 
            output_path=response_json_path,
            continue_from_previous = continue_from_previous, 
            **eval_config["eval_kwargs"]
            )
    
    
    # RAGAS metric evaluation module
    if conf['task'] in ['question_answer', 'bin_summarization']:
        from llm_handlers.azure_openai_handler import AzureChatOpenAIHandler
        from metrics import RAGAS_Evaluator

        eval_llm_client = AzureChatOpenAIHandler("gpt-4-preview")
        metrics = OmegaConf.to_container(conf["scoring"]["metrics"])
        batch_size = conf["scoring"]["batch_size"]
        embedding_model_name = conf["scoring"]["embed_model_name"]

        ragas_evaluator = RAGAS_Evaluator(
            llm_handler = eval_llm_client.llm, 
            embedding_model_name = embedding_model_name,
            )
        ragas_evaluator.evaluate_responses(
            response_json_path,
            scores_json_path,
            metrics=metrics,
            batch_size=batch_size, 
            continue_from_previous=continue_from_previous
            )
        
    elif conf['task'] == 'structured_extraction':
        from metrics import NestedStructureEvaluator
        nested_evaluator = NestedStructureEvaluator()
        nested_evaluator.batch_evaluate_nested_structures(
            response_path=response_json_path, 
            output_path=scores_json_path, 
            list_handling='merged'
            )
    
    else:
        metrics_config = OmegaConf.to_container(conf["scoring"])
        from metrics import MetricsEvaluator
        metrics_evaluator = MetricsEvaluator()

        metrics_evaluator.batch_evaluate(
            response_json_path, 
            config=metrics_config,
            output_path=scores_json_path,
            continue_from_previous=conf["continue_scoring_from_previous"] if 'continue_scoring_from_previous' in conf else True,
            nepabench_directory=conf.get('nepabench_directory', None)
        )


if __name__ == "__main__":
    main()