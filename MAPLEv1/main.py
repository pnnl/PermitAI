def main():

    # Argument parser
    import sys
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--configPath",
        help = "path to the configuration file", 
        default = "configs/v1-eis-mistral.yaml"
    )
    parser.add_argument(
        "-e", "--env", 
        help = "path to the .env file containing the API keys and endpoints",
        default = ".env"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.configPath)

    # Logger setup
    from utils.logging_utils import LogManager
    LogManager.initialize(conf["output"]["logfile"])
    logger = LogManager.get_logger("main")
    logger.info("Application started")

    # Load the benchmark data or benchmark questions
    from utils import process_benchmark_csv
    questions_csv = conf["benchmark"]["questions_csv"]
    question_types = conf["benchmark"]["question_types"] if "question_types" in conf["benchmark"] else None
    benchmark, supported_context_types = process_benchmark_csv(questions_csv, question_type=question_types)

    context = conf["evaluation"]["context"]
    if not context in supported_context_types:
        err_message = f"Context type {conf['context']} is not supported for the input dataset of questions & answers"
        logger.error(err_message)
        sys.exit(1)
    

    # Load the .env file to access credentials for frontier models
    from dotenv import load_dotenv
    load_dotenv(args.env)

    # Connect to the LLM client
    provider = conf["model"]["provider"]
    model_name = conf["model"]["name"]
    max_tokens = conf["model"]["max_tokens"] if "max_tokens" in conf["model"] else 20000

    if provider.lower() == "azure":
        from llm_handlers.azure_openai_handler import AzureOpenAIHandler
        llm_client = AzureOpenAIHandler(model_name)
    elif provider.lower() == "aws":
        aws_role_use = conf["model"]["aws_role_use"] if "aws_role_use" in conf["model"] else False
        DurationSeconds = conf["model"]["aws_session_duration"] if "aws_session_duration" in conf["model"] else 900
        from llm_handlers.aws_bedrock_handler import AWSBedrockHandler
        llm_client = AWSBedrockHandler(model_name, aws_role_use=aws_role_use, DurationSeconds=DurationSeconds)
    elif provider.lower() == "vertex":
        from llm_handlers.vertex_gemini_handler import VertexGeminiHandler
        llm_client = VertexGeminiHandler(model_name)
    elif provider.lower() == "huggingface":
        from llm_handlers.huggingface_handler import HuggingFaceHandler
        llm_client = HuggingFaceHandler(model_path=model_name)
    else:
        raise NotImplementedError(f"Unsupported provider: {provider}")
    

    # Result files
    import os
    from pathlib import Path
    output_dir = conf["output"]["directory"]
    dir = Path(output_dir)
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)
        os.chmod(dir, 0o777)
    response_csv_path = os.path.join(output_dir, f"responses_{context}.csv")
    scores_csv_path = os.path.join(output_dir, f"scores_{context}.csv")
    
    # response evaluator module
    logger.debug(f"Evaluating RAG benchmark with {provider}--{model_name}")
    from evaluation.evaluator import Evaluator
    eval_agent = Evaluator(
        llm_handler = llm_client,
        prompt_file = conf["evaluation"]["prompt_file"]
    )
    max_attempts = conf["evaluation"]["max_attempts"]
    embedding_model_name = conf["evaluation"]["EMBED_model"]
    continue_from_previous = conf["evaluation"]["continue_previous"] if 'continue_previous' in conf["evaluation"] else True

    path_json_chunks = None
    if context == 'pdf':
        if 'pdf_eval' not in conf:
            logger.error("YAML configuration file does not contain relevant fields for loading JSON chunks")
            raise FileNotFoundError("Missing directory with JSON chunks")
        else:
            path_json_chunks = conf['pdf_eval']['path_JSON_chunks']
    
    path_chromadb = None
    chroma_collection_name = None
    docID_csvpath = None
    if context == 'rag':
        if 'rag_eval' not in conf:
            logger.error("YAML configuration file does not contain relevant fields for loading Chromadb")
            raise FileNotFoundError("Missing directory with ChromaDB paths")
        else:
            path_chromadb = conf['rag_eval']['path_chromadb']
            chroma_collection_name = conf['rag_eval']['chroma_name']
            docID_csvpath = conf['rag_eval']['docID_csv']

    eval_agent.evaluate_benchmark(
        benchmark, 
        output_path=response_csv_path,
        context_type=context,
        chromadb_path = path_chromadb,
        collection_name = chroma_collection_name,
        json_directory = path_json_chunks,
        docID_csvpath=docID_csvpath,
        max_attempts = max_attempts,
        continue_from_previous = continue_from_previous, 
        max_tokens = max_tokens, 
        embed_model=embedding_model_name
        )
    
    # RAGAS metric evaluation module
    from llm_handlers.azure_openai_handler import AzureChatOpenAIHandler
    from evaluation import RAGAS_Evaluator

    eval_llm_client = AzureChatOpenAIHandler("gpt-4-preview")
    metrics = conf["evaluation"]["metrics"]
    batch_size = conf["evaluation"]["RAGAS_batch_size"]

    evaluator = RAGAS_Evaluator(
        response_csv_path, 
        llm_handler = eval_llm_client.llm, 
        embedding_model_name = embedding_model_name,
        metrics=metrics
        )
    evaluator.evaluate_results(scores_csv_path, batch_size=batch_size, continue_from_previous=continue_from_previous)

if __name__ == "__main__":
    main()