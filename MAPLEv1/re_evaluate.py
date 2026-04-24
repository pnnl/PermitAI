def main():

    # Argument parser
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
    parser.add_argument(
        "-b", "--batch_size", 
        help = "number of entries to evaluate in a batch",
        default = 5
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.configPath)

    # Logger setup
    from utils.logging_utils import LogManager
    LogManager.initialize(conf['output']["logfile"])
    logger = LogManager.get_logger("re_eval")
    logger.info("Re-eval application started")

    # Load the .env file to access credentials for frontier models
    from dotenv import load_dotenv
    load_dotenv(args.env)
    
    
    # RAGAS metric evaluation module
    from llm_handlers.azure_openai_handler import AzureChatOpenAIHandler
    from evaluation import RAGAS_Evaluator

    out_dir = conf['output']['directory']
    context = conf['evaluation']['context']
    response_csv_path = f"{out_dir}/responses_{context}.csv"
    scores_csv_path = f"{out_dir}/scores_{context}.csv"

    eval_llm_client = AzureChatOpenAIHandler("gpt-4-preview")
    embedding_model_name = conf["evaluation"]["EMBED_model"]

    evaluator = RAGAS_Evaluator(
        response_csv_path, 
        llm_handler = eval_llm_client.llm, 
        embedding_model_name = embedding_model_name
        )
    evaluator.reevaluate_results(scores_csv_path, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
    