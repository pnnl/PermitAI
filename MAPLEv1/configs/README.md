# The `configs` directory

This directory contains configuration file required to run the evaluation pipeline using the `main.py` script. It also contains configuration files to generate plots which summarize the evaluation results.The configuration file is a `yaml` file consisting multiple fields.

## Evaluation `config.yaml` file

The `yaml` files to run the evaluation pipeline is structured as follows with the following fields.

 - `model`:
      - `provider`: name of the provider for the model
      - `name`: name of the LLM under test / path to the model
      - `max_tokens`: maximum number of tokens for the pdf context (optional)
      - `aws_role_use`: AWS role use (optional, only for AWS provider)
      - `aws_session_duration`: AWS session duration (optional, only for AWS provider)


 - `benchmark`:
      - `questions_csv`: path to the CSV file with benchmark questions
      - `question_types`: list of question types to select from the input questions CSV file (optional)
   
 - `rag_eval` (required, only for 'rag' context type):
      - `docID_csv`:  path to the CSV file with metadata mapping PDF documents to chroma collection ID
      - `path_chromadb`: path to the Chromadb database
      - `chroma_name`: name of the chroma collection

 - `pdf_eval` (required, only for 'pdf' context type):
      - `path_JSON_chunks`: path to the directory with JSON files containing chunks

 - `evaluation`:
      - `context`: type of context. (choose from: none, rag, pdf, gold)
      - `prompt_file`: path to text file with the prompt template with context and question
      - `max_attempts`: maximum number of attempts to try response generation
      - `continue_previous`: boolean flag to continue response generation from previous run (optional)
      - `metrics`: list of RAGAS metrics to compute (optional)
      - `RAGAS_batch_size`: batch size to compute RAGAS metric
      - `EMBED_model`: embedding model for chromadb client or RAGAS evaluator (optional, default is BAAI/bge-small-en-v1.5)

 - `output`:
      - `directory`: path to the directory with results
      - `logfile`: path to the log file


## Plot `config.yaml` file

The configuration file for generating plots consists of the following fields. Not all fields are necessary.

- `log_file`: log file path
- `metric`: name of the metric to plot
- `path_pfx`: path prefix to the CSV files
- `percentage`: `True`/`False` -- plot percentage values of not

- `multi_context`:
   - `context_type`: list of context types
   - `models`: dictionary of model names
   - `plot_args`: 
      - `width`: width of plot
      - `height`: height of plot
      - `xtickrotation`: rotation angle for x-ticks
      - `labelsize`: size of axis labels, titles
      - `ticklabelsize`: size of tick labels
      - `annotation_size`: size of annotation inside plot
      - `colormap`: colormap for heatmap

- `single_context`:
   - `context_type`: name of context types
   - `outputs`:
      - `barplot`: path to the output barplot
      - `heatmap`: path to the output heatmap
   - `models`: dictionary of model names
   - `plot_args`: 
      - `width`: width of plot
      - `height`: height of plot
      - `xtickrotation`: rotation angle for x-ticks
      - `labelsize`: size of axis labels, titles
      - `ticklabelsize`: size of tick labels
      - `annotation_size`: size of annotation inside plot
      - `colormap`: colormap for heatmap


## Logging configuration file

This directory contains the `logging_config.ini` file to configure the logging file.