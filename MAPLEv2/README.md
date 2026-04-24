# MAPLE — Multi-context Assessment Pipeline for Language model Evaluation

MAPLE is a framework for evaluating language models on NEPA (National Environmental Policy Act) document tasks. It supports multiple LLM providers, a range of extraction and QA benchmarks, and configurable scoring metrics.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-BSD--2--Clause-green)

This framework is designed to be used with the [NEPABench](https://huggingface.co/datasets/PNNL/NEPABench) dataset.

## Citation

If you use this code, please cite:

```bibtex
@misc{acharya2026nepabench,
    author={Anurag Acharya and Rounak Meyur and Sai Koneru and Kaustav Bhattacharjee and Bishal Lakha and Alexander C. Buchko and Reilly P. Raab and Hung Phan and Koby Hayashi and Dan Nally and Mike Parker and Sai Munikoti and Sameera Horawalavithana},
    title={NEPABench: A Benchmark Suite for Environmental Permitting},
    year={2026},
    institution={Pacific Northwest National Laboratory},
    note={Preprint},
}
```

## Contents

1. [Project Structure](#project-structure)
2. [Subdirectories and Scripts](#subdirectories-and-scripts)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Project Structure

```
evaluation/
│
├── configs/                  # YAML configuration files for each evaluation run
│   ├── cx_gemini.yaml
│   ├── ea_gemini.yaml
│   ├── eis_gemini.yaml
│   ├── tribe_gemini.yaml
│   ├── delineate_gemini.yaml
│   ├── assign_gemini.yaml
│   ├── summarize_gemini.yaml
│   ├── fedreg_gemini.yaml
│   ├── qa_gold_gemini.yaml
│   ├── qa_none_gemini.yaml
│   ├── qa_pdf_gemini.yaml
│   ├── qa_rag_gemini.yaml
│   ├── logging_config.ini
│   └── README.md
│
├── llm_handlers/             # LLM provider-specific handler implementations
│   ├── base_handler.py
│   ├── azure_openai_handler.py
│   ├── aws_bedrock_handler.py
│   ├── vertex_gemini_handler.py
│   ├── google_genai_handler.py
│   ├── huggingface_handler.py
│   └── README.md
│
├── evaluation/               # Task-specific evaluator classes
│   ├── base_evaluator.py
│   ├── qa_evaluator.py
│   ├── information_extractor.py
│   ├── structured_extractor.py
│   ├── tribe_extractor.py
│   ├── comment_delineator.py
│   ├── comment_bin_assigner.py
│   ├── comment_classifier.py
│   ├── bin_summarizer.py
│   ├── map_classifier.py
│   └── README.md
│
├── metrics/                  # Scoring and metric evaluation modules
│   ├── ragas_metrics.py
│   ├── metrics_evaluator.py
│   ├── nested_evaluator.py
│   ├── closed_metrics.py
│   ├── fuzzy_metrics.py
│   └── README.md
│
├── utils/                    # Shared utilities
│   ├── dataloader.py
│   ├── logging_utils.py
│   ├── prompt_utils.py
│   ├── rag_utils.py
│   ├── schema_utils.py
│   ├── parser_utils.py
│   ├── pdf_utils.py
│   ├── response_utils.py
│   ├── image_utils.py
│   └── README.md
│
├── main.py                   # Main entry point
├── Dockerfile
├── makefile
├── requirements.txt
└── README.md                 # This file
```

## Subdirectories and Scripts

### `configs/`

YAML configuration files, one per evaluation run. Each file specifies the model, task, benchmark dataset, evaluation parameters, scoring metrics, and output paths. All benchmark-relative paths are resolved against `nepabench_directory`.

See [configs/README.md](configs/README.md) for the full configuration reference.

### `llm_handlers/`

Provider-specific LLM handler implementations, all inheriting from `BaseLLMHandler`:

| Handler | Provider | Multimodal | Structured output |
|---|---|:---:|:---:|
| `AzureOpenAIHandler` | Azure OpenAI (GPT) | ✓ | ✓ |
| `AWSBedrockHandler` | AWS Bedrock (Claude, Mistral, Llama) | ✓ | ✓ |
| `VertexAIHandler` | Google Vertex AI (Gemini) | ✓ | ✓ |
| `GoogleGenAIHandler` | Google GenAI Client (Gemini) | ✓ | ✓ |
| `HuggingFaceHandler` | Local HuggingFace models | | |

See [llm_handlers/README.md](llm_handlers/README.md) for usage and environment variable requirements.

### `evaluation/`

Task-specific evaluator classes, all inheriting from `BaseEvaluator`. Each evaluator implements `prepare_prompt`, `evaluate_entry`, and `extract_response`.

| Evaluator | Task |
|---|---|
| `QAEvaluator` | `question_answer` |
| `InformationExtractor` | `information_extraction` |
| `StructuredExtractor` | `structured_extraction` |
| `TribeExtractor` | `tribe_extraction` |
| `CommentDelineator` | `comment_delineation` |
| `CommentBinAssigner` | `bin_assignment` |
| `CommentClassifier` | `comment_classification` |
| `BinSummarizer` | `bin_summarization` |
| `MapClassifier` | `map_classification` |

`BaseEvaluator` provides `evaluate_batch` (sequential with resume) and `execute_gemini_batch_job` (GCS-based batch for Gemini).

See [evaluation/README.md](evaluation/README.md) for full documentation.

### `metrics/`

Three scoring evaluators, selected automatically in `main.py` based on the task:

| Evaluator | Tasks | Metrics |
|---|---|---|
| `RAGAS_Evaluator` | `question_answer`, `bin_summarization` | `answer_correctness`, `answer_similarity`, `context_precision`, `context_recall`, `faithfulness` |
| `MetricsEvaluator` | Extraction and comment tasks | Deterministic, edit-distance, semantic, and soft-set metrics |
| `NestedStructureEvaluator` | `structured_extraction` | Auto-detected per-field metrics |

See [metrics/README.md](metrics/README.md) for configuration and metric reference.

### `utils/`

Shared utilities used across the framework:

| Module | Purpose |
|---|---|
| `dataloader.py` | Load and parse benchmark entries from JSON |
| `logging_utils.py` | Centralised `LogManager` |
| `prompt_utils.py` | `PromptManager` for template-based prompt formatting |
| `rag_utils.py` | ChromaDB setup and RAG chat client |
| `schema_utils.py` | Pydantic ↔ Vertex schema conversion, `HierarchicalPydanticGenerator` |
| `parser_utils.py` | Section extraction from NEPA documents |
| `pdf_utils.py` | PDF text extraction and chunk loading |
| `response_utils.py` | Response validation utilities |
| `image_utils.py` | Image loading and preprocessing |

See [utils/README.md](utils/README.md) for details.

### `main.py`

Single entry point for all evaluation runs. Reads the YAML config, instantiates the appropriate LLM handler and evaluator, runs response generation, then runs scoring.

```
main.py flow:
  1. Parse args → load config
  2. Initialize LogManager
  3. Load benchmark entries
  4. Load .env credentials
  5. Instantiate LLM handler (provider dispatch)
  6. Instantiate task evaluator (task dispatch)
  7. Run evaluate_batch or execute_gemini_batch_job → responses.json
  8. Run scoring evaluator → scores.json
```

---

## Setup

1. Clone the repository:
```shell
git clone --filter=blob:none --sparse --branch master https://github.com/pnnl/PermitAI.git
cd PermitAI
git sparse-checkout set MAPLEv2
cd MAPLEv2
```

2. Create and activate a virtual environment:
```shell
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:
```shell
pip install -r requirements.txt
```

4. Configure credentials (see [Environment Variables](#environment-variables)).

5. Download the NEPABench dataset from [PNNL/NEPABench](https://huggingface.co/datasets/PNNL/NEPABench) and set `nepabench_directory` in your config file to the downloaded dataset root. The example configs use `./nepabench`.

---

## Usage

### Running an evaluation

```shell
python main.py configs/qa_gold_gemini.yaml
```

Optionally specify a `.env` file containing model credentials:

```shell
python main.py configs/qa_gold_gemini.yaml --env .env
```

`main.py` uses the `task` field in the config to select the right evaluator and scorer automatically. Output files are written to `output.directory` (relative to the working directory):

- `responses.json` — raw LLM predictions
- `scores.json` — metric scores

For QA tasks the filenames include the context type: `responses_gold.json`, `scores_gold.json`.

### Example config (QA with gold context)

```yaml
model:
  provider: Vertex
  name: gemini-2.5-pro
  max_tokens: 100000

task: question_answer

nepabench_directory: /path/to/benchmark/folder

benchmark:
  input_file: 'nepaquad/benchmark.json'
  field_mapping:
    question: question
    answer: answer
    file_name: file_name
    chunks_json: json_file
    context: context
    question_type: question_type
  question_types:
    - closed
    - evaluation

evaluation:
  prompt_file: 'prompts/with_context.txt'
  continue_previous: true
  eval_kwargs:
    context_type: gold
    max_attempts: 10

scoring:
  batch_size: 4
  embed_model_name: 'all-MiniLM-L6-v2'
  metrics:
    - answer_correctness
    - answer_similarity

output:
  directory: 'results/nepabench_tests/qa_gold_gemini/'
  logfile: 'logs/nepabench_tests/qa_gold_gemini.log'
```

---

## Environment Variables

Create a `.env` file in the project root with credentials for the provider(s) you use:

**Vertex AI (current default)**
```shell
VERTEXAI_CREDENTIALS_JSON_PATH="/path/to/service-account.json"
VERTEXAI_LOCATION="us-central1"
```

**Azure OpenAI**
```shell
AZURE_OPENAI_ENDPOINT="https://your-azure-openai-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_API_VERSION="2023-12-01-preview"
AZURE_DEPLOYMENT_NAME="your-deployment"
```

The Azure endpoint value is a placeholder. Update `AZURE_OPENAI_ENDPOINT` to your own Azure OpenAI resource endpoint before running Azure-backed generation or RAGAS scoring.

**AWS Bedrock**
```shell
AWS_DEFAULT_REGION="us-west-2"
AWS_PROFILE="your-profile"        # or use AWS_BEDROCK_ROLE for IAM role
```

---

## Contributing

1. Fork the repository and create a feature branch:
   ```shell
   git checkout -b feature/my-feature
   ```
2. Implement changes, add logging, and update the relevant `README.md`.
3. Commit and open a Pull Request.

### Development guidelines

- Follow the existing `BaseEvaluator` / `BaseLLMHandler` patterns when adding new tasks or providers
- Add logging for all key operations via `LogManager`
- Update the relevant subdirectory README when adding or changing public interfaces

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

## License

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

---

For component-level details, see the README in each subdirectory.
