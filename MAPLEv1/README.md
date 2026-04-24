# Multi-context Assessment Pipeline for Language model Evaluation (MAPLE) v1.0

This project provides a framework for evaluating language models, particularly in the context of question answering and document retrieval tasks. It includes components for handling various LLM providers, data loading, response generation, and evaluation using RAGAS metrics. It also supports multiple context-types and multiple benchmark datasets.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/framework-RAGAS-orange)

## Citation

```bibtex
@inproceedings{meyur2025benchmarking,
    author={Rounak Meyur and Hung Phan and Koby Hayashi and Ian Stewart and Shivam Sharma and Sarthak Chaturvedi and Mike Parker and Dan Nally and Sadie Montgomery and Karl Pazdernik and Ali Jannesari and Mahantesh Halappanavar and Sai Munikoti and Sameera Horawalavithana and Anurag Acharya},
    title={Benchmarking LLMs for Environmental Review and Permitting},
    booktitle={Proceedings of the Workshop on Large Language Models for Scientific and Societal Advances (SciSoc LLM) at the 2025 ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year={2025},
    address={Toronto, Canada},
}
```

## Contents

1. [Project Structure](#project-structure)
2. [Subdirectories and Scripts](#subdirectories-and-scripts)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [Disclaimer](#disclaimer)
7. [License](#license)

## Project Structure

```
maple/
│
├── configs/                  # Configuration files for different evaluation runs
│   ├── eval_gpt4_config.yaml
│   ├── eval_mistral_config.yaml
│   ├── eval_gemini_config.yaml
│   ├── plot_config.yaml
│   ├── logging_config.ini
│   └── README.md
│
├── llm_handlers/                # LLM provider-specific implementations
│   ├── base_handler.py
│   ├── aws_bedrock_handler.py
│   ├── azure_openai_handler.py
│   ├── vertex_gemini_handler.py
│   ├── huggingface_handler.py
│   └── README.md
│
├── utils/                       # Utility functions for data handling and visualization
│   ├── dataloader.py
│   ├── logging_utils.py
│   ├── plot_utils.py
│   └── README.md
│
├── evaluation/                  # Core evaluation components
│   ├── evaluator.py
│   ├── metrics.py
│   └── README.md
│
├── main.py                      # Main entry point script
├── generate_plot.py             # Generate plot script
├── re_evaluate.py               # Script to re-evaluate RAGAS score
├── .env                         # Environment variables and API keys
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Subdirectories and Scripts

### The `configs` Directory

Contains configuration files for different evaluation scenarios:
- `eval_*_config.yaml`: Model-specific evaluation parameters
- `plot_config.yaml`: Visualization settings for results comparison
- `logging_config.ini`: Logging configuration settings

Detailed documentation: [configs/README.md](configs/README.md)

### The `llm_handlers` Directory

Provider-specific implementations for different LLM services:
- Base abstract handler defining the interface
- Concrete implementations for:
  - AWS Bedrock (Claude, Mistral, Llama3)
  - Azure OpenAI ([GPT-4](https://openai.com/index/gpt-4/), GPT-3.5)
  - Google Vertex AI (Gemini)
  - Local HuggingFace models

Detailed documentation: [llm_handlers/README.md](llm_handlers/README.md)

### The `utils` Directory

Essential utility functions:
- `dataloader.py`: CSV loading with context type detection
- `logging_utils.py`: Centralized logging configuration
- `plot_utils.py`: Visualization tools for results analysis

Detailed documentation: [utils/README.md](utils/README.md)

### The `evaluation` Directory

Core evaluation components:
1. `evaluator.py`: Response generation pipeline
   - Multiple context type support
   - Batched processing
   - Progress persistence
2. `metrics.py`: RAGAS metric evaluation
   - Customizable metrics selection
   - Batched processing
   - Progress persistence
   - Reevaluation for NaN values

Detailed documentation: [evaluation/README.md](evaluation/README.md)

### Main Script `main.py`

Entry point script that:
- Processes command-line arguments
- Loads configuration
- Initializes components
- Runs evaluation pipeline
- Handles error reporting

## Environment Setup

The `.env` file must contain credentials for the LLM services you plan to use:

1. **Azure OpenAI Setup**
```shell
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_API_VERSION="2023-12-01-preview"
AZURE_DEPLOYMENT_NAME="your-deployment"
```

2. **AWS Bedrock Setup**
```shell
# Run SSO configuration
aws configure sso

# Follow CLI prompts for:
# - SSO start URL
# - SSO Region
# - Role selection
```

3. **Vertex AI Setup**
```shell
VERTEXAI_CREDENTIALS_JSON_PATH="/path/to/credentials.json"
```

## Setup

1. Clone the repository:
```shell
git clone <repository-url>
cd maple
```

2. Create and activate virtual environment:
```shell
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```shell
pip install -r requirements.txt
```

4. Set up environment variables:
```shell
cp .env.example .env
# Edit .env with your credentials
```

## Usage

### Basic Evaluation

Run evaluation using a configuration file. For example, to run [GPT-4](https://openai.com/index/gpt-4/) evaluation:
```shell
python main.py -c configs/eval_gpt4_config.yaml
```

Configuration options:
```yaml
model:
  provider: azure  # azure|aws|vertex|huggingface
  name: gpt-4-turbo-preview
  max_tokens: 100000  # optional
  use_aws_role: false  # For AWS Bedrock

evaluation:
  context_type: gold  # none|rag|pdf|gold
  prompt_file: 'input/prompts/with_context.txt'
  max_attempts: 3
  RAGAS_batch_size: 20
  continue_previous: true # optional
  metrics:
    - answer_correctness

benchmark:
  questions_csv: "input/questions.csv"
  
output:
  directory: "results/model_gpt4"
  logfile: "logs/application_gpt4.log"
```

### Input File Format

The `questions_csv` file specified in the configuration must be a CSV file containing your benchmark questions and answers. The file supports the following columns:

#### Required Columns
- **`question`** - The question text to be answered by the LLM
- **`answer`** - The expected/ground truth answer

#### Optional Columns
- **`file_name`** - Name of the source document (required for RAG and PDF context types)
- **`page_number`** - Page number where the question/answer can be found
- **`question_type`** - Type of question (e.g., "closed", "open", "factual")
- **`proof`** - Supporting evidence or explanation for the answer
- **`context`** - Relevant context text (required for "gold" context type)

#### Example CSV Structure
```csv
file_name,page_number,question_type,question,answer,proof,context
document1.pdf,5,closed,Is this project subject to NEPA?,Yes,Section 102 applies,The National Environmental Policy Act requires...
document2.pdf,12,open,What are the main environmental concerns?,Air quality and noise,Table 3-1 shows impacts,Environmental analysis indicates...
```

#### Using Your Own Benchmark

To use your own benchmark data:

1. **Minimum Requirements**: Ensure your CSV file contains at least the `question` and `answer` columns
2. **Context Types**: 
   - For `context_type: none` - Only `question` and `answer` are needed
   - For `context_type: gold` - Include the `context` column with relevant text
   - For `context_type: rag` or `context_type: pdf` - Include the `file_name` column
3. **File Format**: Save as a standard CSV file with headers
4. **Text Encoding**: Use UTF-8 encoding for special characters


## Context Types

The pipeline supports different context types for evaluation:

- **`none`** - Direct question-answering without additional context
- **`gold`** - Uses human-provided context from the `context` column
- **`rag`** - Retrieval Augmented Generation using ChromaDB (requires `file_name`)
- **`pdf`** - Provides full PDF content as context (requires `file_name`)


### Plotting Results

Generate comparison plots:
```shell
python generate_plot.py -c configs/plot_config.yaml
```

Plot configuration:
```yaml
log_file: "logs/plot_generator.log"
metric: "answer_correctness"
path_pfx: "results/model"

single_context:
  context_type: "gold"
  outputs:
    barplot: "plots/barplot-compare"
    heatmap: "plots/heatmap-compare"
  plot_args:
    labelsize: 15
    colormap: "Blues"
  models:
    gpt4: "GPT-4"
    gemini: "Gemini-1.5Pro"
```

## Contributing

1. Fork the repository
2. Create your feature branch:
   ```shell
   git checkout -b feature/new-feature
   ```
3. Commit your changes:
   ```shell
   git commit -m 'Add new feature'
   ```
4. Push to the branch:
   ```shell
   git push origin feature/new-feature
   ```
5. Open a Pull Request

### Development Guidelines

- Add tests for new features
- Update documentation
- Follow existing code style
- Add logging for important operations

---

For detailed information about specific components, please refer to their respective README files in each directory.

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

## LICENSE
Copyright Battelle Memorial Institute 2025

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