# DraftNEPABench: A Benchmark for Drafting NEPA Document Sections with Coding Agents
DraftNEPABench is a novel benchmark designed to evaluate the capabilities of Large Language Models (LLMs) and LLM-based agents in drafting sections of Environmental Impact Statements (EIS). This benchmark is curated by subject matter experts (SMEs) to ensure that the tasks reflect realistic, domain-relevant drafting challenges encountered in environmental planning and compliance processes.

This repo provides scripts to generate EIS section draft from the benchmark using coding agents and vanilla RAG. It also includes a script to grade the generated drafts based on the rubric provided in the benchmark using LLM-judges. 

The dataset necessary to run this benchmark lives on [HuggingFace](https://huggingface.co/datasets/PNNL/DraftNEPABench). You would need to download the dataset and put it in the appropriate folder as explained below in order to run this code.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-BSD-green)

![](https://img.shields.io/badge/ChatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![](https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=claude&logoColor=white)
![](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{acharya_draftnepabench,
  title        = {DraftNEPABench: A Benchmark for Drafting NEPA Document Sections with Coding Agents},
  author       = {Acharya, Anurag and Lakha, Bishal and Meyur, Rounak and Nuttall, Rohan and Chaturvedi, Sarthak and Halappanavar, Anika and Hare, Leah and Zeng, Lin and Parker, Mike and Munikoti, Sai and Horawalavithana, Sameera},
  note         = {Manuscript},
  year         = {2026}
```


## Setup
### Environmental Setup
The .env file must contain credentials for the LLM services you plan to use:

1. Azure OpenAI
```bash
CODEX_AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_ENDPOINT= 
AZURE_EMBEDDING_ENDPOINT=
AZURE_OPENAI_API_VERSION=
```
2. AWS Bedrock
```bash
# Run SSO configuration
aws configure sso

# Follow CLI prompts for:
# - SSO start URL
# - SSO Region
# - Role selection
```
3. Vertex AI 
```bash
VERTEXAI_CREDENTIALS_JSON_PATH="/path/to/credentials.json"
```

4. Gemini 
```bash
GOOGLE_GENAI_USE_VERTEXAI=
GEMINI_API_KEY=
```

5. Claude Code
```bash
bash claude_env.sh
```

### Project Setup
1. Clone the repository
```bash
git clone <repository-url>
cd DraftNEPABench-public
```
2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Set up environmental variables
```
cp .env.example .env
# Edit .env with your credentials
```

## Project Structure

Download the DraftNEPABench dataset from HuggingFace and keep the Tasks  under `run-and-grade/tasks/task-<N>`.

**HuggingFace Dataset Link:** [https://huggingface.co/datasets/PNNL/DraftNEPABench](https://huggingface.co/datasets/PNNL/DraftNEPABench)


### Docx Case Documents
Each task should include case documents in `.docx` format inside a dedicated folder within `workdir/`:

```
run-and-grade/
  tasks/
    agent_instructions.md
    wrapup.md   # optional
    task-<N>/
      References/
        reference.pdf
      task.docx
      ground_truth.docx
      rubric.docx
```

- `task.docx`: Task description and instructions.
- `ground_truth.docx`: The authoritative expected answer/content.
- `rubric.docx`: Grading criteria for evaluation.

Notes
- All three `.docx` files are required when using the docx-based workflow.

### Preprocess
```bash
bash preprocess.sh
```
This will generate `task.md` from the `task.docx` and `rubric.md` from `rubric.docx` and `ground_truth.docx`. Once they are generated, a `workdir` folder is created and Reference and `task.md` is moved to the `workdir`. All other documents are moved to evaluation-documents folder.


```
run-and-grade/
  tasks/
    agent_instructions.md
    wrapup.md   # optional
    task-<N>/
      workdir
        References/
          reference.pdf
        task.md
  evaluation-documents/
    task-<N>/
      task_documents/
        ground_truth.docx
        rubric.docx
        task.docx
      rubric.md
```



## Generate drafts from Coding agents
Run tasks to generate reports: `bash run-and-grade/run_tasks.sh --task task-id --agent agentName --trials N`

- task-id -> all|1|1,2
- agentName -> codex or claude or gemini 
- trials -> 1 to M

Each run will use `agent_instructions.md`, which is shared across all task, and go the the task-id directory and follow the task specific prompt `task.md` . This will generate final draft and intermediate files which is moved to results directory. 

```
run-and-grade/
  results/
    agentName/
      task-<N>/
        trials-<M>/
          out/
            images/
              image.png
            report.md
            report.html
          text/
            reference.txt
          PLAN.md
          scratchpad.md
```


## Generate drafts from RAG Baseline
Grade outputs (multi‑trial): `python run-and-grade/run_baseline_rag.py --task task-id --model modelName --trials N`

- task-id -> all|1|1,2
- modelName -> gpt or sonnet or gemini 
- trials -> 1 to M


## Grade Drafts
2. Grade outputs (multi‑trial): `python run-and-grade/grade_tasks.py --task task-id --model --modelName --agent agentName --trials 5`

- task-id -> all|1|1,2
- modelName -> gpt or sonnet or gemini 
- agentName -> codex or gemini or claude
- trials -> 1 to M

`model` is a LLM-judge

### Grading Output
Produce Per task per trial JSON with the grades and justification

```run-and-grade/
  results/
    agentName/
      task-<N>/
        grades/
          modelName/
            trial-M-grade.json
```
### Aggregate and analyse results
evlauation/ folder contains three notebooks:
1. classify_criteria.ipynb -> classifies task specific criteria to 4 broad categories
2. aggregate_results.ipynb -> generates result.csv by aggregating information from all grades and metadata of DraftNEPABench.
3. result_analysis_and_visualization.ipynb -> contain code to read result.csv and carry out further analysis of the result



## DISCLAIMER

This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

```
             PACIFIC NORTHWEST NATIONAL LABORATORY
                          operated by
                            BATTELLE
                            for the
               UNITED STATES DEPARTMENT OF ENERGY
                under Contract DE-AC05-76RL01830
```

## LICENSE
Copyright Battelle Memorial Institute 2026

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.