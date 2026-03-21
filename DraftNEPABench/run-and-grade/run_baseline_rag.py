import argparse
import os
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, Iterable, List, Tuple, Optional
from openai import OpenAI, AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
import boto3
from botocore.exceptions import ClientError
import logging
import traceback
import numpy as np
import fitz  # PyMuPDF
import pickle
import numpy as np
import faiss
import tiktoken
from botocore.config import Config
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import vertexai
from dotenv import load_dotenv

# Load .env file
load_dotenv()

embedding_key = os.environ["OPEN_AI_EMBEDDING_KEY"]

endpoint = os.environ['AZURE_EMBEDDING_ENDPOINT']
model_name = "text-embedding-3-small"

embedding_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=embedding_key
)


# Configure the logger to write to 'logs.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs.log"),
        logging.StreamHandler()
    ]
)


# === Step 1: Extract Text from PDFs ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def read_pdfs(folder_path):
    texts, metadata = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            try:
                full_path = os.path.join(folder_path, filename)
                
                text = extract_text_from_pdf(full_path)
                texts.append(text)
                metadata.append({"filename": filename})
            except:
                print(f"Error reading {filename}")
    return texts, metadata

# === Step 2: Chunk Text ===
def chunk_text(text, max_tokens=500, overlap=50, model=model_name):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

# === Step 3: Generate Embeddings ===
def get_embedding(text):
    return embedding_client.embeddings.create(
        input=[text],
        model=model_name
    ).data[0].embedding


def get_embeddings_batched(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = embedding_client.embeddings.create(
            input=batch,
            model=model_name
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


# === Step 4: Build and Save Vector Store ===
def build_vector_store(folder_path, filename="references.pkl"):
    save_path = os.path.join(folder_path,filename)
    print(save_path)
    reference_path = os.path.join(folder_path,"References")
    if not os.path.exists(save_path):
        print(folder_path)
        texts, metadata = read_pdfs(reference_path)
        all_chunks, all_embeddings, all_meta = [], [], []

        for text, meta in zip(texts, metadata):
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                # all_embeddings.append(embedding)
                all_meta.append({**meta, "chunk_index": i})
            
        all_embeddings = get_embeddings_batched(all_chunks)


        # Save with FAISS
        if all_embeddings:
            dim = len(all_embeddings[0])
        else:
            dim = 1536	
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_embeddings).astype("float32"))

        with open(save_path, "wb") as f:
            pickle.dump({
                "index": index,
                "chunks": all_chunks,
                "metadata": all_meta
            }, f)
    else:
        print("Reference embeddings already exists")

# === Step 5: Load Vector Store and Query ===
def load_vector_store(embedding_path):
    vector_path = os.path.join(embedding_path, "references.pkl")
    with open(vector_path, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunks"], data["metadata"]

def retrieve_relevant_chunks(query, index, chunks,metadata, k=5):
    # print("Getting embedding for instructions and then relevant chunk")
    query_embedding = np.array(get_embedding(query)).astype("float32")
    D, I = index.search(np.array([query_embedding]), k)
    # return [chunks[i] for i in I[0]]

    return [{
            "chunk": chunks[i],
            "metadata": metadata[i],
        } for i in I[0]]



def write_to_markdown(content: str, filename):
    """
    Writes the given string content to a Markdown file.

    Parameters:
    - content (str): The text to write into the Markdown file.
    - filename (str): The name of the Markdown file (default is 'output.md').

    Returns:
    - None
    """
    with open(filename, "w", encoding="utf-8") as md_file:
        md_file.write(content)
    print(f"Markdown file '{filename}' created successfully.")


def read_markdown_file(filepath: str) -> str:
    """
    Reads a Markdown (.md) file and returns its content as a string.

    Args:
        filepath (str): Path to the Markdown file.

    Returns:
        str: Content of the file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at {filepath}"
    except Exception as e:
        return f"Error reading file: {e}"


def get_prompt(task_instruction,context):
   
    prompt = f"""
    You are a subject matter expert at drafting Environmental Impact Statement for National Environmental Policy Act(NEPA). 
    Using given task instruction and context from references, draft a section that meets all the success criteria. 

    Context from references: {context}

    Here is the task instruction: {task_instruction}

    While referencing provide:
        - Specific facts** with **(PDFName, p. X)** page References
        -Direct quotes** where relevant (short, judicious)
        Example:
        According to recent findings (LakeErieStatus.pdf, p. 66), ...
        “Quoted sentence.” (LakeErieStatus.pdf, p. 70)

        ![Nitrate trend (LakeErieStatus.pdf, p. 67)](images/LakeErieStatus_p67-000.png)
        *Figure 1. Nitrate concentration trend (LakeErieStatus.pdf, p. 67).*

    If context from the refrences is not provided use the urls provided in the reference section of task instruction.
    Check if the success criteria is met but do not include in the final draft. 
    Donot forget to add references. 

    Strictly return only the generated draft suitable for saving as markdown with all heading and sections.
    """
    return prompt

def get_all_reference_pdfs(task,task_root):
    reference_path = os.path.join(task_root,task,"workdir","References/*.pdf")
    all_pdfs_path = glob(reference_path)
    return all_pdfs_path

def extract_section_bullets(text, section_title="Detailed Instruction"):
    # Create a regex pattern to find the section and stop at the next section title or end of text
    pattern = rf'{re.escape(section_title)}\s*(.*?)(?=\n[A-Z][a-zA-Z ]+?:|\Z)'
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        return []

    section_text = match.group(1).strip()
    lines = section_text.splitlines()
    bullets = []

    for line in lines:
        line = line.strip()
        if line.startswith('-'):
            bullets.append(line[1:].strip())
        elif bullets and line:  # continuation of previous bullet
            bullets[-1] += ' ' + line

    return bullets

def get_task_instruction_prompt(task_instruction):

    return f"""
        You are a Subject Matter Expert (SME) in drafting Environmental Impact Statements (EIS). 

        You have been provided with a task instruction intended to guide the drafting of an EIS section. 
        However, the instruction is not currently well-suited for retrieval-augmented generation (RAG), which relies on precise and contextualized queries to retrieve relevant information from a vector database.

        Your objective is to rewrite or restructure the instruction to make it more effective for information retrieval. You may:
        - Rephrase the instruction for clarity and specificity.
        - Break it down into smaller, more focused sub-instructions if that improves retrieval accuracy.
        - Ensure the reformulated instruction is self-contained and contextually rich, even though you only have access to the original instruction.

        Here is the original task instruction:
        {task_instruction}

        Return only the improved instruction(s) separated by newline, optimized for retrieval. Do not include any additional commentary or explanation like Instruction set, RAG Instruction. 
        Only the update instruction.
        Limit maximum instruction to 20 instructions
        """

def extract_instructions(task_instruction,client, embedding_path, model_name):


    llm_instructions_path = os.path.join(embedding_path, f"{model_name}_llm_instruction.txt")

    if not os.path.exists(llm_instructions_path):
        prompt = get_task_instruction_prompt(task_instruction)

        if model_name == "gpt_5":
            response = client.responses.create(
                    model="gpt-5",  # replace with your model deployment name
                    input=prompt,
                    reasoning={"effort": "high"}
                )
            all_instructions = response.output_text.split("\n")
        elif model_name == "sonnet":
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"

            messages = [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]

            try:
                # Invoke the model with the request.
                response = client.converse(modelId=model_id, messages=messages)
            except (ClientError, Exception) as e:
                print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
                exit(1)
            all_instructions = response['output']['message']['content'][0]['text'].split("\n")
        else:
            response = client.generate_content(prompt)
            all_instructions = response.candidates[0].content.parts[0].text.split("\n")

        all_instructions = [instruction for instruction in all_instructions if len(instruction) > 1]

                
        with open(llm_instructions_path, 'w') as file:
            for instrct in all_instructions:
                file.write(f"{instrct}\n")
    else:
        with open(llm_instructions_path, 'r') as file:
            all_instructions = [line.strip() for line in file]

    return all_instructions

def get_all_instructions(task_prompt,client,embedding_path, model_name):
    instructions = extract_section_bullets(task_prompt)
    instruction_llm = extract_instructions(task_prompt,client,embedding_path,model_name)

    instructions.extend(instruction_llm)
    return instructions

def get_all_relavant_chunks(task_prompt,embedding_path,client, model_name):
    all_chunk = []

    relevant_chunks_name = f"relevant_chunk_{model_name}.pkl"
    relevant_chunk_path = os.path.join(embedding_path,relevant_chunks_name)
    if os.path.exists(relevant_chunk_path):
        
        with open(relevant_chunk_path, "rb") as f:
            all_chunk = pickle.load(f)

    else:
        instructions = get_all_instructions(task_prompt, client, embedding_path, model_name)


        build_vector_store(embedding_path)

        index, chunks, metadata = load_vector_store(embedding_path)

        for instruction in instructions:
            relevant_chunks = retrieve_relevant_chunks(instruction, index, chunks,metadata)
            all_chunk.extend(relevant_chunks)
        
        all_chunk = list(set(all_chunk))

                
        with open(relevant_chunk_path, "wb") as f:
            pickle.dump(all_chunk, f)

    return all_chunk
    

def get_draft_from_gpt(task,task_root,client):

    all_references_path = os.path.join(task_root, task, "workdir", "References", "*.pdf")
    pdf_files = glob(all_references_path)

    if pdf_files:
        embedding_path = os.path.join(task_root,task,"workdir")
        task_instruction_path = os.path.join(task_root, task, "workdir","task.md")
        task_instruction = read_markdown_file(task_instruction_path)
        # all_pdfs_path = get_all_reference_pdfs(task,task_root)
        all_relevant_chunks = get_all_relavant_chunks(task_instruction,embedding_path,client,"gpt_5")
    else:
        print("No reference pdfs found")
        all_relevant_chunks = []

    task_prompt = get_prompt(task,all_relevant_chunks)
    
    response = client.responses.create(
            model="gpt-5",  # replace with your model deployment name
            input=task_prompt,
            reasoning={"effort": "high"}
        )
    return response.output_text


def get_draft_from_gemini(task,task_root,client):
    all_references_path = os.path.join(task_root, task, "workdir", "References", "*.pdf")
    pdf_files = glob(all_references_path)

    if pdf_files:
        embedding_path = os.path.join(task_root,task,"workdir")
        task_instruction_path = os.path.join(task_root, task, "workdir","task.md")
        task_instruction = read_markdown_file(task_instruction_path)
        # all_pdfs_path = get_all_reference_pdfs(task,task_root)
        all_relevant_chunks = get_all_relavant_chunks(task_instruction,embedding_path,client,"gemini-2.5-pro")
    else:
        print("No reference pdfs found")
        all_relevant_chunks = []

    task_prompt = get_prompt(task,all_relevant_chunks)
    
    response = client.generate_content(task_prompt)
    return response.candidates[0].content.parts[0].text


def get_draft_from_sonnet(task,task_root,client):
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    all_references_path = os.path.join(task_root,task,"workdir","References")

    if os.listdir(all_references_path):

        embedding_path = os.path.join(task_root,task,"workdir")
        task_instruction_path = os.path.join(task_root, task, "workdir","task.md")
        task_instruction = read_markdown_file(task_instruction_path)
        # all_pdfs_path = get_all_reference_pdfs(task,task_root)
        all_relevant_chunks = get_all_relavant_chunks(task_instruction,embedding_path,client,model_name="sonnet")
    else:
        print("No reference pdfs found")
        all_relevant_chunks = []

    task_prompt = get_prompt(task,all_relevant_chunks)
    
     # Define the prompt for the model.

    messages = [
            {
                "role": "user",
                "content": [{"text": task_prompt}],
            }
        ]

    try:
        # Invoke the model with the request.
        response = client.converse(modelId=model_id, messages=messages)
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)
        raise
    return response['output']['message']['content'][0]['text']


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate EIS section draft using different LLMs")
    p.add_argument("--task", required=True, help="'all' or a number like '1' or '1,2'")
    p.add_argument("--model", default="gpt-5", help="Model name (default: gpt-5)")
    p.add_argument(
        "--results-root",
        default=None,
        help="Base results directory containing agent folders (auto-detects run-and-grade/results, run_and_grade/results, or results/)",
    )
    p.add_argument("--trials", type=int, default=1, help="Number of grading evaluations to run in parallel per execution trial")
    p.add_argument("--workers", type=int, default=None, help="Max parallel workers (default: trials)")
    return p.parse_args(list(argv))

def create_result_folder(root,model_name, task, trial):
    model_name_map = {"gpt":"gpt-5", "gemini":"gemini-2.5-pro","sonnet":"sonnet-4.5"}
    result_dir = os.path.join(root,"run-and-grade","results",model_name_map[model_name],task,f"trial-{trial}","out")
    os.makedirs(result_dir, exist_ok=True)

    return result_dir


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    base = Path.cwd()

    task_folder = os.path.join(base,"run-and-grade","tasks-baseline")

    print(task_folder)

    arg_task = args.task
    model = args.model 
    trials = args.trials

    arg_task_lc = arg_task.strip().lower()
    if arg_task_lc == "all":
        individual_tasks = [task for task in os.listdir(task_folder) if os.path.isdir(os.path.join(task_folder, task))]

    else:
        # Support single number or comma-separated list for robustness
        individual_tasks = [f"task-{p.strip()}" for p in arg_task_lc.split(",") if p.strip()]
    

    if model == "gpt":
        subscription_key = os.environ["OPEN_AI_KEY"]
        client = OpenAI(  
        base_url = os.environ["AZURE_ENDPOINT"],
        api_key=subscription_key 
        )
    elif model == "sonnet":
        session_kwargs = {
        "region_name": os.environ.get("AWS_REGION"),
        "profile_name": os.environ.get("AWS_PROFILE") 
        }
        session_kwargs['aws_access_key_id'] = os.environ.get("AWS_ACCESS_KEY_ID", None)
        session_kwargs['aws_secret_access_key'] = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        session_kwargs['aws_session_token'] = os.environ.get("AWS_SESSION_TOKEN", None)

        session = boto3.Session(**session_kwargs)

        
        timeout_config = Config(
            connect_timeout=30,  # seconds to wait for connection
            read_timeout=180,    # seconds to wait for response
            retries={
                'max_attempts': 2,
                'mode': 'standard'
            }
        )


        # Initialize the Bedrock Runtime client
        client = session.client(service_name='bedrock-runtime', region_name=session_kwargs['region_name'], config=timeout_config)
    elif model == "gemini":
        credentials_path = os.environ.get(
            "VERTEXAI_CREDENTIALS_JSON_PATH", 
            "vertexai_cred.json"
        )
        credentials = service_account.Credentials.from_service_account_file(credentials_path)

        # Initialize Vertex AI with project from credentials
        vertexai.init(project=credentials.project_id, credentials=credentials)

        # Return the initialized model
        client = GenerativeModel('gemini-2.5-pro')  # gemini-2.5-flash  

        
    for task in individual_tasks:
        
        for trial in range(1,trials+1):
            print(trial)
            try:
                result_dir = create_result_folder(base,model,task,trial)
                report_path = os.path.join(result_dir,"report.md")
                if not os.path.exists(report_path):
                    print(f"Generating draft for {task} and trial-{trial}")
                    if model == "gpt":
                        draft = get_draft_from_gpt(task=task,task_root=task_folder,client=client)
                    elif model == "sonnet":
                        draft = get_draft_from_sonnet(task=task,task_root=task_folder,client=client)
                    elif model == "gemini":
                        draft = get_draft_from_gemini(task=task,task_root=task_folder,client=client)


                    write_to_markdown(draft,report_path)
                else:
                    print(f"Skiping {task}, trial-{trial} as it already exists")
                    
            except Exception as E:

                logging.error(f"Failed to generate draft for {model} and {task}': {E}")
                logging.debug("Exception details:\n" + traceback.format_exc())
          

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


