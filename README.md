# RPR-RAG: Reasoning Path Retrieval for Knowledge-Graph-Augmented Question Answering

This repository implements a **Reasoning Path Retrieval based RAG (RPR-RAG)** framework for knowledge-graph question answering.  
The system supports **pure LLM-based QA** and **RAG-enhanced QA** using a locally deployed Neo4j knowledge graph.

---

## 1. Dataset and Model Preparation

Please download the dataset from the following Google Drive link:

https://drive.google.com/drive/folders/1XnS5lVDxmvwmzcg7TOkUKAvIwHw-WvQK?usp=drive_link

After downloading, **extract all files into the `data/` directory**:

```
project_root/
├── data/
│   ├── metaqa-1hop_test_dataset.jsonl
│   ├── ...
```

Please download the embedding model from the following Google Drive link:

https://drive.google.com/drive/folders/1QxYdLlqWucrtpM7jYX4YS7zhASlO1VYk?hl=ko

After downloading, **extract all files into the `embedding_model/` directory**:

```
project_root/
├── embedding_models/
│   ├── webqsp-embedding-model/
│   ├── ...
```

---

## 2. Runtime Environment

### 2.1 Neo4j (Local Deployment)

- Deployment mode: **Local Docker deployment**
- Experimental version:
```
neo4j:2025.01.0
```

Make sure Neo4j is running before executing any retrieval-related scripts.

---

### 2.2 Python Dependencies

```
python==3.10.16
numpy==1.26.4
scikit-learn==1.6.1
networkx==3.1
tqdm==4.67.1
torch==2.6.0+cu126
transformers==4.48.3
datasets==3.5.0
huggingface-hub==0.28.1
spacy==3.8.5
langchain==0.3.19
langchain-core==0.3.40
langchain-community==0.3.18
langchain-huggingface==0.1.2
langchain-neo4j==0.3.0
langchain-openai==0.3.7
openai==1.65.4
neo4j==5.28.1
pydantic==2.10.6
dashscope==1.23.1
en-core-web-sm==3.8.0
```

---

## 3. Create Python Environment

```bash
conda create -n env_rpr python=3.10.16
conda activate env_rpr
python -m pip install -r requirements.txt
```

> **Note**  
> The torch version in requirements.txt may be different for your device.  
> Please modify it accordingly.

Install the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

## 4. Configure Large Language Model API

Example for GPT-4 (in gpt4.py):

```python
self.llm = BaseChatOpenAI(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="YOUR_API_KEY"
)
```

---

## 5. Configure Neo4j Connection

In 'retriever.py', configure your local Neo4j credentials:

```python
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
```

---

## 6. Retrieval Parameters

In the 'do_retrieve' function of 'retriever.py', configure retrieval parameters:

```python
K_base = [2, 2, 2, 2]
K_new = [2, 2, 2, 2]
K_base_r = [2, 2, 2, 2]
K_new_r = [2, 2, 2, 2]
bidirectional = True
```

| Parameter       | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `K_base`        | Top-K value for **base (forward) retrieval**                      |
| `K_new`         | Top-K value for **supplementary (forward) retrieval**             |
| `K_base_r`      | Top-K value for **base reverse retrieval**                        |
| `K_new_r`       | Top-K value for **supplementary reverse retrieval**               |
| `bidirectional` | Whether to enable **bidirectional retrieval** (`True` or `False`) |

---

## 7. Knowledge Base Caching

Each question-specific Knowledge Base JSON file occupies approximately **500 MB**.  
Use `export_graph_to_json` to export the Knowledge Base.
Use `import_json_to_graph` to reload it and avoid repeated construction.

---

## 8. Running the Code

### 8.1 Pure LLM QA

```bash
python gen_answer.py   --model_name gpt-5-mini   --d metaqa-1hop   --data_path data/metaqa-1hop_test_dataset.jsonl
```
Output:
```bash
results/answer/metaqa-1hop/gpt-5-mini/test/answer.jsonl
```

### 8.2 RAG-based QA

#### Step 1: Retrieval
```bash
python gen_retrieved_knowledge.py   --model_name gpt-5-mini   --d metaqa-1hop   --data_path data/metaqa-1hop_test_dataset.jsonl   --embedding_model_name bge-small-en   --with_question_extracted
```
Output:
```bash
results/retrieved_knowledge/metaqa-1hop/gpt-5-mini/test/retrieved_knowledge.jsonl
```

#### Step 2: Answer Generation
```bash
python gen_answer.py   --model_name gpt-5-mini   --d metaqa-1hop   --data_path results/retrieved_knowledge/metaqa-1hop/gpt-5-mini/test/retrieved_knowledge.jsonl   --with_rag
```
Output:
```bash
results/answer/metaqa-1hop/gpt-5-mini+rag/test/answer.jsonl
```

#### Step 3: Evaluation
```bash
python evaluate_answer.py   --path results/answer/metaqa-1hop/gpt-5-mini+rag/test/answer.jsonl
```
Output:
```bash
results/answer/metaqa-1hop/gpt-5-mini+rag/test/detailed_eval_result.jsonl
results/answer/metaqa-1hop/gpt-5-mini+rag/test/eval_result.txtl
```

---

## 9. Directory

All logs are stored in the `log/` directory.

All result files are stored in the `results/` directory.

All large language models are stored in the `llm_models/` directory.

All embedding models are stored in the `embedding_models/` directory. 

Fine-tuned models `webqsp-embedding-model` and `cwq-embedding-model` can be loaded.


---

## 10. Command-Line Arguments

| Argument                    | Description               |
| --------------------------- | ------------------------- |
| `--model_name`              | Large language model name |
| `--d`                       | Dataset name              |
| `--data_path`               | Dataset path              |
| `--embedding_model_name`    | Embedding model           |
| `--rag`                     | Enable RAG                |
| `--with_question_extracted` | Extract question keywords |

---

## 11. License

This project is intended for **research purposes only**.
