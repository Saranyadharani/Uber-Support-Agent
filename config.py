import os
from dotenv import load_dotenv
load_dotenv()

BRAND_HANDLE = "Uber_Support"

RAW_CSV = "data/raw/twcs.csv"                     
THREADS_JSONL = f"data/processed/{BRAND_HANDLE}_threads.jsonl"
RESOLVED_THREADS_JSONL = f"data/processed/{BRAND_HANDLE}_resolved_threads.jsonl"  
INDEX_DIR = f"data/processed/{BRAND_HANDLE}_index"

GOLDEN_LABELING_BATCH = "labeling/labeling_batch.json"   
GOLDEN_LABELING_OUTPUT = "labeling/labeling_output.json" 
GOLDEN_SET_CSV = "eval/golden_set.csv"                   

# Sampling
MAX_THREADS_TO_LOAD = 20000     
GOLDEN_SET_SIZE = 200
PER_INTENT_TARGET = 20          
EDGE_CASE_TARGET = 40           


GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # local sentence-transformers model, no API cost

RANDOM_SEED = 42

def groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY in your environment or .env file "
                            "(get one free at https://console.groq.com/keys)")
    return key
