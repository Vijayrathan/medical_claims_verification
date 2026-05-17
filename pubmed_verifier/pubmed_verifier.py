import requests
import ast
import time
import os
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
# --- CONFIGURATION ---
# Get an API Key here: https://account.ncbi.nlm.nih.gov/
# If you have one, paste it below to run 3x faster.
API_KEY = os.getenv("PUBMED_API_KEY")

# Rate Limits: 
# No Key = 3 req/sec -> Max Workers ~3
# With Key = 10 req/sec -> Max Workers ~10
MAX_WORKERS = 10 if API_KEY else 3 

class PubMedMeshVerifier:
    def __init__(self, email, api_key=None):
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.session = requests.Session()

    def get_count(self, query):
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "email": self.email,
            "retmax": 0
        }
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            r = self.session.get(self.base_url, params=params, timeout=10)
            
            # Handle Rate Limiting (HTTP 429)
            if r.status_code == 429:
                time.sleep(2) # Backoff
                return self.get_count(query) # Retry
                
            r.raise_for_status()
            data = r.json()
            return int(data["esearchresult"]["count"])
        except Exception:
            return 0

    def verify_claim(self, subject, user_predicate, object_term):
        # Queries
        query_therapy = (
            f'("{subject}"[Title] AND "therapeutic use"[sh]) AND "{object_term}"[Title/Abstract]'
        )
        query_harm = (
            f'("{subject}"[Title] AND ("adverse effects"[sh] OR "toxicity"[sh] OR "etiology"[sh] OR "chemically induced"[sh])) '
            f'AND "{object_term}"[Title/Abstract]'
        )

        count_therapy = self.get_count(query_therapy)
        # Small sleep only if no API key to be extra safe inside the thread
        if not self.api_key: time.sleep(0.2) 
        count_harm = self.get_count(query_harm)
        
        total = count_therapy + count_harm

        if total == 0:
            return "UNVERIFIED"

        if count_harm == 0:
            ratio = 999.0
        else:
            ratio = count_therapy / count_harm

        is_positive_claim = user_predicate.lower() in ["treats", "cures", "helps", "heals", "prevents"]

        if is_positive_claim:
            if ratio >= 2.0: return "SUPPORTED"
            elif ratio <= 0.5: return "UNVERIFIED"
            else: return "AMBIGUOUS_SUPPORT"
        else:
            if ratio <= 0.5: return "SUPPORTED"
            elif ratio >= 2.0: return "UNVERIFIED"
            else: return "AMBIGUOUS_SUPPORT"

# --- WORKER FUNCTION ---
# This runs inside each thread
def process_single_row(row_data):
    # We re-instantiate or pass the verifier. 
    # Since requests.Session is thread-safe-ish, we can use a global or pass it.
    # Ideally, instantiate a lightweight verifier here or use a shared one.
    
    verifier = row_data['verifier']
    row = row_data['row']
    
    try:
        triples_list = ast.literal_eval(row["triples"])
    except:
        return "UNVERIFIED"
    
    if not isinstance(triples_list, list) or len(triples_list) == 0:
        return "UNVERIFIED"
    
    triple_verdicts = []
    for triple in triples_list:
        if len(triple) != 3: continue
        subject, predicate, object_term = triple
        
        if predicate not in ["treats", "cures", "helps", "heals", "causes", "worsens", "induces"]:
            continue

        verdict = verifier.verify_claim(subject, predicate, object_term)
        triple_verdicts.append(verdict)
    
    # Aggregation
    if not triple_verdicts:
        return "UNVERIFIED"
    elif "SUPPORTED" in triple_verdicts:
        return "SUPPORTED"
    elif "AMBIGUOUS_SUPPORT" in triple_verdicts:
        return "AMBIGUOUS_SUPPORT"
    else:
        return "UNVERIFIED"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    MY_EMAIL = "vijayrathank@gmail.com"

    
    # Initialize one verifier instance (Session handles connection pooling)
    main_verifier = PubMedMeshVerifier(MY_EMAIL, API_KEY)

    print("Loading dataset...")
    dataset = pd.read_parquet("../dataset/100k_dataset/triples_output_100k.parquet")
    dataset=Dataset.from_pandas(dataset)
    
    # Prepare data for threads
    # We wrap the row and the verifier object together
    tasks = [{'row': row, 'verifier': main_verifier} for row in dataset]
    
    results = []
    print(f"Starting processing with {MAX_WORKERS} workers...")
    
    # ThreadPoolExecutor is better for Network I/O than ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_row, task): i for i, task in enumerate(tasks)}
        
        # We use a list of size N to keep order correct
        results = [None] * len(tasks)
        
        # Process as they complete (with progress bar)
        for future in tqdm(as_completed(futures), total=len(tasks)):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = "UNVERIFIED" # Fallback on crash

    print("Saving...")
    dataset = dataset.add_column("unified_claim_status", results)
    dataset.to_parquet("triples_pubmed_unified_claims.parquet")
    
    df = pd.read_parquet("triples_pubmed_unified_claims.parquet")
    print(df['unified_claim_status'].value_counts())
