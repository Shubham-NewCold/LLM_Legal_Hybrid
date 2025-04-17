# routes.py
import time
from flask import Blueprint, render_template, request, jsonify
import langchain_utils.qa_chain as qa_module
from langchain_utils.qa_chain import get_detected_customer_names
import markdown
from langchain_core.callbacks.manager import CallbackManager
try:
    from email_tracer import EmailLangChainTracer
except ImportError:
    print("WARN: email_tracer.py not found. Using basic CallbackManager.")
    class EmailLangChainTracer:
        def __init__(self, project_name): pass # Dummy init
import re
import sys
import traceback
import itertools
import torch
from langchain.chains.mapreduce import MapReduceDocumentsChain
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Any, Set, Optional

# --- Updated Config Imports ---
# Import necessary config values including the new RRF threshold
from config import (
    RRF_K, PROJECT_NAME,
    RERANKER_ENABLED, RERANK_CANDIDATE_POOL_SIZE,
    PASS_2_SCORE_THRESHOLD,
    SPLADE_TOP_N, TOP_K as FAISS_TOP_K,
    DYNAMIC_K_ENABLED,
    DYNAMIC_K_COMBINED_SCORE_THRESHOLD, # Renamed threshold
    DYNAMIC_K_RRF_SCORE_THRESHOLD,      # New threshold for RRF
    DYNAMIC_K_MIN_CHUNKS, DYNAMIC_K_MAX_CHUNKS
)
print("--- Imported settings from config ---")


main_blueprint = Blueprint("main", __name__)

# --- Helpers (generate_keyword, get_customer_filter_keyword) (Unchanged) ---
# [ ... Keep the generate_keyword and get_customer_filter_keyword functions exactly as they were ... ]
def generate_keyword(customer_name):
    if not customer_name: return None
    name_cleaned = customer_name
    suffixes = [' Pty Ltd', ' Pty Limited', ' Ltd', ' Limited', ' Inc']
    for suffix in suffixes:
        # Use lower() for case-insensitive comparison
        if name_cleaned.lower().endswith(suffix.lower()):
            name_cleaned = name_cleaned[:-len(suffix)].strip()
            break
    parts = name_cleaned.split()
    return parts[0].lower() if parts else None

def get_customer_filter_keyword(query) -> Tuple[Optional[str], List[str]]:
    """
    Identifies customer names in the query.
    Returns:
        - A single customer name if exactly one is found (for filtering).
        - None if zero or multiple are found.
        - The list of all unique customer names found in the query.
    """
    detected_names = get_detected_customer_names()
    found_original_names: List[str] = [] # Keep track of names found in query

    if not detected_names:
        print("DEBUG [Filter]: No detected customer names loaded. Cannot filter.")
        return None, found_original_names # Return None and empty list

    # --- Build keyword map (same as before) ---
    customer_keywords_map = {}
    # Use normalized names from metadata for the map keys
    for name in detected_names: # detected_names should contain normalized names now
        keyword = generate_keyword(name) # Generate keyword from normalized name
        if keyword:
            # Map keyword and full normalized name (lower, no spaces) to the normalized name
            customer_keywords_map[keyword] = name
            full_name_keyword = name.lower().replace(" ", "")
            if full_name_keyword != keyword:
                 customer_keywords_map[full_name_keyword] = name
            # No need to strip suffixes again if detected_names are already normalized
    # --- End keyword map build ---

    query_lower = query.lower()
    # Sort keywords by length descending to match longer names first
    sorted_keywords = sorted(customer_keywords_map.keys(), key=len, reverse=True)
    temp_query = query_lower # Use a temporary query string for replacement

    # Find names in the query using the keyword map
    for query_keyword in sorted_keywords:
        # Use word boundaries for better matching to avoid partial word matches
        regex_pattern = rf'\b{re.escape(query_keyword)}\b'
        match = re.search(regex_pattern, temp_query)
        if match:
            # Get the original (normalized) name corresponding to the keyword
            original_name = customer_keywords_map[query_keyword]
            if original_name not in found_original_names:
                 found_original_names.append(original_name)
                 # Replace found keyword in temp_query to prevent re-matching subsets
                 temp_query = temp_query[:match.start()] + "_"*len(query_keyword) + temp_query[match.end():]

    filter_name = None
    if len(found_original_names) == 1:
        filter_name = found_original_names[0]
        print(f"DEBUG [Filter]: SUCCESS - Identified single customer: '{filter_name}'")
    elif len(found_original_names) > 1:
        print(f"DEBUG [Filter]: Multiple customers found ({found_original_names}). Will use comparative logic.")
    else:
        print("DEBUG [Filter]: No specific customer detected.")

    # Return the potential filter name AND the list of all unique names found
    return filter_name, list(set(found_original_names)) # Use set to ensure uniqueness
# --- End Helper ---


# --- RRF Function (Unchanged) ---
# [ ... Keep the reciprocal_rank_fusion function exactly as it was ... ]
def reciprocal_rank_fusion(
    results: List[List[Tuple[Document, float]]], k: int = RRF_K
) -> List[Tuple[Document, float]]:
    """
    Performs Reciprocal Rank Fusion on multiple ranked lists of (doc, score).
    Lower k in RRF means higher importance given to rank.
    Returns a list of (doc, score) sorted by fused score.
    Assumes higher scores are better for input lists.
    """
    fused_scores = {}
    doc_map = {} # Store doc objects by a unique key (metadata + hash)

    for docs_with_scores in results:
        if not docs_with_scores:
            continue

        for rank, item in enumerate(docs_with_scores):
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            doc, score = item
            if not isinstance(doc, Document):
                 continue

            source = doc.metadata.get('source', 'UnknownSource')
            page = doc.metadata.get('page_number', 'UnknownPage')
            try: content_hash = hash(doc.page_content)
            except TypeError: content_hash = hash(str(doc.page_content)) # Fallback
            doc_key = f"{source}-{page}-{content_hash}"

            if doc_key not in doc_map: doc_map[doc_key] = doc
            if doc_key not in fused_scores: fused_scores[doc_key] = 0.0
            try: fused_scores[doc_key] += 1.0 / (rank + k)
            except ZeroDivisionError: print(f"WARN [RRF]: Encountered rank 0 and k=0? Skipping item {doc_key}")

    reranked_results_keys = sorted(fused_scores.keys(), key=fused_scores.get, reverse=True)
    reranked_results = []
    for key in reranked_results_keys:
        if key in doc_map:
             fused_score = fused_scores.get(key, 0.0)
             reranked_results.append((doc_map[key], float(fused_score)))

    return reranked_results
# --- End RRF ---


# --- Function to format docs for debug output (Unchanged) ---
# [ ... Keep the format_docs_for_debug function exactly as it was ... ]
def format_docs_for_debug(docs: List[Document], scores: List[float] = None) -> List[Dict[str, Any]]:
    """Helper to format documents and optional scores for JSON output."""
    output = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, Document):
            print(f"WARN [DebugFormat]: Skipping non-Document item at index {i}: {type(doc)}")
            continue
        item = {
            "rank": i + 1,
            "content_preview": doc.page_content[:250] + "..." if doc.page_content else "",
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
        }
        if scores and i < len(scores):
             score_val = scores[i]
             try: item["score"] = float(score_val)
             except (ValueError, TypeError): item["score"] = str(score_val) # Fallback
        output.append(item)
    return output
# --- End Format Helper ---


# --- UPDATED Smart Selection Function (Unchanged) ---
# [ ... Keep the select_smart_chunks function exactly as it was ... ]
def select_smart_chunks(
    ranked_docs_with_scores: List[Tuple[Document, float]], # Input now contains combined scores
    target_customers: List[str],
    max_chunks: int, # This will now be the dynamically calculated K for comparative queries
    reranking_performed: bool, # Keep track if reranking was done for logging/thresholds
    min_docs_per_target: int = 2
) -> List[Document]:
    """
    Selects up to max_chunks documents for processing using Round-Robin for comparative queries.
    Prioritizes overall best combined scores first, then uses customer-specific lists for round-robin.
    """
    if not ranked_docs_with_scores:
        return []

    target_customer_list = sorted(list(set(target_customers)))
    is_comparative = len(target_customer_list) > 1

    if not is_comparative:
         print("ERROR [SelectChunks]: This function should only be called for comparative queries.")
         selected_docs_list = [doc for doc, score in ranked_docs_with_scores[:max_chunks]]
         return selected_docs_list

    score_type_label = "CombinedScore" if reranking_performed else "RRFScore"
    selected_docs_list: List[Document] = []
    customer_doc_counts: Dict[str, int] = {cust: 0 for cust in target_customer_list}
    docs_added_keys: Set[str] = set()

    # Pass 1
    for doc, score in ranked_docs_with_scores:
        if len(selected_docs_list) >= max_chunks: break
        customer = doc.metadata.get('customer', 'Unknown Customer')
        if customer in target_customer_list and customer_doc_counts[customer] < min_docs_per_target:
            source = doc.metadata.get('source', 'UnknownSource')
            page = doc.metadata.get('page_number', 'UnknownPage')
            try: content_hash = hash(doc.page_content)
            except TypeError: content_hash = hash(str(doc.page_content))
            doc_key = f"{source}-{page}-{content_hash}"
            if doc_key not in docs_added_keys:
                selected_docs_list.append(doc)
                docs_added_keys.add(doc_key)
                customer_doc_counts[customer] += 1
        all_targets_met_min = all(count >= min_docs_per_target for count in customer_doc_counts.values())
        if all_targets_met_min:
             break

    # Pass 2
    if len(selected_docs_list) < max_chunks:
        customer_specific_ranked_docs: Dict[str, List[Tuple[Document, float]]] = {cust: [] for cust in target_customer_list}
        for doc, score in ranked_docs_with_scores:
            customer = doc.metadata.get('customer', 'Unknown Customer')
            if customer in target_customer_list:
                customer_specific_ranked_docs[customer].append((doc, score))

        customer_next_candidate_index: Dict[str, int] = {cust: 0 for cust in target_customer_list}
        customer_cycle = itertools.cycle(target_customer_list)
        num_targets = len(target_customer_list)
        attempts_since_last_add = 0

        while len(selected_docs_list) < max_chunks and attempts_since_last_add < num_targets:
            current_customer = next(customer_cycle)
            customer_list = customer_specific_ranked_docs.get(current_customer, [])
            start_index = customer_next_candidate_index[current_customer]
            found_doc_for_customer = False
            for i in range(start_index, len(customer_list)):
                doc, score = customer_list[i]
                source = doc.metadata.get('source', 'UnknownSource')
                page = doc.metadata.get('page_number', 'UnknownPage')
                try: content_hash = hash(doc.page_content)
                except TypeError: content_hash = hash(str(doc.page_content))
                doc_key = f"{source}-{page}-{content_hash}"
                if doc_key not in docs_added_keys:
                    selected_docs_list.append(doc)
                    docs_added_keys.add(doc_key)
                    customer_next_candidate_index[current_customer] = i + 1
                    found_doc_for_customer = True
                    attempts_since_last_add = 0
                    break
            if not found_doc_for_customer:
                customer_next_candidate_index[current_customer] = len(customer_list)
                attempts_since_last_add += 1
            else:
                 if len(selected_docs_list) >= max_chunks:
                     break
    return selected_docs_list
# --- End Smart Selection ---


# --- NEW: SPLADE Helper Functions (Unchanged) ---
# [ ... Keep the get_splade_vector and search_splade functions exactly as they were ... ]
def get_splade_vector(text: str, model, tokenizer) -> Dict[int, float]:
    """Generates a sparse vector for a single text using SPLADE."""
    device = model.device
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        rep = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])[0]
        rep_activated = torch.relu(rep)
        pooled_rep, _ = torch.max(rep_activated * tokens['attention_mask'].unsqueeze(-1), dim=1)
        pooled_rep_log_sat = torch.log1p(pooled_rep)

    vec = pooled_rep_log_sat.cpu().numpy().squeeze() # Remove batch dim
    indices = vec.nonzero()[0]
    weights = vec[indices]
    sparse_dict = dict(zip(indices.tolist(), weights.tolist()))
    return sparse_dict

def search_splade(
    query_vector: Dict[int, float],
    doc_vectors: List[Dict[int, float]],
    documents: List[Document],
    top_n: int
) -> List[Tuple[Document, float]]:
    """Performs sparse search using dot product."""
    scores = []
    for i, doc_vec in enumerate(doc_vectors):
        score = 0.0
        for token_id, query_weight in query_vector.items():
            if token_id in doc_vec:
                score += query_weight * doc_vec[token_id]
        if score > 0:
             scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    results = []
    valid_indices = {i for i in range(len(documents))}
    for score, index in scores[:top_n]:
        if index in valid_indices:
             results.append((documents[index], score))
        else:
             print(f"WARN [SearchSPLADE]: Index {index} out of bounds for documents list (len={len(documents)}). Skipping.")
    return results
# --- End SPLADE Helpers ---


@main_blueprint.route("/", methods=["GET", "POST"])
def home():
    query_for_template = ""
    answer_for_template = ""
    sources_for_template = None
    email_for_template = ""

    if request.method == "POST":
        request_start_time = time.time()

        debug_retrieval = False
        user_query = ""
        user_email = ""

        # Handle JSON or Form data
        if request.is_json:
            try:
                data = request.get_json()
                user_query = data.get("query", "").strip()
                user_email = data.get("email", "").strip()
                debug_retrieval = data.get("debug_retrieval", False) is True
            except Exception as e:
                 print(f"ERROR: Failed to parse JSON request body: {e}")
                 return jsonify({"error": "Invalid JSON payload"}), 400
        else:
            user_query = request.form.get("query", "").strip()
            user_email = request.form.get("email", "").strip()
            debug_retrieval = request.form.get("debug_retrieval") == "true"

        query_for_template = user_query
        email_for_template = user_email

        answer = "An error occurred processing your request."
        sources = []
        docs_before_rerank: List[Tuple[Document, float]] = []

        print(f"\n--- NEW REQUEST ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
        print(f"User Query: {user_query}")
        print(f"User Email: {user_email}")
        print(f"Debug Retrieval Mode: {debug_retrieval}")
        reranker_loaded = hasattr(qa_module, 'reranker_model') and qa_module.reranker_model is not None
        splade_ready = qa_module.splade_model is not None and qa_module.splade_tokenizer is not None and qa_module.splade_vectors is not None and qa_module.splade_docs is not None
        print(f"Reranker Enabled in Config: {RERANKER_ENABLED}")
        print(f"Reranker Model Loaded: {reranker_loaded}")
        print(f"SPLADE Model/Data Loaded: {splade_ready}")
        should_rerank = RERANKER_ENABLED and reranker_loaded # Determine if reranking should happen

        if not user_query:
            answer = "Please enter a valid query."
            sources = None
            if request.is_json: return jsonify({"error": answer}), 400
            else:
                answer_for_template = answer; sources_for_template = sources
                return render_template("index.html", query=query_for_template, answer=answer_for_template, sources=sources_for_template, email=email_for_template), 400

        core_components_missing = (
            qa_module.retriever is None or
            not splade_ready or
            qa_module.map_reduce_chain is None
        )
        # Check if reranker is enabled BUT failed to load
        reranker_missing_when_enabled = (RERANKER_ENABLED and not reranker_loaded)

        if core_components_missing or reranker_missing_when_enabled:
             error_details = []
             if qa_module.retriever is None: error_details.append("FAISS retriever missing.")
             if not splade_ready: error_details.append("SPLADE components missing.")
             if qa_module.map_reduce_chain is None: error_details.append("MapReduce chain missing.")
             if reranker_missing_when_enabled: error_details.append("Reranker enabled but model failed to load.")
             error_msg = f"System not ready. Details: {' '.join(error_details)} Please check initialization and config."
             print(f"ERROR: {error_msg}")
             if request.is_json: return jsonify({"error": error_msg}), 500
             else:
                 answer_for_template = f"Error: {error_msg}"; sources_for_template = None
                 return render_template("index.html", query=query_for_template, answer=answer_for_template, sources=sources_for_template, email=email_for_template), 500

        try:
            tracer = EmailLangChainTracer(project_name=PROJECT_NAME)
            callback_manager = CallbackManager([tracer])
        except NameError:
             callback_manager = CallbackManager([])
        except Exception as e:
            print(f"Error initializing tracer: {e}")
            callback_manager = CallbackManager([])

        filter_customer_name, target_customers_in_query = get_customer_filter_keyword(user_query)

        t_start_pipeline = time.time()
        t_last = t_start_pipeline

        try:
            # --- HYBRID SEARCH PIPELINE ---

            # 1. Dense Retrieval (FAISS)
            # [ ... FAISS retrieval logic ... ]
            dense_results_for_rrf: List[Tuple[Document, float]] = []
            try:
                retrieved_dense_docs = qa_module.retriever.invoke( user_query, config={"callbacks": callback_manager} )
                if retrieved_dense_docs and isinstance(retrieved_dense_docs[0], tuple) and len(retrieved_dense_docs[0]) == 2:
                     dense_docs_with_scores = retrieved_dense_docs
                elif retrieved_dense_docs and isinstance(retrieved_dense_docs[0], Document):
                     dense_docs_with_scores = [(doc, 1.0) for doc in retrieved_dense_docs]
                else: dense_docs_with_scores = []
                dense_results_for_rrf = dense_docs_with_scores
            except Exception as faiss_err:
                 print(f"ERROR during FAISS retrieval: {faiss_err}"); traceback.print_exc()
                 dense_results_for_rrf = []
            t_now = time.time(); print(f"DEBUG Timing: Dense Retrieval (FAISS k={FAISS_TOP_K}) took {t_now-t_last:.4f}s"); t_last = t_now

            # 2. Sparse Retrieval (SPLADE)
            # [ ... SPLADE retrieval logic ... ]
            sparse_results_for_rrf: List[Tuple[Document, float]] = []
            try:
                if qa_module.splade_model and qa_module.splade_tokenizer and qa_module.splade_vectors and qa_module.splade_docs:
                    query_sparse_vector = get_splade_vector(user_query, qa_module.splade_model, qa_module.splade_tokenizer)
                    sparse_results_for_rrf = search_splade(query_sparse_vector, qa_module.splade_vectors, qa_module.splade_docs, SPLADE_TOP_N)
                else:
                    print("WARN: SPLADE model/data not loaded. Skipping SPLADE retrieval.")
            except Exception as splade_err:
                print(f"ERROR during SPLADE retrieval: {splade_err}"); traceback.print_exc()
                sparse_results_for_rrf = []
            t_now = time.time(); print(f"DEBUG Timing: Sparse Retrieval (SPLADE n={SPLADE_TOP_N}) took {t_now-t_last:.4f}s"); t_last = t_now

            # 3. Combine Results (RRF)
            # [ ... RRF logic ... ]
            rrf_input_lists = []
            if dense_results_for_rrf: rrf_input_lists.append(dense_results_for_rrf)
            if sparse_results_for_rrf: rrf_input_lists.append(sparse_results_for_rrf)
            if not rrf_input_lists:
                print("WARN: No results from either FAISS or SPLADE to fuse.")
                fused_results = []
            else:
                fused_results = reciprocal_rank_fusion(rrf_input_lists)
            t_now = time.time(); print(f"DEBUG Timing: RRF Fusion (k={RRF_K}, produced {len(fused_results)} docs) took {t_now-t_last:.4f}s"); t_last = t_now


            # --- Get candidate pool size ---
            candidate_pool_size = RERANK_CANDIDATE_POOL_SIZE if should_rerank else DYNAMIC_K_MAX_CHUNKS
            initial_hybrid_candidates = fused_results[:candidate_pool_size]
            docs_before_rerank = initial_hybrid_candidates

            # --- 4. Reranking & Combined Score Calculation ---
            final_candidates_with_scores: List[Tuple[Document, float]] = [] # Renamed for clarity
            perform_rerank = should_rerank and initial_hybrid_candidates

            W_RRF = 0.4
            W_RERANK = 0.6

            if perform_rerank:
                # --- Reranking Logic (as before) ---
                # [ ... Keep the reranking and score combination logic exactly as it was ... ]
                reranker_model = qa_module.reranker_model
                reranker_model_name = getattr(qa_module, 'RERANKER_MODEL_NAME', 'Unknown Reranker')
                print(f"DEBUG: Reranking {len(initial_hybrid_candidates)} candidates using {reranker_model_name}...")
                try:
                    candidate_docs = [doc for doc, score in initial_hybrid_candidates]
                    doc_to_rrf_score = {id(doc): score for doc, score in initial_hybrid_candidates}
                    rerank_pairs = [[user_query, doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)] for doc in candidate_docs]
                    rerank_scores_raw = reranker_model.predict(rerank_pairs, show_progress_bar=False)
                    docs_with_all_scores: List[Tuple[Document, float, float]] = []
                    for i, doc in enumerate(candidate_docs):
                        rrf_score = doc_to_rrf_score.get(id(doc), 0.0)
                        rerank_score = rerank_scores_raw[i]
                        docs_with_all_scores.append((doc, float(rrf_score), float(rerank_score)))

                    # Normalize and combine scores
                    if len(docs_with_all_scores) > 1:
                        min_rrf = min(s[1] for s in docs_with_all_scores); max_rrf = max(s[1] for s in docs_with_all_scores)
                        min_rerank = min(s[2] for s in docs_with_all_scores); max_rerank = max(s[2] for s in docs_with_all_scores)
                        range_rrf = max_rrf - min_rrf; range_rerank = max_rerank - min_rerank
                        normalized_scores = []
                        for doc, rrf, rerank in docs_with_all_scores:
                            norm_rrf = (rrf - min_rrf) / range_rrf if range_rrf > 0 else 0.0
                            norm_rerank = (rerank - min_rerank) / range_rerank if range_rerank > 0 else 0.0
                            combined_score = (W_RRF * norm_rrf) + (W_RERANK * norm_rerank)
                            normalized_scores.append((doc, combined_score))
                    elif docs_with_all_scores:
                         normalized_scores = [(docs_with_all_scores[0][0], docs_with_all_scores[0][2])] # Use rerank score if only one doc
                    else:
                         normalized_scores = []

                    normalized_scores.sort(key=lambda x: x[1], reverse=True)
                    final_candidates_with_scores = normalized_scores # Assign combined scores
                except Exception as rerank_err:
                    print(f"ERROR during reranking or score combination: {rerank_err}"); traceback.print_exc()
                    print("WARN: Falling back to using RRF results due to error.")
                    final_candidates_with_scores = sorted(initial_hybrid_candidates, key=lambda x: x[1], reverse=True) # Use RRF scores
                    perform_rerank = False # Mark reranking as failed/skipped
                t_now = time.time(); print(f"DEBUG Timing: Reranking (Pool={len(initial_hybrid_candidates)}) took {t_now-t_last:.4f}s"); t_last = t_now
            else:
                # Reranking is disabled or model not loaded - use RRF scores directly
                final_candidates_with_scores = sorted(initial_hybrid_candidates, key=lambda x: x[1], reverse=True)
                perform_rerank = False # Explicitly set to False
                print(f"DEBUG Timing: Reranking skipped.")
                # t_last remains unchanged

            # --- <<<< MODIFIED DYNAMIC K LOGIC START (Option 1) >>>> ---
            # --- 5. Determine Candidate Pool for Dynamic K ---
            # Filter candidates if it's a single-customer query
            candidates_for_dynamic_k = []
            is_single_customer_query = False
            if filter_customer_name and len(target_customers_in_query) == 1:
                is_single_customer_query = True
                customer_specific_candidates = [
                    (doc, score) for doc, score in final_candidates_with_scores # Use final scores (combined or RRF)
                    if doc.metadata.get('customer') == filter_customer_name
                ]
                candidates_for_dynamic_k = customer_specific_candidates
            else:
                # Use the full list (sorted by combined or RRF score)
                candidates_for_dynamic_k = final_candidates_with_scores
                is_single_customer_query = False

            # --- 6. Calculate Dynamic K ---
            dynamic_k = DYNAMIC_K_MAX_CHUNKS # Default to max
            if DYNAMIC_K_ENABLED:
                # Determine which threshold and score type to use
                if perform_rerank: # Check if reranking actually happened
                    threshold = DYNAMIC_K_COMBINED_SCORE_THRESHOLD # Use threshold for combined scores
                    score_type = "CombinedScore"
                else:
                    threshold = DYNAMIC_K_RRF_SCORE_THRESHOLD # Use the RRF threshold
                    score_type = "RRFScore"

                print(f"DEBUG: Calculating Dynamic K using {score_type} (Threshold={threshold}, Min={DYNAMIC_K_MIN_CHUNKS}, Max={DYNAMIC_K_MAX_CHUNKS})...")
                count_above_threshold = 0
                # candidates_for_dynamic_k contains (doc, score) tuples
                # score is either combined score or RRF score depending on perform_rerank
                for doc, score in candidates_for_dynamic_k:
                    if score >= threshold:
                        count_above_threshold += 1
                    else:
                        # Optimization: Since scores are sorted, we can stop early
                        break
                dynamic_k = max(DYNAMIC_K_MIN_CHUNKS, count_above_threshold)
                dynamic_k = min(dynamic_k, DYNAMIC_K_MAX_CHUNKS)
                print(f"DEBUG: Calculated Dynamic K = {dynamic_k} (Found {count_above_threshold} above threshold using {score_type})")
            else:
                 # Dynamic K is disabled in config
                 dynamic_k = DYNAMIC_K_MAX_CHUNKS
                 print(f"DEBUG: Dynamic K disabled in config. Using fixed K = {dynamic_k}")

            # --- 7. Select Final Documents based on K ---
            # This logic remains the same, using the calculated dynamic_k
            selected_docs_for_processing = []
            if is_single_customer_query:
                selected_docs_for_processing = [doc for doc, score in candidates_for_dynamic_k[:dynamic_k]]
            elif len(target_customers_in_query) > 1: # Comparative query
                 selected_docs_for_processing = select_smart_chunks(
                     ranked_docs_with_scores=candidates_for_dynamic_k, # Pass candidates sorted by appropriate score
                     target_customers=target_customers_in_query,
                     max_chunks=dynamic_k,
                     reranking_performed=perform_rerank # Pass whether reranking happened
                 )
            else: # No specific customer detected
                 selected_docs_for_processing = [doc for doc, score in candidates_for_dynamic_k[:dynamic_k]]

            t_now = time.time(); print(f"DEBUG Timing: Dynamic K / Selection (Final K={dynamic_k}) took {t_now-t_last:.4f}s"); t_last = t_now
            # --- <<<< MODIFIED DYNAMIC K LOGIC END >>>> ---


            # --- Chain Execution ---
            # [ ... Chain execution logic remains the same ... ]
            docs_to_process = selected_docs_for_processing
            if not docs_to_process:
                 if is_single_customer_query:
                     answer = f"Could not find relevant documents specifically for '{filter_customer_name}' matching your query."
                 elif not answer or answer == "An error occurred processing your request.":
                    answer = f"Could not find relevant documents for your query after selection (Limit: {dynamic_k} chunks)."
                 sources = []
                 t_now = time.time(); print(f"DEBUG Timing: MapReduce Chain skipped (No docs)"); t_last = t_now
            else:
                processed_docs_for_map = []
                for doc in docs_to_process:
                    meta = doc.metadata if hasattr(doc, 'metadata') else {}
                    header_parts = [f"Source: {meta.get('source', 'Unknown')}", f"Page: {meta.get('page_number', 'N/A')}", f"Customer: {meta.get('customer', 'Unknown')}", f"Clause: {meta.get('clause', 'N/A')}"]
                    header = " | ".join(header_parts) + "\n---\n"
                    content = doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)
                    cleaned_content = re.sub(r'^```[a-zA-Z]*\n', '', content, flags=re.MULTILINE)
                    cleaned_content = re.sub(r'\n```$', '', cleaned_content, flags=re.MULTILINE)
                    processed_docs_for_map.append( Document(page_content=header + cleaned_content.strip(), metadata=meta) )

                chain_input = { "input_documents": processed_docs_for_map, "question": user_query }
                print(f"DEBUG: Invoking MapReduce chain with {len(processed_docs_for_map)} documents...")
                try:
                    chain_config = {"callbacks": callback_manager}
                    if user_email: chain_config["metadata"] = {"user_email": user_email}
                    result = qa_module.map_reduce_chain.invoke(chain_input, config=chain_config)
                    answer_raw = result.get("output_text", "Error: Could not generate answer from MapReduce chain.")
                    answer = answer_raw
                except Exception as e:
                     print(f"Error invoking MapReduce chain: {e}"); traceback.print_exc()
                     answer = "Error processing query via MapReduce chain."
                t_now = time.time(); print(f"DEBUG Timing: MapReduce Chain ({len(processed_docs_for_map)} docs) took {t_now-t_last:.4f}s"); t_last = t_now


            # --- Source Generation (Unchanged) ---
            # [ ... Source generation logic remains the same ... ]
            sources = []
            seen_sources = set()
            for doc in docs_to_process:
                meta = doc.metadata if hasattr(doc, 'metadata') else {}
                source_file = meta.get('source', 'Unknown Source')
                page_num = meta.get('page_number', 'N/A')
                customer_display = meta.get('customer', 'Unknown Customer')
                source_key = f"{source_file}|Page {page_num}"
                if source_key not in seen_sources:
                    source_str = f"{source_file} (Customer: {customer_display}) - Page {page_num}"
                    clause_display = meta.get('clause', None)
                    hierarchy_display = meta.get('hierarchy', [])
                    if clause_display and clause_display != 'N/A': source_str += f" (Clause: {clause_display})"
                    elif hierarchy_display and isinstance(hierarchy_display, list) and hierarchy_display:
                        try: source_str += f" (Section: {hierarchy_display[-1]})"
                        except IndexError: pass
                    sources.append(source_str)
                    seen_sources.add(source_key)


            # --- Final Formatting (Unchanged) ---
            # [ ... Final formatting logic remains the same ... ]
            is_error_answer = "Error:" in answer or "Could not find" in answer or "System not ready" in answer or "Found relevant documents, but none matched" in answer
            if not is_error_answer:
                if not isinstance(answer, str): answer = str(answer)
                try: answer = markdown.markdown(answer, extensions=['fenced_code', 'tables', 'nl2br'])
                except Exception as md_err: print(f"WARN: Markdown conversion failed: {md_err}. Returning raw answer.")
            elif not isinstance(answer, str): answer = str(answer)
            t_now = time.time(); print(f"DEBUG Timing: Source Gen/Formatting took {t_now-t_last:.4f}s"); t_last = t_now


        except Exception as e:
             print(f"Error during main processing block: {e}")
             traceback.print_exc()
             answer = "An unexpected error occurred while processing your query."
             sources = []

        # --- Return Response ---
        answer_for_template = answer
        sources_for_template = sources

        total_request_time = time.time() - request_start_time
        print(f"--- Request Complete --- Total Time: {total_request_time:.4f}s ---")

        if request.is_json: return jsonify({"answer": answer, "sources": sources})
        else: return render_template("index.html", query=query_for_template, answer=answer_for_template, sources=sources_for_template, email=email_for_template)

    # --- GET request (Unchanged) ---
    return render_template("index.html", query=query_for_template, answer=answer_for_template, sources=sources_for_template, email=email_for_template)