# -*- coding: utf-8 -*-

import re
import networkx as nx
from langchain_core.documents import Document
import os
import copy # Needed for deep copying the stack

# --- Configuration Import ---
try:
    # Assumes config.py is in the same directory (e.g., document_processing)
    from .config import (
        CHUNK_MAX_TOKENS,
        OVERLAP_RATIO,
        MIN_TITLE_WORDS,
        MAX_HEADER_TITLE_WORDS,
    )
    print("--- Successfully imported settings from ./config.py ---")
except ImportError:
    print("--- WARNING: Could not import settings from config.py. Using default values. ---")
    print("--- Ensure config.py exists and defines necessary variables (e.g., CHUNK_MAX_TOKENS). ---")
    # Adjusted default token limit based on evaluation
    CHUNK_MAX_TOKENS = 400
    OVERLAP_RATIO = 0.3
    MIN_TITLE_WORDS = 10
    MAX_HEADER_TITLE_WORDS = 40

print(f"--- EXECUTING PARSER: {os.path.abspath(__file__)} ---")
print(f"--- Using Config: CHUNK_MAX_TOKENS={CHUNK_MAX_TOKENS}, OVERLAP_RATIO={OVERLAP_RATIO}, MIN_TITLE_WORDS={MIN_TITLE_WORDS}, MAX_HEADER_TITLE_WORDS={MAX_HEADER_TITLE_WORDS} ---")


# --- Constants and Regular Expressions ---
HEADER_RE = re.compile(
    # Optional leading whitespace, optional non-capturing group for "Clause " prefix
    r'^\s*(?:Clause\s+)?'
    # Group 1: Capture number sequence. Handles multi-part (e.g., 1.1.1), single-dot (e.g., 1.1.), or plain number (e.g., 1)
    # Allows optional parenthesized letters like 1.1(a)
    r'((?:\d+\.)*\d+(?:\([a-zA-Z]+\))?)'
    # Separator: One or more spaces or tabs (must follow the number)
    r'[ \t]+'
    # Group 2: Title - Must start with a non-whitespace character and capture the rest of the line
    r'(\S.*)'
)
# Validation regex for the captured clause number itself (more precise)
CLAUSE_NUM_RE = re.compile(r'^(?:(?:\d+\.)*\d+(?:\([a-zA-Z]+\))?)$') # Anchored version
# Regex to detect DocuSign IDs (added for cleaning)
DOCUSIGN_RE = re.compile(r'DocuSign Envelope ID:', re.IGNORECASE)
# Regex to detect simple list items like (a), 1., i) etc. to avoid merging them into titles
LIST_ITEM_RE = re.compile(r'^\s*\(?[a-zA-Z0-9]+[\.\)]')
# Regex to detect simple page numbers or isolated digits
PAGE_NUM_RE = re.compile(r'^\s*\d+\s*$')
# Heuristic word count threshold for filtering header-only chunks - Increased slightly
HEADER_ONLY_CHUNK_MAX_WORDS = 8

# --- Helper Functions ---

def is_valid_clause(clause_num, title):
    """
    Return True if the clause number format (as captured) is valid and title is not empty
    and it doesn't look like a number extracted from mid-sentence or a schedule reference.
    """
    if not clause_num or not CLAUSE_NUM_RE.match(clause_num):
        # print(f"DEBUG: Invalid clause number format for validation: '{clause_num}'")
        return False
    if not title:
        # print(f"DEBUG: Title is empty for clause {clause_num}")
        return False

    # --- HEURISTIC TO REJECT FALSE POSITIVES ---
    # Check if clause_num is purely numeric (no dots/parens) like "4", "10", "12"
    if re.fullmatch(r'\d+', clause_num):
        is_likely_false_positive = False
        # Starts lowercase and is reasonably long? Likely false positive from sentence.
        if title and title[0].islower() and len(title.split()) > 1:
            is_likely_false_positive = True
        # Starts with '('? Likely false positive (like "10 (Service...)").
        elif title and title[0] == '(':
             is_likely_false_positive = True
        # Starts with "Part "? Likely a reference like "Schedule 4 Part 1".
        elif title and title.lower().startswith("part ") and len(title.split()) > 1:
             is_likely_false_positive = True

        if is_likely_false_positive:
             # print(f"DEBUG: Rejecting potential false positive header: Clause='{clause_num}', Title='{title[:50]}...'")
             return False
    # --- END HEURISTIC ---
    return True

def is_spurious_line(line):
    """Return True if the line appears to be spurious (e.g., page number, DocuSign ID)."""
    stripped = line.strip()
    if not stripped: return True
    # Check for DocuSign ID
    if DOCUSIGN_RE.search(stripped): return True
    # Check for simple page numbers
    if PAGE_NUM_RE.match(stripped): return True
    # Allow simple list markers like (1), (a) etc. when they are the *only* thing on the line
    if re.fullmatch(r'\([\da-zA-Z]+\)', stripped): return False
    # Add other checks for common spurious patterns if needed
    return False

def extend_title_if_incomplete(title, next_line, next_line_words=10):
    """Extend title if it ends with conjunctions, comma, etc., avoiding list items."""
    stripped_title = title.strip()
    if not stripped_title: return title
    incomplete_endings = ("or", "and", "but", "for", "nor", "yet", ",", "(", ":")
    words = stripped_title.split()
    # Check the very last word, stripping trailing punctuation from it for the check
    if words and words[-1].lower().rstrip('.,:;') in incomplete_endings:
        extra_words = next_line.strip().split()[:next_line_words]
        # Avoid merging if the next line looks like a list item
        if extra_words and not LIST_ITEM_RE.match(next_line.strip()):
            title += " " + " ".join(extra_words)
    return title

def enrich_title_if_short(title, lines, start_index, target_word_count=MIN_TITLE_WORDS, max_extra_lines=3):
    """Append subsequent non-header, non-spurious, non-list-item lines to short titles."""
    word_count = len(title.split())
    extra_lines_used = 0
    idx = start_index
    current_title = title
    while word_count < target_word_count and idx < len(lines) and extra_lines_used < max_extra_lines:
        next_line_raw = lines[idx]
        next_line_stripped = next_line_raw.strip()
        # Stop if the next line IS a header, is empty, or looks spurious
        if not next_line_stripped or HEADER_RE.match(next_line_raw.strip()) or is_spurious_line(next_line_stripped):
            break
        # Avoid merging if the next line looks like a sub-list item
        if LIST_ITEM_RE.match(next_line_stripped):
            break
        current_title += " " + next_line_stripped
        word_count = len(current_title.split())
        extra_lines_used += 1
        idx += 1
    return current_title

def clean_trailing_punctuation(title):
    """Remove trailing punctuation like commas, semicolons, or colons."""
    return title.rstrip(" ,;:")

def strip_markdown_emphasis(title):
    """Removes leading/trailing * or _ used for markdown bold/italics."""
    # Repeatedly strip in case of multiple chars like "**Title**"
    cleaned_title = title.strip()
    while len(cleaned_title) > 1 and cleaned_title[0] in '*_' and cleaned_title[-1] == cleaned_title[0]:
        cleaned_title = cleaned_title[1:-1]
    return cleaned_title.strip() # Final strip for safety

# --- Core Hierarchical Chunking Function ---

def pyparse_hierarchical_chunk_text(full_text, source_name, page_number=None, extra_metadata=None, initial_stack=None):
    """
    Parses text, chunks based on clauses/tokens, handles hierarchy state across pages,
    filters out header-only chunks, and prevents false header detection after token splits.
    """
    lines = full_text.splitlines()
    documents = []
    current_chunk_lines = []
    hierarchy_stack = initial_stack if initial_stack is not None else []
    # print(f"DEBUG: Initializing parser for page {page_number}. Initial Stack: {[item[0] for item in hierarchy_stack]}")

    def flush_chunk(overlap_lines=None, metadata_stack_override=None):
        """
        Appends the current chunk to documents and resets.
        Filters out chunks consisting only of a short header line.
        Ensures metadata is correctly assigned based on the current hierarchy stack
        or an override stack if provided (for flushing content before a header).
        """
        nonlocal current_chunk_lines, hierarchy_stack, documents # Ensure documents is modified
        if not current_chunk_lines: return

        chunk_text = "\n".join(current_chunk_lines).strip()

        # --- FILTERING LOGIC for Header-Only Chunks ---
        lines_in_chunk = chunk_text.splitlines()
        is_likely_header_only = False
        if len(lines_in_chunk) == 1:
            first_line_stripped = lines_in_chunk[0].strip()
            match = HEADER_RE.match(first_line_stripped)
            if match:
                 title_part = match.group(2) if match.group(2) else ""
                 # Heuristic: Clause number counts roughly as 1 word
                 word_count_heuristic = len(title_part.split()) + 1
                 if word_count_heuristic <= HEADER_ONLY_CHUNK_MAX_WORDS:
                     is_likely_header_only = True

        if is_likely_header_only:
            # print(f"DEBUG: Skipping header-only chunk: '{chunk_text[:80]}...'")
            current_chunk_lines.clear() # Discard it
            if overlap_lines: current_chunk_lines.extend(overlap_lines) # Keep overlap for next chunk
            return # Don't create the document
        # --- END FILTERING ---

        if chunk_text:
            # Create metadata, inheriting page-level metadata first
            metadata = extra_metadata.copy() if extra_metadata else {}
            metadata["source"] = source_name
            if page_number is not None: # Only add if provided
                metadata["page_number"] = page_number

            # Determine which stack to use for metadata
            active_stack = metadata_stack_override if metadata_stack_override is not None else hierarchy_stack

            # Add hierarchy information from the active stack state
            current_hierarchy = [item[0] for item in active_stack] # List of clause IDs
            metadata["hierarchy"] = current_hierarchy
            if active_stack:
                metadata["clause"] = active_stack[-1][0] # Cleaned ID of the current level
                metadata["clause_title"] = active_stack[-1][1] # Processed title of the current level
            else:
                 metadata["clause"] = None
                 metadata["clause_title"] = None # Or perhaps a default document title?

            # print(f"DEBUG: Flushing Chunk. Text starts: '{chunk_text[:50]}...'. Metadata: {metadata}")
            documents.append(Document(page_content=chunk_text, metadata=metadata))

        # Reset for next chunk
        current_chunk_lines.clear()
        if overlap_lines: current_chunk_lines.extend(overlap_lines)

    def get_current_token_count():
        """Estimates token count based on simple word split."""
        # Consider a more robust tokenizer if needed (e.g., tiktoken)
        return sum(len(line.split()) for line in current_chunk_lines)

    # --- Main Parsing Loop ---
    i = 0
    # Flag to indicate if the previous action was a token-based split,
    # used to prevent false header detection on the first line of the overlap.
    just_split_due_to_tokens = False

    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        # Skip lines that are completely empty or identified as spurious
        if not stripped_line or is_spurious_line(line):
            i += 1
            continue # Skip to next line

        # --- Process Line (Header or Content) ---
        match = HEADER_RE.match(stripped_line)
        processed_as_real_header = False # Track if we updated hierarchy

        # Check for header pattern AND ensure we didn't *just* split based on tokens
        if match and not just_split_due_to_tokens:
            # --- Potentially a real header ---
            clause_id_raw = match.group(1)
            potential_title = match.group(2)
            first_header_line_index = i

            # --- Try to merge subsequent lines for multi-line titles ---
            header_content_lines = [potential_title]
            j = i + 1
            while j < len(lines):
                next_line_raw = lines[j]; next_line_stripped = next_line_raw.strip()
                if not next_line_stripped or is_spurious_line(next_line_raw): break
                if HEADER_RE.match(next_line_raw.strip()): break
                if LIST_ITEM_RE.match(next_line_stripped): break
                header_content_lines.append(next_line_stripped)
                j += 1
            merged_title = " ".join(header_content_lines).strip()

            # --- Process and Validate Header Title ---
            processed_title = enrich_title_if_short(merged_title, lines, j, target_word_count=MIN_TITLE_WORDS)
            if j < len(lines): processed_title = extend_title_if_incomplete(processed_title, lines[j])
            processed_title = clean_trailing_punctuation(processed_title)
            title_words = processed_title.split()
            if len(title_words) > MAX_HEADER_TITLE_WORDS:
               processed_title = " ".join(title_words[:MAX_HEADER_TITLE_WORDS]) + "..."
            processed_title = strip_markdown_emphasis(processed_title)

            # --- Final Validation ---
            is_valid = is_valid_clause(clause_id_raw, processed_title)

            # --- DEBUG PRINTS for Header Processing ---
            # print(f"\n--- DEBUG: Header Check ---")
            # print(f"Line {first_header_line_index}: '{stripped_line}'")
            # print(f"Matched: clause_id_raw='{clause_id_raw}', potential_title='{potential_title}'")
            # print(f"Processed Title: '{processed_title}'")
            # print(f"Is Valid Clause: {is_valid}")
            # print(f"Just Split Due To Tokens: {just_split_due_to_tokens}")
            # --- END DEBUG PRINTS ---

            if is_valid:
                # --- Confirmed REAL header ---

                # **MODIFICATION START:** Flush preceding content *before* updating stack
                # Save the current stack state to assign to the preceding chunk
                metadata_stack_for_preceding_chunk = copy.deepcopy(hierarchy_stack)
                if current_chunk_lines: # Only flush if there's something before this header
                    # print(f"DEBUG: Flushing preceding content before header '{clause_id_raw}'. Using stack: {[item[0] for item in metadata_stack_for_preceding_chunk]}")
                    flush_chunk(metadata_stack_override=metadata_stack_for_preceding_chunk)
                    # current_chunk_lines is now empty after flush_chunk
                # **MODIFICATION END**

                # Now update the stack for the *new* header
                clause_id_cleaned = clause_id_raw.rstrip('.')
                level = clause_id_cleaned.count('.') + clause_id_cleaned.count('(')

                # --- DEBUG PRINTS for Stack Adjustment ---
                # print(f"Calculated Level: {level}")
                # print(f"Stack BEFORE pop check: {[item[0] for item in hierarchy_stack]}")
                # --- END DEBUG PRINTS ---

                # Adjust hierarchy stack based on level
                while hierarchy_stack and hierarchy_stack[-1][2] >= level:
                    # --- DEBUG PRINT for Pop ---
                    # print(f"  Condition met: Stack top level {hierarchy_stack[-1][2]} >= new level {level}. Popping {hierarchy_stack[-1][0]}")
                    # --- END DEBUG PRINT ---
                    hierarchy_stack.pop()

                # --- DEBUG PRINT for Append ---
                # print(f"Stack BEFORE append: {[item[0] for item in hierarchy_stack]}")
                # print(f"Appending: ('{clause_id_cleaned}', '{processed_title[:30]}...', {level})")
                # --- END DEBUG PRINT ---

                hierarchy_stack.append((clause_id_cleaned, processed_title, level))

                # --- DEBUG PRINT for Stack After Append ---
                # print(f"Stack AFTER append: {[item[0] for item in hierarchy_stack]}")
                # print(f"--- End Header Check ---")
                # --- END DEBUG PRINT ---

                # Add the original header line(s) to start the new chunk's content
                # current_chunk_lines should be empty here due to the flush above
                current_chunk_lines.extend(lines[first_header_line_index:j])
                i = j # Move index past the header lines
                processed_as_real_header = True
            else:
                # --- Invalid header pattern match ---
                # print(f"--- DEBUG: Invalid Header - Treating as Content ---") # DEBUG
                # Treat the line(s) that matched the pattern as content
                current_chunk_lines.extend(lines[first_header_line_index:j])
                i = j # Move index past these lines
        else:
            # --- Not a header OR immediately after split ---
            # if just_split_due_to_tokens and match: # DEBUG
            #     print(f"--- DEBUG: Header pattern matched but ignored due to just_split_due_to_tokens ---") # DEBUG
            # Treat as content
            current_chunk_lines.append(line)
            i += 1

        # --- Reset flag AFTER processing the line(s) for this iteration ---
        # Ensures the flag is only active for the very first line check after a split
        just_split_due_to_tokens = False

        # --- Check token limit AFTER adding the line(s) ---
        current_tokens = get_current_token_count()
        if current_tokens > CHUNK_MAX_TOKENS:
            # print(f"DEBUG: Chunk exceeds token limit ({current_tokens} > {CHUNK_MAX_TOKENS}) on page {page_number}. Splitting.")
            if len(current_chunk_lines) <= 1:
                # print(f"DEBUG: Warning: Single line exceeds CHUNK_MAX_TOKENS. Flushing as is.")
                flush_chunk(overlap_lines=None)
                # After flushing a single long line, reset the flag as we are starting fresh
                just_split_due_to_tokens = False
                continue # Start next iteration

            # Calculate overlap
            overlap_line_count = max(1, int(len(current_chunk_lines) * OVERLAP_RATIO))
            overlap_line_count = min(overlap_line_count, len(current_chunk_lines) - 1)

            lines_for_current_chunk = current_chunk_lines[:-overlap_line_count]
            lines_to_keep_for_next_chunk = current_chunk_lines[-overlap_line_count:]

            # Temporarily set current_chunk_lines for flushing the main part
            current_chunk_lines = lines_for_current_chunk
            # Set flag *before* flushing, so the next iteration knows it resulted from a split
            just_split_due_to_tokens = True
            # Flush the main part, passing the overlap lines to be used for the *next* chunk's start
            # Use the *current* hierarchy_stack for this chunk being flushed due to size
            flush_chunk(overlap_lines=lines_to_keep_for_next_chunk, metadata_stack_override=None)
            # current_chunk_lines is now correctly set with the overlap lines by flush_chunk
            # The flag 'just_split_due_to_tokens' is TRUE for the next iteration check

    # --- End of loop for the current page text ---
    # print(f"\nDEBUG: End of processing for page {page_number}.")
    # Flush any remaining text from this page
    flush_chunk()
    # print(f"DEBUG: Final stack state for page {page_number}: {[item[0] for item in hierarchy_stack]}")
    # Return documents generated from this page and the final stack state
    return documents, hierarchy_stack


# --- Other functions (reference, financial, parser class) ---
# These remain unchanged but are included for completeness
reference_graph = nx.DiGraph()
def normalize_reference(ref):
    ref = ref.strip().rstrip('.').strip()
    if ref.lower().startswith("clause"): return ref[len("clause"):].strip()
    if ref.lower().startswith("schedule"):
        parts = ref[len("schedule"):].lower().split("part")
        return "Schedule-" + "-".join([p.strip() for p in parts if p.strip()])
    return ref

def build_reference_map(doc_text, current_location=""):
    references = re.findall(r'\b(Clause\s[\d\.]+(?:\([a-zA-Z]+\))?|Schedule\s\d+\sPart\s\d+)\b', doc_text, re.IGNORECASE)
    source_node = normalize_reference(current_location)
    for ref in references:
        target_node = normalize_reference(ref)
        if source_node and target_node and source_node != target_node:
             reference_graph.add_edge(source_node, target_node)
    return reference_graph

formula_pattern = r'\\text{Balancing Payment}\s*=\s*\\left\(\\frac{A}{B}\s*\\times\s*\$?\s*2,600,000\\right\)\s*-\s*\$?\s*2,600,000'
def parse_formula(formula_str):
    if "Balancing Payment" in formula_str:
        return { 'formula': formula_str, 'variables': ['A', 'B'], 'calculation': '((A / B) * 2600000) - 2600000', 'context': 'Schedule 1 Part 6 (Incentive Payment)' }
    return None

def extract_tables_with_context(text):
    table_pattern = r'(\+[-+]+?\+[\n\r].*?\+[-+]+?\+)'
    tables = re.findall(table_pattern, text, re.DOTALL)
    extracted = []
    for tbl in tables:
        tbl_pos = text.find(tbl)
        context = "N/A"
        headers_before = list(HEADER_RE.finditer(text[:tbl_pos]))
        if headers_before:
            last_header_match = headers_before[-1]
            context_clause_id = last_header_match.group(1).rstrip('.')
            context = f"Clause {context_clause_id}: {last_header_match.group(2).strip()}"
        extracted.append({ 'table': tbl, 'context': context, 'related_clauses': [] })
    return extracted

class LegalDocumentParser:
    def __init__(self):
        self.hierarchy_parser = pyparse_hierarchical_chunk_text
        self.financial_parser = self.extract_financials
        self.reference_builder = self.build_references
        self.global_reference_graph = reference_graph # Use the module-level graph

    def extract_financials(self, text):
        tables = extract_tables_with_context(text)
        formulas = []; formula_matches = re.findall(formula_pattern, text, re.IGNORECASE | re.DOTALL)
        for f_match in formula_matches:
            parsed = parse_formula(f_match)
            if parsed: formulas.append(parsed)
        return {"tables": tables, "formulas": formulas}

    def build_references(self, text, current_location=""):
        # This method now directly modifies the module-level graph
        build_reference_map(text, current_location)
        return self.global_reference_graph

    # This parse method is primarily for if the class itself manages state,
    # which isn't the case here as qa_chain.py calls the function directly.
    # It's kept for potential future use or different integration patterns.
    def parse(self, text, source_name, page_number=None, extra_metadata=None, initial_stack=None):
        parsed_docs, final_stack = self.hierarchy_parser(
            text, source_name, page_number=page_number, extra_metadata=extra_metadata, initial_stack=initial_stack
        )
        # Potentially build references or extract financials here if needed per-call
        # self.build_references(text, ...)
        # financials = self.extract_financials(text)
        return parsed_docs, final_stack

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Ensure config.py is in the same directory as this script if running directly
    config_file = "config.py"
    if not os.path.exists(config_file):
         print(f"\n--- Creating dummy {config_file} for example usage ---")
         with open(config_file, "w") as f:
              # Use the adjusted defaults in the dummy file
              f.write(f"CHUNK_MAX_TOKENS = 400\n")
              f.write(f"OVERLAP_RATIO = {OVERLAP_RATIO}\n")
              f.write(f"MIN_TITLE_WORDS = {MIN_TITLE_WORDS}\n")
              f.write(f"MAX_HEADER_TITLE_WORDS = {MAX_HEADER_TITLE_WORDS}\n")

    print("\n--- Running Example Usage ---")
    # Using a slightly modified sample text to demonstrate filtering and cleaning
    sample_text = """
18. Record Keeping
18.1 NewCold must keep records.
DocuSign Envelope ID: 123-ABC-456
18.2 Records must be accurate.

19. Liability
This section outlines liability. It is very important.
19.1 Nothing in this Agreement shall be deemed to limit or exclude the liability of a party (the
"Defaulting Party") for:
(a) death or personal injury caused by the Defaulting Party's wilful act, negligence or default
or the wilful act, negligence or default of the Defaulting Party's Representatives;
(b) fraud or fraudulent misrepresentation of the Defaulting Party or its Representatives; or
(c) any other type of liability which cannot be validly limited or excluded at law.
19.2 Subject to Clause 19.1, neither party will be liable, whether in contract, tort (including
negligence), breach of statutory duty, under any indemnity or otherwise in connection with or
arising from this Agreement for any:
(a) loss of profits, revenues or business opportunities;
(b) depletion of goodwill or loss of reputation;
(c) loss of actual or anticipated savings; or
(d) any indirect or consequential loss;
save that nothing in this Clause 19.2 shall relieve Customer of its obligations to make payments
in accordance with this Agreement (including without limitation Customer's obligation to make
the Minimum Annual Storage Revenue payments) or any liability of the Customer in connection
with Minimum Storage Requirement.

20

21. Service Credits
21.1 KPIs must be met. Failure results in credits as per Schedule 4 Part 3. This part is long and might exceed the token limit causing a split based on the CHUNK_MAX_TOKENS setting which is currently set quite high for testing purposes but in a real scenario might be lower like 200 or 250 depending on the embedding model and desired granularity for retrieval augmentation generation frameworks like the one being built here requiring careful consideration of chunk size versus context preservation trade-offs especially with overlapping strategies defined by OVERLAP_RATIO.
21.2 Credits are pre-estimates of loss.

DocuSign Envelope ID: 789-DEF-GHI

22. Termination Clause
"""

    parser = LegalDocumentParser()
    # Simulate calling from qa_chain.py (passing initial_stack=None for first page)
    docs, final_stack = pyparse_hierarchical_chunk_text(
        sample_text,
        source_name="sample_main.txt",
        page_number=1, # Assuming this is all page 1 for the example
        extra_metadata={"customer": "Simplot Australia", "region": "Australia"},
        initial_stack=None
    )

    print("\n--- Final Chunk Output ---")
    print(f"Total chunks created: {len(docs)}")
    for idx, chunk in enumerate(docs):
        print(f"--- Chunk {idx+1} ---")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)
        print("-" * 20)

    # Optional: Clean up dummy config file
    # if os.path.exists(config_file) and f"CHUNK_MAX_TOKENS = 400" in open(config_file).read():
    #     print(f"\n--- Cleaning up dummy {config_file} ---")
    #     os.remove(config_file)