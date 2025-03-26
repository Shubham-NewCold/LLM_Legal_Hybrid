import re
from pyparsing import (
    Word, alphas, nums, Literal, Suppress, Optional, Combine, restOfLine,
    ParseException, ParserElement
)
from langchain_core.documents import Document

# Configure pyparsing defaults
ParserElement.setDefaultWhitespaceChars(" \t")

# Define pyparsing grammar components
roman_nums = Word("IVXLCDM")

def is_valid_clause(clause_num, title):
    """
    Return True if the clause number and title appear to be valid.
    Requires that the clause number is a number (or number with a dot)
    and falls within an expected range (e.g., 1 to 44),
    and that the title has at least 3 words.
    """
    if not re.match(r'^\d+(\.\d+)?$', clause_num):
        return False
    try:
        num = float(clause_num)
        if num < 1 or num > 44:  # Adjust as needed
            return False
    except ValueError:
        return False
    if len(title.split()) < 3:
        return False
    return True

# Header expression: captures a clause number and a title.
header_expr = (
    Optional(Suppress("Clause")) +
    Combine(Word(nums) + Optional(Literal(".") + Word(nums)))("clause_number") +
    Optional(Literal(" ")) +
    restOfLine("title")
)

# Grammar for schedule definitions.
schedule_def = (
    Suppress("Schedule") +
    Word(nums)("schedule_num") +
    Suppress("Part") +
    Word(nums)("part_num")
)

def is_spurious_line(line):
    """
    Return True if the line appears to be spurious (e.g., an isolated number or very short).
    """
    stripped = line.strip()
    return re.fullmatch(r'\d+', stripped) is not None or len(stripped.split()) < 2

def extend_title_if_incomplete(title, next_line):
    """
    If the header title ends with a conjunction, comma, or open parenthesis,
    extend it using up to 10 words from the next line.
    """
    incomplete_endings = ("or", "and", "but", "for", "nor", "yet", ",", "(")
    words = title.split()
    if words and words[-1].lower() in incomplete_endings:
        extra_words = next_line.split()[:10]
        title += " " + " ".join(extra_words)
    return title

def enrich_title_if_short(title, lines, start_index, target_word_count=10, max_extra_lines=3):
    """
    If the header title is shorter than target_word_count words,
    append additional text from subsequent lines (up to max_extra_lines)
    to provide more context.
    """
    word_count = len(title.split())
    extra_lines_used = 0
    idx = start_index
    # Only add extra text if the next line is not header-like.
    while word_count < target_word_count and idx < len(lines) and extra_lines_used < max_extra_lines:
        next_line = lines[idx].strip()
        if next_line and not is_spurious_line(next_line) and not new_header_re.match(next_line):
            title += " " + next_line
            word_count = len(title.split())
            extra_lines_used += 1
        else:
            break
        idx += 1
    return title

def clean_trailing_punctuation(title):
    """
    Remove trailing punctuation like commas, semicolons, or colons from the title.
    """
    return title.rstrip(" ,;:")

# Improved regex: require a word boundary after the clause number.
new_header_re = re.compile(r'^\d+(\.\d+)?\b')

def pyparse_hierarchical_chunk_text(full_text, source_name, page_number=None, extra_metadata=None):
    """
    Parse the full text using header_expr to capture hierarchy.
    Merges subsequent lines into the header title if they appear to be continuations.
    """
    lines = full_text.splitlines()
    documents = []
    current_chunk_lines = []
    hierarchy_stack = []  # Each element: (clause_id, title, level)

    def flush_chunk(overlap_lines=None):
        nonlocal current_chunk_lines, hierarchy_stack
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                metadata = {
                    "source": source_name,
                    "page_number": page_number,
                    "hierarchy": [item[0] for item in hierarchy_stack]
                }
                if hierarchy_stack:
                    metadata["clause"] = hierarchy_stack[-1][0]
                    metadata["clause_title"] = hierarchy_stack[-1][1]
                    # Inject extra metadata (e.g., customer name)
                if extra_metadata:
                    metadata.update(extra_metadata)
                documents.append(Document(page_content=chunk_text, metadata=metadata))
            current_chunk_lines.clear()
            if overlap_lines is not None:
                current_chunk_lines.extend(overlap_lines)

    def current_token_count():
        return sum(len(line.split()) for line in current_chunk_lines)

    # Thresholds (adjust as needed)
    MIN_TITLE_WORDS = 10         # Minimum words desired for a complete header title
    MAX_HEADER_TITLE_WORDS = 40  # Maximum words to keep in the clause title

    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip spurious lines.
        if is_spurious_line(line):
            i += 1
            continue
        try:
            # Try parsing the current line as a header.
            result = header_expr.parseString(line)
            header_lines = [line]
            j = i + 1
            # Merge subsequent lines if they are continuations (not headers and not spurious).
            while j < len(lines):
                next_line = lines[j]
                if new_header_re.match(next_line) or is_spurious_line(next_line):
                    break
                # No need for extra try/except here; simply add the line.
                header_lines.append(next_line)
                j += 1
            merged_header = " ".join(header_lines)
            # Parse the merged header.
            result = header_expr.parseString(merged_header)
            title = result.get("title", "").strip().lstrip(". ")
            if ":" in title:
                title = title.split(":", 1)[0].strip()
            # Enrich the title if too short.
            if len(title.split()) < MIN_TITLE_WORDS:
                title = enrich_title_if_short(title, lines, j, target_word_count=MIN_TITLE_WORDS)
            # Also try to extend using the next line if the title seems incomplete.
            if j < len(lines):
                title = extend_title_if_incomplete(title, lines[j])
            title = clean_trailing_punctuation(title)
            # Truncate the title if it is too long.
            title_words = title.split()
            if len(title_words) > MAX_HEADER_TITLE_WORDS:
                title = " ".join(title_words[:MAX_HEADER_TITLE_WORDS])
            clause_id = result["clause_number"]
            if not is_valid_clause(clause_id, title):
                # If the header is not valid, treat the merged header as part of the text.
                current_chunk_lines.append(merged_header)
                i = j
                continue
            flush_chunk()  # Flush previous chunk before starting a new header.
            level = clause_id.count('.') + clause_id.count('(')
            while hierarchy_stack and hierarchy_stack[-1][2] >= level:
                hierarchy_stack.pop()
            hierarchy_stack.append((clause_id, title, level))
            # Add the header line (merged_header) to the new chunk.
            current_chunk_lines.append(merged_header)
            i = j
        except ParseException:
            # If parsing fails, add the line as normal text.
            current_chunk_lines.append(line)
            i += 1

        # Check if current chunk exceeds maximum tokens.
        from config import CHUNK_MAX_TOKENS, OVERLAP_RATIO  # Import tunable parameters
        if current_token_count() > CHUNK_MAX_TOKENS:
            overlap_count = int(len(current_chunk_lines) * OVERLAP_RATIO)
            overlap_count = max(overlap_count, 1)
            overlap_lines = current_chunk_lines[-overlap_count:]
            flush_chunk(overlap_lines=overlap_lines)
    flush_chunk()
    return documents

# --- Cross-Reference Resolution functions ---
import networkx as nx
reference_graph = nx.DiGraph()

def normalize_reference(ref):
    ref = ref.strip()
    if ref.startswith("Clause"):
        return ref.replace("Clause", "").strip()
    if ref.startswith("Schedule"):
        parts = ref.replace("Schedule", "").split("Part")
        return "-".join([p.strip() for p in parts])
    return ref

def build_reference_map(doc_text, current_location=""):
    references = re.findall(r'\b(Clause\s[\d\.]+(?:\([a-zA-Z]+\))?|Schedule\s\d+\sPart\s\d+)', doc_text)
    for ref in references:
        source = current_location
        target = normalize_reference(ref)
        reference_graph.add_edge(source, target)
    return reference_graph

# --- Financial Formula Handling ---
formula_pattern = r'\\text{Balancing Payment} = \\left\(\\frac{A}{B} \\times \$2,600,000\\right\) - \$2,600,000'

def parse_formula(formula_str):
    return {
        'formula': formula_str,
        'variables': ['A', 'B'],
        'calculation': lambda A, B: ((A / B) * 2600000) - 2600000,
        'context': 'Schedule 1 Part 6 (Incentive Payment)'
    }

# --- Legal Document Parser Class ---
class LegalDocumentParser:
    def __init__(self):
        self.hierarchy_parser = self.parse_hierarchy
        self.financial_parser = self.extract_financials
        self.reference_builder = self.build_references

    def parse_hierarchy(self, text, source_name, page_number, extra_metadata=None):
        return pyparse_hierarchical_chunk_text(text, source_name, page_number, extra_metadata)

    def extract_financials(self, text):
        from .parser import parse_formula, formula_pattern
        tables = extract_tables_with_context(text)
        formulas = []
        formula_matches = re.findall(formula_pattern, text)
        for f in formula_matches:
            formulas.append(parse_formula(f))
        return {"tables": tables, "formulas": formulas}

    def build_references(self, text, current_location=""):
        return build_reference_map(text, current_location)

    def parse(self, text, source_name, page_number, extra_metadata=None):
        hierarchy_docs = self.hierarchy_parser(text, source_name, page_number, extra_metadata=extra_metadata)
        _ = self.extract_financials(text)
        self.build_references(text, current_location=f"{source_name}-Page{page_number}")
        return hierarchy_docs

def extract_tables_with_context(text):
    table_pattern = r'(\+[-+]+?\+[\n\r].*?\+[-+]+?\+)'
    tables = re.findall(table_pattern, text, re.DOTALL)
    extracted = []
    for tbl in tables:
        context_match = re.search(r'(Part\s\d+.*?)(?=\n\S)', text, re.DOTALL)
        context = context_match.group(1) if context_match else "N/A"
        related_clauses = []  # Placeholder for related clauses.
        extracted.append({
            'table': tbl,
            'context': context,
            'related_clauses': related_clauses
        })
    return extracted
