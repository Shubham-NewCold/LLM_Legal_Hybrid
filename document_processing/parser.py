import re
from pyparsing import Word, alphas, nums, Literal, Suppress, Optional, OneOrMore, Group, Combine, restOfLine, ParseException, ParserElement
from langchain_core.documents import Document

# Configure pyparsing defaults
ParserElement.setDefaultWhitespaceChars(" \t")

# Define pyparsing grammar components
roman_nums = Word("IVXLCDM")
header_expr = (
    Optional(Suppress("Clause")) +
    Combine(Word(nums) + OneOrMore(Literal(".") + Word(nums)))("clause_number") +
    Optional(OneOrMore(Group(Suppress("(") + Word(alphas + "IVXLCDM" + alphas.upper()) + Suppress(")"))))("nested_items") +
    Optional(Literal(" ") + restOfLine("title"))
)

# Grammar for schedule definitions.
schedule_def = (
    Suppress("Schedule") +
    Word(nums)("schedule_num") +
    Suppress("Part") +
    Word(nums)("part_num")
)

def pyparse_hierarchical_chunk_text(full_text, source_name, page_number=None):
    """
    Parse the full markdown text using the pyparsing grammar (header_expr) to capture hierarchy.
    Maintains a stack to record the current clause path.
    """
    lines = full_text.splitlines()
    documents = []
    current_chunk_lines = []
    hierarchy_stack = []  # Each element: (clause_id, title, level)

    def flush_chunk(overlap_lines=None):
        nonlocal current_chunk_lines, hierarchy_stack
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            metadata = {
                "source": source_name,
                "page_number": page_number,
                "hierarchy": [item[0] for item in hierarchy_stack]
            }
            if hierarchy_stack:
                metadata["clause"] = hierarchy_stack[-1][0]
                metadata["clause_title"] = hierarchy_stack[-1][1]
            documents.append(Document(page_content=chunk_text, metadata=metadata))
            current_chunk_lines.clear()
            if overlap_lines is not None:
                current_chunk_lines.extend(overlap_lines)

    def current_token_count():
        return sum(len(line.split()) for line in current_chunk_lines)

    for line in lines:
        try:
            result = header_expr.parseString(line)
            flush_chunk()
            clause_id = result["clause_number"]
            if "nested_items" in result and result["nested_items"]:
                clause_id += "".join([f"({item[0]})" for item in result["nested_items"]])
            title = result.get("title", "").strip()
            level = clause_id.count('.') + clause_id.count('(')
            while hierarchy_stack and hierarchy_stack[-1][2] >= level:
                hierarchy_stack.pop()
            hierarchy_stack.append((clause_id, title, level))
            current_chunk_lines.append(line)
        except ParseException:
            current_chunk_lines.append(line)

        # Check if current chunk exceeds maximum tokens
        from config import CHUNK_MAX_TOKENS, OVERLAP_RATIO  # Import tunable parameters from config
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
    """
    Normalize a legal reference string.
    """
    ref = ref.strip()
    if ref.startswith("Clause"):
        return ref.replace("Clause", "").strip()
    if ref.startswith("Schedule"):
        parts = ref.replace("Schedule", "").split("Part")
        return "-".join([p.strip() for p in parts])
    return ref

def build_reference_map(doc_text, current_location=""):
    """
    Build a reference map from the document text.
    """
    references = re.findall(r'\b(Clause\s[\d\.]+(?:\([a-zA-Z]+\))?|Schedule\s\d+\sPart\s\d+)', doc_text)
    for ref in references:
        source = current_location
        target = normalize_reference(ref)
        reference_graph.add_edge(source, target)
    return reference_graph

# --- Financial Formula Handling ---
formula_pattern = r'\\text{Balancing Payment} = \\left\(\\frac{A}{B} \\times \$2,600,000\\right\) - \$2,600,000'

def parse_formula(formula_str):
    """
    Parse a financial formula string and return its details.
    """
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
        
    def parse_hierarchy(self, text, source_name, page_number):
        return pyparse_hierarchical_chunk_text(text, source_name, page_number)
    
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
    
    def parse(self, text, source_name, page_number):
        hierarchy_docs = self.hierarchy_parser(text, source_name, page_number)
        # You can further process financial data if needed:
        _ = self.extract_financials(text)
        self.build_references(text, current_location=f"{source_name}-Page{page_number}")
        return hierarchy_docs

def extract_tables_with_context(text):
    """
    Extract tables from text while preserving their context.
    Uses regex to find table-like structures.
    """
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