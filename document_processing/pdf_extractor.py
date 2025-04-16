# document_processing/pdf_extractor.py

import os
import re
import pymupdf # Keep for initial attempt and image extraction
import pymupdf4llm
from langchain_core.documents import Document
import traceback
import sys

# --- NEW: Import OCR and Image libraries ---
try:
    import pytesseract
    from PIL import Image
    import io # Needed for image byte handling
    OCR_AVAILABLE = True
    # Optional: If Tesseract isn't in PATH on Windows/macOS, specify its location
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example Windows path
    print("--- OCR Dependencies (pytesseract, Pillow) loaded successfully. ---")
except ImportError:
    print("WARN: pytesseract or Pillow not found. OCR fallback will be disabled.")
    print("Install them: pip install pytesseract Pillow")
    print("Ensure the Tesseract OCR engine is installed on your system (e.g., 'sudo apt install tesseract-ocr').")
    OCR_AVAILABLE = False

# --- LLM Imports ---
try:
    # Use AzureChatOpenAI as per previous changes
    from langchain_openai import AzureChatOpenAI
except ImportError:
    print("ERROR: langchain-openai not found. Please install it: pip install langchain-openai")
    sys.exit(1)


# --- Config Import (adjust path if needed, assuming config.py is in project root) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    # Import Azure config values
    from config import (
        TEMPERATURE, MAX_TOKENS, # Keep these
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, # Add Azure specifics
        AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME,
        # <<< ADDED >>>
        MAX_TOKENS_REGION_EXTRACTION # Make sure this is in your config.py if used
    )
    # Add checks for Azure keys
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
         raise ImportError("Required Azure OpenAI credentials (Endpoint, Key, Deployment Name) not found in config or .env file.")

except ImportError as e:
    # Fallback for MAX_TOKENS_REGION_EXTRACTION if not defined
    if 'MAX_TOKENS_REGION_EXTRACTION' not in locals():
        print("WARN: MAX_TOKENS_REGION_EXTRACTION not found in config, using default value 100.")
        MAX_TOKENS_REGION_EXTRACTION = 100 # Assign a default value
    # Check essential Azure credentials again before potentially exiting
    if 'AZURE_OPENAI_ENDPOINT' not in locals() or not AZURE_OPENAI_ENDPOINT or \
       'AZURE_OPENAI_API_KEY' not in locals() or not AZURE_OPENAI_API_KEY or \
       'AZURE_OPENAI_DEPLOYMENT_NAME' not in locals() or not AZURE_OPENAI_DEPLOYMENT_NAME:
        print(f"ERROR in pdf_extractor.py: Could not import required Azure settings from config.py: {e}")
        print("Ensure config.py is in the project root and Azure OpenAI vars (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME) are set in your .env file.")
        sys.exit(1) # Exit if essential Azure config is missing

# --- Define Service Provider Name Variations (to exclude in LLM prompt) ---
# <<< UPDATED LIST >>>
SERVICE_PROVIDER_NAMES_LOWER = [
    "newcold melbourne no.2 pty ltd",
    "newcold melbourne no 2 pty ltd",
    "newcold melbourne no.2 pty limited",
    "newcold melbourne no 2 pty limited",
    "newcold pty ltd", # Added from example
    "newcold",
    "newcold indianapolis, llc", # Added from example
    "newcold poland operations sp. z o.o.", # Added from example
    "newcold atlanta, llc", # Added from example
    "newcold atlanta operations, llc", # Added from example
    "newcold burley, llc", # Added from example
    "newcold burley operations, llc", # Added from example
    "newcold cooperatief u.a", # Added from example
    "newcold transport limited", # Added from example
    # Add variations without punctuation or common abbreviations
    "newcold indianapolis llc",
    "newcold poland operations sp z o o",
    "newcold atlanta llc",
    "newcold atlanta operations llc",
    "newcold burley llc",
    "newcold burley operations llc",
    "newcold cooperatief ua",
    "newcold transport ltd",
]
# Create a string representation for the LLM prompt
SERVICE_PROVIDER_EXCLUSION_STRING = ", ".join(f"'{name}'" for name in SERVICE_PROVIDER_NAMES_LOWER)

# --- Constants ---
MIN_TEXT_LENGTH_FOR_OCR_TRIGGER = 50 # If get_text() returns less than this, try OCR

# --- Helper function for basic cleaning of LLM output ---
# <<< REFINED FUNCTION >>>
def clean_llm_output_name(name: str) -> str:
    """Basic cleaning for LLM output names."""
    if not name: return ""
    # Remove surrounding quotes, whitespace, periods, commas, parentheses first
    cleaned = name.strip(' ,.()"\'')
    # Remove potential markdown like bold markers if the LLM adds them
    cleaned = cleaned.strip('*_')
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Remove trailing parenthesized text (like G.B.) - ensure it's truly at the end
    # Requires a space before the opening parenthesis
    cleaned = re.sub(r'\s+\([^)]+\)$', '', cleaned).strip()

    # Specific fix for Polish suffix (case-insensitive check)
    if cleaned.lower().endswith("sp. z o.o"):
        # Find the start of the suffix, case-insensitive
        idx = cleaned.lower().rfind("sp. z o.o")
        if idx != -1:
            cleaned = cleaned[:idx].strip(' ,.')
    elif cleaned.lower().endswith("sp z o o"):
        idx = cleaned.lower().rfind("sp z o o")
        if idx != -1:
            cleaned = cleaned[:idx].strip(' ,.')

    return cleaned.strip(' ,.') # Final strip
# --- END REFINED FUNCTION ---

# --- LLM-Powered Customer Finder ---
def find_customer_automatically_llm(first_page_text: str, llm_instance: AzureChatOpenAI) -> str:
    """
    Uses an Azure OpenAI LLM to identify the customer name from the first page text,
    excluding known service provider names, and normalizes the result.
    """
    print("  DEBUG [LLM Detect]: Starting Azure OpenAI LLM customer detection...")

    # --- Define the Prompt (Refined for robustness) ---
    # <<< REFINED PROMPT >>>
    prompt_template = f"""
Analyze the following text extracted from the first page of a legal agreement.
Your primary task is to identify the full legal name of the main counterparty to the service provider, often referred to as the 'Customer'.

**CRITICAL INSTRUCTION:** You MUST EXCLUDE the following entity names and any minor variations thereof: {SERVICE_PROVIDER_EXCLUSION_STRING}. These are service providers, not the customer.

Follow these steps:
1.  Read the text carefully to identify the parties involved. Look for sections like "Between", "Parties", or lists designating parties (often numbered like (1), (2)...). Pay attention to formatting.
2.  Identify the entity listed first (often designated with '(1)' or appearing first after a "Between" keyword). Call this the 'Potential Customer'. If there's an explicit designation like "(the 'Customer')", prioritize that entity as the 'Potential Customer'.
3.  Verify that the 'Potential Customer' name does NOT contain any of the excluded names: {SERVICE_PROVIDER_EXCLUSION_STRING}.
4.  If the 'Potential Customer' does NOT contain an excluded name, return ONLY its full legal name.
5.  If the first entity *does* contain an excluded name, look for the *next* listed entity (e.g., designated with '(2)' or second after 'Between'). Verify *this* entity is not excluded. If it is not excluded, return its full legal name.
6.  If multiple non-excluded potential customer names are found (e.g., parties (1), (3), (4) are not excluded), prioritize the one explicitly labeled 'Customer' or listed first (e.g., as '(1)'). If no such label exists, return the first non-excluded party listed (e.g., the one associated with '(1)').
7.  If, after careful analysis following these steps, you cannot identify a clear customer/counterparty name (other than the excluded service provider), return the exact phrase: Unknown Customer

**Output Format:** Return ONLY the identified full legal name OR the exact phrase "Unknown Customer". Do not include any other explanations, introductions, apologies, or surrounding text.

Text:
---
{first_page_text}
---

Identified Customer Name:
"""
    # <<< END REFINED PROMPT >>>

    try:
        print("  DEBUG [LLM Detect]: Invoking Azure OpenAI LLM...")
        response = llm_instance.invoke(prompt_template)
        extracted_name_raw = response.content if hasattr(response, 'content') else str(response)
        print(f"  DEBUG [LLM Detect]: Raw LLM Response: '{extracted_name_raw}'")

        # --- Post-process and Validate ---
        extracted_name_cleaned = clean_llm_output_name(extracted_name_raw)

        if not extracted_name_cleaned or extracted_name_cleaned.lower() == "unknown customer":
            print("  DEBUG [LLM Detect]: LLM indicated 'Unknown Customer' or returned empty.")
            return "Unknown Customer"

        # Check against exclusion list AFTER cleaning but BEFORE normalization
        # This check might be redundant if the LLM follows the prompt perfectly, but good safety net
        if extracted_name_cleaned.lower() in SERVICE_PROVIDER_NAMES_LOWER:
            print(f"  WARN [LLM Detect]: LLM returned an excluded service provider name ('{extracted_name_cleaned}') despite instructions. Overriding to 'Unknown Customer'.")
            return "Unknown Customer"

        print(f"  DEBUG [LLM Detect]: Cleaned Name (Before Normalization): '{extracted_name_cleaned}'")

        # ***** NORMALIZATION STEP *****
        normalized_name = extracted_name_cleaned # Start with the cleaned name

        # <<< ORDER SUFFIXES LONGEST TO SHORTEST >>>
        # <<< USING endswith INSTEAD OF REGEX FOR SIMPLICITY/ROBUSTNESS >>>
        suffixes_to_strip = sorted([
            ' Pty Ltd', ' Pty Limited', ' Ltd', ' Limited', ' Inc', ' LLC',
            ' plc',
            # Polish/Parenthesized suffixes handled by clean_llm_output_name now
        ], key=len, reverse=True)

        normalized_name_lower = normalized_name.lower() # Use lower case for matching suffix
        for suffix in suffixes_to_strip:
            # Check with and without leading space for robustness
            suffix_lower_with_space = " " + suffix.lower().strip()
            suffix_lower_no_space = suffix.lower().strip()

            if normalized_name_lower.endswith(suffix_lower_with_space):
                # Find the actual suffix in the original string to get correct length
                original_suffix = normalized_name[-len(suffix_lower_with_space):]
                normalized_name = normalized_name[:-len(original_suffix)].strip(' ,.')
                print(f"    DEBUG [LLM Normalize]: Stripped suffix '{original_suffix}'. Result: '{normalized_name}'")
                break # Stop after first suffix match
            elif normalized_name_lower.endswith(suffix_lower_no_space):
                 original_suffix = normalized_name[-len(suffix_lower_no_space):]
                 normalized_name = normalized_name[:-len(original_suffix)].strip(' ,.')
                 print(f"    DEBUG [LLM Normalize]: Stripped suffix '{original_suffix}'. Result: '{normalized_name}'")
                 break

        # Final cleanup after potential suffix stripping
        normalized_name = normalized_name.strip(' ,.')
        # <<< END REFINED NORMALIZATION LOOP >>>

        if normalized_name != extracted_name_cleaned:
             print(f"  DEBUG [LLM Detect]: Final Normalized Name: '{normalized_name}'")
        else:
             print(f"  DEBUG [LLM Detect]: Name already normalized or no suffix found: '{normalized_name}'")

        # Return the NORMALIZED name
        return normalized_name
        # *********************************

    except Exception as e:
        print(f"  ERROR [LLM Detect]: Azure OpenAI LLM invocation or processing failed: {e}")
        traceback.print_exc()
        return "Unknown Customer"

# --- OCR FUNCTION ---
def ocr_page_image(page: pymupdf.Page) -> str:
    """Performs OCR on a PyMuPDF page image."""
    if not OCR_AVAILABLE:
        print("    WARN [OCR]: OCR libraries not available.")
        return ""
    try:
        print("    DEBUG [OCR]: Rendering page to image for OCR...")
        zoom = 3
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_bytes))
        print("    DEBUG [OCR]: Performing OCR using Tesseract...")
        ocr_text = pytesseract.image_to_string(pil_image, lang='eng')
        print(f"    DEBUG [OCR]: OCR extracted {len(ocr_text)} characters.")
        return ocr_text
    except Exception as ocr_err:
        print(f"    ERROR [OCR]: OCR failed for page: {ocr_err}")
        traceback.print_exc()
        return ""

# --- Find Region (Country) Function ---
# <<< UPDATED FUNCTION: Find Country >>>
def find_region_for_customer_llm(text_first_n_pages: str, customer_name: str, llm_instance: AzureChatOpenAI) -> str:
    """
    Uses an LLM to find the COUNTRY associated with a specific customer name
    within the text of the first N pages, excluding service provider addresses.
    Returns "Unknown Region" if not found or ambiguous.
    """
    print(f"  DEBUG [Region Detect]: Starting LLM COUNTRY detection for customer: '{customer_name}'...")

    # Define the prompt specifically for country extraction
    region_prompt_template = f"""
Analyze the following text extracted from the first few pages of a legal agreement.
Your primary task is to find the COUNTRY where the specified Customer entity is located or registered.

**Customer Entity Name:** {customer_name}

**CRITICAL INSTRUCTION:** You MUST IGNORE any addresses or locations associated with the following service provider entity names (and minor variations): {SERVICE_PROVIDER_EXCLUSION_STRING}. Focus ONLY on the address linked to the Customer Entity Name provided above.

Follow these steps:
1.  Scan the text for mentions of the **Customer Entity Name:** '{customer_name}'.
2.  Locate the address details (like street, city, state, province, country, postal code) listed immediately near or explicitly linked to that Customer Entity Name. Look in sections like "Between", party lists (e.g., under "(1)"), or definitions.
3.  From the customer's full address, identify the COUNTRY.
4.  If multiple addresses are found for the *customer*, prioritize the one labeled "principal place of business" or "registered office". If no such label exists, use the first address found associated with the customer to determine the country.
5.  If you find a clear COUNTRY for the specified Customer Entity Name, return ONLY the name of that COUNTRY (e.g., "Australia", "United States", "Canada", "Poland", "Netherlands"). Standardize common names (e.g., return "United States" instead of "USA" or "US").
6.  If you cannot confidently determine the COUNTRY specifically for '{customer_name}' after careful analysis (e.g., the address is incomplete or only the service provider's country is found), return the exact phrase: Unknown Region

**Output Format:** Return ONLY the identified COUNTRY name OR the exact phrase "Unknown Region". Do not include any other explanations, introductions, apologies, or surrounding text.

Text (First Few Pages):
---
{text_first_n_pages}
---

Identified COUNTRY for {customer_name}:
"""

    try:
        print("  DEBUG [Region Detect]: Invoking Azure OpenAI LLM for COUNTRY...")
        response = llm_instance.invoke(region_prompt_template, config={'max_tokens': MAX_TOKENS_REGION_EXTRACTION})
        extracted_region_raw = response.content if hasattr(response, 'content') else str(response)
        print(f"  DEBUG [Region Detect]: Raw LLM Response: '{extracted_region_raw}'")

        # Basic cleaning for country name
        extracted_region_cleaned = extracted_region_raw.strip(' ,.()"\'*_')
        extracted_region_cleaned = re.sub(r'\s+', ' ', extracted_region_cleaned).strip()

        # Basic standardization (can be expanded)
        if extracted_region_cleaned.lower() in ['usa', 'us', 'united states of america']:
            extracted_region_cleaned = 'United States'
        elif extracted_region_cleaned.lower() in ['uk', 'united kingdom', 'great britain', 'g.b.']:
             extracted_region_cleaned = 'United Kingdom'
        # Add other standardizations if needed

        if not extracted_region_cleaned or extracted_region_cleaned.lower() == "unknown region":
            print("  DEBUG [Region Detect]: LLM indicated 'Unknown Region' or returned empty.")
            return "Unknown Region"
        else:
            # Optional: Validate against a known list of countries if desired
            print(f"  DEBUG [Region Detect]: Final Country Detected: '{extracted_region_cleaned}'")
            return extracted_region_cleaned

    except Exception as e:
        print(f"  ERROR [Region Detect]: Azure OpenAI LLM invocation or processing failed: {e}")
        traceback.print_exc()
        return "Unknown Region"
# --- <<< END UPDATED FUNCTION >>> ---


# --- Main PDF Extraction Function ---
def extract_documents_from_pdf(pdf_path):
    """
    Extract documents using PyMuPDF4LLM, uses Azure OpenAI LLM for customer detection
    (with OCR fallback) and region (country) detection.
    """
    file_name = os.path.basename(pdf_path)
    print(f"\n--- Processing PDF: {file_name} ---")
    customer_name_final = "Unknown Customer"
    region = "Unknown Region" # Default value
    documents = []
    pdf_doc = None
    pdf_metadata_from_pymupdf = {}
    llm_customer = None
    llm_region = None

    # --- Initialize LLM Instances ---
    try:
        llm_customer = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS, # Max tokens for customer name response
        )
        llm_region = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=TEMPERATURE, # Keep temp low for factual extraction
            max_tokens=MAX_TOKENS_REGION_EXTRACTION # Use specific limit for region
        )
        print("DEBUG [Extractor]: Initialized AzureChatOpenAI instance(s).")
    except Exception as llm_init_e:
        print(f"ERROR [Extractor]: Failed to initialize AzureChatOpenAI: {llm_init_e}.")

    try:
        # --- Step 1: Open PDF and Get First Page Text (with OCR fallback) ---
        first_page_text = ""
        MAX_PAGES_FOR_REGION = 5
        text_first_n_pages = ""

        print(f"DEBUG [Extractor]: Opening '{file_name}' with pymupdf...")
        pdf_doc = pymupdf.open(pdf_path)
        pdf_metadata_from_pymupdf = pdf_doc.metadata if pdf_doc else {}

        if len(pdf_doc) > 0:
            first_page = pdf_doc[0]
            print(f"DEBUG [Extractor]: Attempting direct text extraction from first page (page 0)...")
            first_page_text_direct = first_page.get_text("text", sort=True) or ""

            # --- OCR FALLBACK LOGIC ---
            first_page_text = first_page_text_direct
            if len(first_page_text.strip()) < MIN_TEXT_LENGTH_FOR_OCR_TRIGGER:
                print(f"WARN [Extractor]: Direct text extraction yielded minimal text ({len(first_page_text.strip())} chars). Attempting OCR fallback...")
                if OCR_AVAILABLE:
                    first_page_text_ocr = ocr_page_image(first_page)
                    if len(first_page_text_ocr.strip()) > len(first_page_text.strip()):
                        print("INFO [Extractor]: Using OCR text for customer detection.")
                        first_page_text = first_page_text_ocr
                    else:
                        print("WARN [Extractor]: OCR text was not significantly longer than direct text. Using direct text (if any).")
                else:
                    print("WARN [Extractor]: OCR not available or failed. Proceeding with minimal/no text from direct extraction.")
            else:
                 print(f"DEBUG [Extractor]: Direct text extraction successful ({len(first_page_text.strip())} chars).")
            # --- <<< END OCR FALLBACK LOGIC >>> ---

            print(f"\n--- Analyzing First Page Text (final) for: {file_name} ---")
            print(first_page_text[:2000] + ("..." if len(first_page_text) > 2000 else ""))
            print("--- End First Page Text Snippet ---\n")

            # ***** Step 1b: USE LLM TO FIND CUSTOMER NAME *****
            if llm_customer and first_page_text.strip():
                customer_name_final = find_customer_automatically_llm(first_page_text, llm_customer)
            elif not llm_customer:
                print("WARN [Extractor]: LLM (customer) not available, skipping customer detection.")
            else:
                print("WARN [Extractor]: No usable text extracted from first page. Skipping customer detection.")

            # ***** Step 1c: IF CUSTOMER FOUND, FIND REGION (COUNTRY) *****
            if customer_name_final != "Unknown Customer" and llm_region:
                print(f"DEBUG [Extractor]: Extracting text from first {MAX_PAGES_FOR_REGION} pages for region analysis...")
                try:
                    page_texts = []
                    num_pages_to_scan = min(len(pdf_doc), MAX_PAGES_FOR_REGION)
                    for page_num in range(num_pages_to_scan):
                        page = pdf_doc[page_num]
                        # Try direct text first, fallback to OCR for each page if needed for region
                        page_text_direct = page.get_text("text", sort=True) or ""
                        page_text = page_text_direct
                        if len(page_text.strip()) < MIN_TEXT_LENGTH_FOR_OCR_TRIGGER:
                             print(f"  WARN [Extractor/Region]: Low text on page {page_num+1}. Attempting OCR.")
                             if OCR_AVAILABLE:
                                 page_text_ocr = ocr_page_image(page)
                                 if len(page_text_ocr.strip()) > len(page_text.strip()):
                                     page_text = page_text_ocr
                        page_texts.append(page_text)

                    text_first_n_pages = "\n\n--- Page Break ---\n\n".join(page_texts)
                    print(f"DEBUG [Extractor]: Extracted text from {num_pages_to_scan} pages ({len(text_first_n_pages)} chars) for region analysis.")

                    if text_first_n_pages.strip():
                         # Call the function to find the COUNTRY
                         region = find_region_for_customer_llm(text_first_n_pages, customer_name_final, llm_region)
                    else:
                         print("WARN [Extractor]: Failed to extract any text from first few pages for region analysis.")

                except Exception as region_err:
                    print(f"ERROR [Extractor]: Failed during multi-page text extraction or region detection: {region_err}")
                    traceback.print_exc()
            elif customer_name_final == "Unknown Customer":
                 print("DEBUG [Extractor]: Skipping region detection because customer name is Unknown.")
            elif not llm_region:
                 print("WARN [Extractor]: LLM (region) not available, skipping region detection.")

        else:
            print(f"WARN [Extractor]: PDF '{file_name}' has no pages.")

        # --- Close PDF Document ---
        if pdf_doc:
            print(f"DEBUG [Extractor]: Closing PDF document '{file_name}'.")
            pdf_doc.close()
            pdf_doc = None

        print(f"DEBUG [Extractor]: Final Customer Name for Metadata: '{customer_name_final}', Region: '{region}' for '{file_name}'") # Log final results

        # --- Step 2: Extract clean markdown using pymupdf4llm ---
        print(f"DEBUG [Extractor]: Extracting markdown text using pymupdf4llm for '{file_name}' (forcing page chunks)...")
        md_text_data = []
        try:
            md_text_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=False)
        except Exception as md_err:
            print(f"ERROR [Extractor]: pymupdf4llm markdown extraction failed for {pdf_path}: {md_err}")
            traceback.print_exc()

        # --- Step 3: Create LangChain Documents ---
        if isinstance(md_text_data, list):
            for page_num_zero_based, page_item in enumerate(md_text_data):
                page_num_one_based = page_num_zero_based + 1
                page_content_str = ""
                if isinstance(page_item, dict):
                    page_content_str = page_item.get('text', page_item.get('content', str(page_item)))
                elif isinstance(page_item, str):
                    page_content_str = page_item
                else:
                    print(f"  ERROR [Extractor]: Page {page_num_one_based} item has unexpected type: {type(page_item)}. Skipping.")
                    continue

                metadata = {}
                metadata["source"] = file_name
                metadata["page_number"] = page_num_one_based
                metadata["customer"] = customer_name_final # <<< Assign final customer
                metadata["region"] = region # <<< Assign final region (Country)
                pdf_meta_cleaned = {k: v for k, v in pdf_metadata_from_pymupdf.items() if v is not None and isinstance(v, (str, int, float, bool))}
                metadata.update(pdf_meta_cleaned)

                try:
                    documents.append(Document(page_content=page_content_str, metadata=metadata))
                except Exception as doc_error:
                     print(f"  ERROR [Extractor]: Failed to create Document for page {page_num_one_based}. Error: {doc_error}")

        elif isinstance(md_text_data, str):
            print(f"WARN [Extractor]: pymupdf4llm returned a single string. Handling as single doc.")
            metadata = {"source": file_name, "page_number": 1, "customer": customer_name_final, "region": region} # <<< Add region
            pdf_meta_cleaned = {k: v for k, v in pdf_metadata_from_pymupdf.items() if v is not None and isinstance(v, (str, int, float, bool))}
            metadata.update(pdf_meta_cleaned)
            documents.append(Document(page_content=md_text_data, metadata=metadata))
        else:
             print(f"ERROR [Extractor]: Unexpected output format from pymupdf4llm for {file_name}: {type(md_text_data)}")


    except Exception as e:
        print(f"ERROR processing PDF {pdf_path}: {e}")
        traceback.print_exc()
        if pdf_doc: # Ensure closure even on outer error
            try: pdf_doc.close()
            except: pass
        return []

    print(f"DEBUG [Extractor]: Extracted {len(documents)} LangChain documents for '{file_name}'.")
    return documents

# --- Example Usage (__main__ block) ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_main = os.path.abspath(os.path.join(script_dir, '..'))
    pdf_directory = os.path.join(project_root_main, 'pdfs')
    print(f"--- PDF Extractor Test (Using Azure OpenAI w/ OCR Fallback & Country Detection) ---") # Updated log
    print(f"Looking for PDFs in: {pdf_directory}")

    if not os.path.isdir(pdf_directory):
        print(f"ERROR: PDF directory not found: {pdf_directory}")
    else:
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
        else:
            print(f"Found PDFs: {pdf_files}")
            # Test specific problematic files + one good one
            files_to_test = [f for f in pdf_files if f in ['mccainuk.pdf', 'MDLZ.pdf', 'pinaccle.pdf', 'conagra.pdf', 'nowel.pdf', 'fgf.pdf']] # Added nowel and fgf
            if not files_to_test: files_to_test = pdf_files[:1] # Fallback to first file if none match

            print(f"Testing extraction on: {files_to_test}")
            for pdf_file in files_to_test:
                pdf_path = os.path.join(pdf_directory, pdf_file)
                print(f"\n--- Processing {pdf_file} ---")
                extracted_docs = extract_documents_from_pdf(pdf_path)
                if extracted_docs:
                    print(f"Successfully extracted {len(extracted_docs)} documents for {pdf_file}.")
                    # Print detected customer and region from first doc's metadata
                    customer_meta = extracted_docs[0].metadata.get("customer", "Metadata Missing")
                    region_meta = extracted_docs[0].metadata.get("region", "Metadata Missing")
                    print(f"  Detected Customer (Assigned): '{customer_meta}'")
                    print(f"  Detected Region (Country):    '{region_meta}'") # Updated label
                else:
                    print(f"Extraction failed or returned no documents for {pdf_file}.")
            print("\n--- PDF Extractor Test Complete ---")