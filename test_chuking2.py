# test_chunking.py
import os
import sys

# --- Add parent directory to sys.path to find config and document_processing ---
# Adjust the number of '..' based on your project structure if test_chunking.py
# is not directly in the root or a sibling directory to document_processing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- End Path Adjustment ---

from document_processing.parser import pyparse_hierarchical_chunk_text

# --- Sample Text (Unchanged) ---
sample_text = """
Background:
NewCold has agreed to provide the Services to the Customer at the Warehouse in respect of goods owned by
the Customer on the terms and conditions set out in this Agreement.
It is agreed as follows:
1. Definitions and interpretation
1.1 In this Agreement, unless the context otherwise requires, the following definitions shall apply:
"$" means the lawful currency of Australia;
"Additional Services has the meaning given to it in Clause 4.11;
"Affected Party" has the meaning given to it in Clause 20.2;
"Agreement" means this agreement including its Schedules;
"Ambient Products means finished food and drink goods which are manufactured or supplied by the
Customer and do not require refrigerated storage, including:
(a) milk powder;
(b) ultra heat treated (UHT) white milk, flavoured milk, milk alternatives, fruit juice and other dairy
products including cream; and
(c) such other products as agreed in writing between the parties from time to time.
Anti-Slavery Laws means:
(a) Division 270 and 271 of the Criminal Code Act 1995 (Cth);
(b) the Modern Slavery Act 2018 (Cth); and
(c) any other applicable law which prohibits exploitation of a worker, human trafficking, slavery,
slavery-like behaviour, servitude, forced labour, child labour, debt bondage or deceptive
recruiting for labour or sen/ices (or similar), and is applicable in the jurisdiction in which the
Customer and NewCold are registered or conduct business or in which activities relevant to
the supply of the Services are to be performed;
"Annual Forecast means the annual forecast for each of Ambient Products and Chilled Products
provided by the Customer in accordance with Clause 6.4;
"Applicable Law means any legislation including any statute, statutory instrument, treaty, regulation,
directive or by-law relating to the Services in force from time to time in the Territory;
Associated Entity has the meaning given to that term in the Corporations Act;
"Best Industry Standards" means standards, practices, methods and procedures in accordance with
the degree of skill, care, efficiency and timeliness as would reasonably be expected from a wellmanaged cold-store and ambient warehousing and logistics provider, performing services substantially
similar to the Services in the Territory;
"Business Day means any day other than a Saturday, Sunday, or public holiday in the Territory;
"Calendar Year" means a 12 month period beginning on 1 January and ending on 31 December;
1
Change in Control means the acquisition by any person (including the successors and permitted
assigns of the person), either alone or together with any Associated Entity of that person of:
(a) the power to direct the management or policies of NewCold; or
(b) where NewCold is a corporation, an interest in more than 50% of the issued voting capital of
NewCold; or
(c) where NewCold is the trustee of a unit trust, an interest in more than 50% of the issued units
in that trust; or
(d) where NewCold is the trustee of a discretionary trust, an interest as a taker in default of that
trust;
"Charges" means each of the charges for the Standard Services as set out in Schedule 1 Part 2 and
Part 3 and any charges for Additional Services notified to the Customer by NewCold in accordance
with Clause 4.11;
Charges Review Date has the meaning given to it in Schedule 1 (Part 4 - Charges Review
Mechanism);
Chilled Products" means finished goods which are fresh dairy products manufactured or supplied by
the Customer including yoghurts, desserts, mousses, custards, sour creams, hard cheeses (including
shredded and sliced), meal bases, soft serve ice cream, speciality cheeses (including soft cheeses),
butter and such other products as agreed in writing between the parties from time to time, which require
refrigerated storage and which can be refrigerated at any temperature between +0°C and +4°C, but
excluding fresh white milk and fresh flavoured milk;
"Confidential Information" has the meaning given to it in Clause 23.1;
"Corporations Act" means the Corporations Act 2001 (Cth);
Customer Group" means the Customer and its Associated Entities, and references to a member of
the Customer Group shall be construed accordingly;
"Customer's Information" means the information provided to NewCold by Customer as specified in
paragraph 1.2 of Schedule 2 (Part 2 -Operating Specification);
"Customer IPR" means patents, trademarks, design rights, copyright (including rights in computer
software and databases), know-how and moral rights and other intellectual property rights of any
member of the Customer Group, in each case whether registered or unregistered and including
applications for, and the right to apply for, the foregoing and all rights or forms of protection having
equivalent or similar effect to any of the foregoing which may subsist anywhere in the world;
"Delivered into NewCold's Custody" has the meaning given to it in Clause 13.2;
"Delivered out of NewCold's Custody" has the meaning given to it in Clause 13.3;
"Disaster" has the meaning given to it in Clause 19.3;
"Disaster Recovery Plan" has the meaning given to it in Clause 19.2;
"Disclosing Party" has the meaning given to it in Clause 23.1;
"Dispute" means any dispute or difference between the parties arising out of or in connection with thi
"""

# --- Call the parser ---
# Note: pyparse_hierarchical_chunk_text now returns a tuple: (documents, final_stack)
# We only need the documents list for this test script.
# The initial_stack is None because this is the start of processing this text.
documents_list, _ = pyparse_hierarchical_chunk_text(
    full_text=sample_text,
    source_name="sample.txt",
    page_number=1,
    # Pass the expected page-level metadata
    extra_metadata={"customer": "Simplot Australia", "region": "Australia"},
    initial_stack=None # Start with an empty stack for this text block
)

# --- Print the results ---
print(f"Total chunks created: {len(documents_list)}")
for idx, chunk in enumerate(documents_list):
    print(f"\n--- Chunk {idx+1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
    print("-" * 20)