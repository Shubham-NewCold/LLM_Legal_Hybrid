from document_processing.parser import pyparse_hierarchical_chunk_text

sample_text = """
19. Disaster Recovery
19.1 NewCold shall, within 30 days of the earliest Services Commencement Date, put in place in accordance
with the provisions of Schedule 7 a customary and commercially reasonable plan of the business
continuity and recovery procedures to be followed by NewCold in the event of a Disaster (as such term
is defined in Clause 19.3 below) ( Disaster Recovery Plan"). The Disaster Recovery Plan shall be
such as to meet any reasonable requirement set by the Customer and shall be designed to ensure that
as far as reasonably practicable, despite any Disaster (including any Disaster that results from a Force
Majeure Event), the Services continue to be performed without interruption or derogation and in
accordance with this Agreement.
19.2 NewCold must comply with all relevant provisions of the Disaster Recovery Plan, and, where a Disaster
occurs, as soon as practicable deploy and act in accordance with the Disaster Recovery Plan.
19.3 If NewCold becomes aware of any event or circumstance which has or may lead to circumstances likely
to affect NewCold's ability to provide all or part of the Services (which is likely to have a material impact
for the Customer) in accordance with this Agreement (a "Disaster"), it shall notify the Customer as
soon as practicable and indicate the expected duration of such effect.
"""


chunks = pyparse_hierarchical_chunk_text(sample_text, source_name="sample.txt", page_number=1, extra_metadata={"customer": "Lactalis Australia", "region": "Australia"})
for idx, chunk in enumerate(chunks):
    print(f"--- Chunk {idx+1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
    print("-------------------\n")
