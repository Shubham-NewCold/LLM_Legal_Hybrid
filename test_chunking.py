from document_processing.parser import pyparse_hierarchical_chunk_text

sample_text = """
3.8 NewCold shall keep the Customer reasonably informed regarding the construction and
completion of the Warehouse. Without limitation, NewCold shall provide the Customer with
monthly progress reports and otherwise keep the Customer reasonably informed of material
delays and the steps NewCold is taking to address those delays.
4. Term
4.1 This Agreement shall come into force on the Signature Date and shall (unless terminated at an
earlier date in accordance with its terms) continue in force for a term of ten (10) years from the
Services Commencement Date (the "Initial Term").
4.2 The Customer shall be entitled to extend this Agreement from the end of the Initial Term for a
period of between six (6) months and five (5) years by providing not less than 12 months' written
notice to NewCold prior to the end of the Initial Term specifying the period of the extension. On
receipt of such a notice by NewCold, this Agreement will be extended by the period of the
extension specified in the notice. If the Customer does not provide such written notice at least
12 months prior to the end of the Initial Term, the Agreement will expire at the end of the Initial
Term.
5. Services to be provided by NewCold
5.1 NewCold shall provide the Services to the Customer with effect from the Services
Commencement Date in accordance with the terms of the Agreement, the Operating
Specification and the Standard KPIs save that, during the Initial Period, NewCold shall provide
the Services to the Customer in accordance with the Initial KPIs instead of the Standard KPIs.
5.2 NewCold shall carry out the Services in accordance with Best Industry Standards and, once the
parties have agreed the Quality Assurance Standards in accordance with this clause 5.2, the
Quality Assurance Standards. The Quality Assurance Standards, as drafted at the date of this
Agreement, are designed for a manual warehouse and require modification for an automated
warehouse. Accordingly, the parties will together update the standard and the updated standard,
once agreed, will from part of the Operating Specification. If there is any contradiction or
duplication between the Best Industry Standards and the Quality Assurance Standards, the
higher standard will apply. NewCold shall and shall procure that its Representatives engaged
in the provision of the Services observe all health, safety and security rules and procedures
applicable at the Warehouse or required under Applicable Law or this Agreement.
5.3 NewCold shall, from the Services Commencement Date, accept delivery of Products which it
has been notified by the Customer in accordance with Clause 7 (Forecasts) are to be stored at
the Warehouse, provided that such Products do not exceed either:
(a) the number of Pallet positions specified in the Forecast (as defined in Clause 7.3) and
updated under Clause 7.7 plus the buffer contemplated by Clause 7.6; or
(b) the Maximum Volume (unless otherwise agreed in accordance with Clause 7.4).
5.4 NewCold shall provide regular reports to the Customer (as set out in the Operating
Specification) on the number of Pallet positions available and remaining for use by the
Customer.
5.5 NewCold shall not provide the Services at any site other than the Warehouse without the prior
written consent of the Customer (such consent not to be unreasonably withheld, delayed or
conditioned).
5.6 The Customer may from time to time request (using the Service Change Procedure set out in
Schedule 3) that NewCold carry out other services separate from and in addition to the Standard
Services in relation to the Products provided such services are within the scope of services
generally provided by NewCold and can reasonably be accommodated (the "Additional
Services"). The charges and conditions (which must be reasonable and in accordance with market prices) for providing such Additional Services shall be agreed between the parties in
advance of such Additional Services being performed. 
5.7 The Customer acknowledges that the charges, conditions and specification of the Additional
Services are based on assumptions and information which may not be correct, or may cease to
be correct over time as circumstances change. If the actual costs incurred by NewCold in
providing any Additional Services exceed the charges agreed pursuant to Clause 5.6, then
NewCold shall be entitled to give Customer not less than 20 Business Days' notice in writing of
any proposed increase in the charges for any Additional Services, and the parties shall seek to
agree the increase to the charges, each acting in good faith. If the parties are unable to agree
the amount of the increase within such 20 Business Day period, then NewCold shall be entitled
to cease provision of the Additional Services on not less than 20 Business Days' notice.
6. Continuous Improvement
6.1 In order to provide an ongoing focus on:
(a) improving the standard of delivery of the Services to the Customer;
(b) developing sustainable productivity and customer service initiatives;
(c) improving the sustainability of the provision of the Services; and
(d) minimising the existing cost base to NewCold of providing the Services,
NewCold shall establish a team of appropriately skilled employees and experts, who will be
responsible for identifying continuous improvements throughout the Term, and, in consultation
with the Customer, developing and implementing a continuous improvement plan in relation to
the Services. NewCold shall, utilizing agreed evaluation processes, identify possible ways to
improve the quality of the Services and, together with the Customer, shall review the Minimum
Performance Targets and use reasonable endeavours to improve the performance of the
Services where necessary and commercially feasible to do so. Where commercially viable
opportunities for improvements are identified, they will be processed in accordance with Clause
10 (Service Change Procedure).
6.2 To support the continuous improvement plan, teams will be formed on different operational and
strategic levels with members both from NewCold and the Customer. These teams will meet
on a regular basis and will report to the steering committee established under Clause 6.3. The
overall objective of these meetings is to achieve cost effectiveness for the management of the
Products and quality improvement for the combined operation with associated savings being
shared between the parties as agreed during the relevant meeting.
6.3 The parties shall constitute a steering committee made up of representatives of each of the
parties to oversee the operation of the Services and the performance of this Agreement and to
consider proposals made by the teams referred to in Clause 6.2. The steering committee may
propose potential modifications to the Services and the Minimum Performance Targets. Such
proposals shall be managed in accordance with the process set out in Clause 10 (Service
Change Procedure).
7. Forecasts
7.1 The Customer has provided to NewCold the forecast set out in Schedule 8 for the period
commencing on the Services Commencement Date and ending on the last day of the Term
covering the storage, inbound, outbound, handling, case picking and other services required by
the Customer ("Term Forecast"). NewCold has configured the Warehouse size, reserved pallet
capacity for the Customer and made operational arrangements to service the Customer’s
requirements on the basis of the Term Forecast and the Customer’s Information. 
"""


chunks = pyparse_hierarchical_chunk_text(sample_text, source_name="sample.txt", page_number=1, extra_metadata={"customer": "Simplot Australia", "region": "Australia"})
for idx, chunk in enumerate(chunks):
    print(f"--- Chunk {idx+1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
    print("-------------------\n")
