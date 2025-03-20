from flask import Blueprint, render_template, request
import langchain_utils.qa_chain as qa_module

main_blueprint = Blueprint("main", __name__)

@main_blueprint.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("query", "")
        if not user_query.strip():
            answer = "Please enter a valid query."
            sources = None
        else:
            result = qa_module.qa_chain.invoke({"query": user_query})
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            sources = []
            for doc in source_docs:
                clause = doc.metadata.get('clause', 'N/A')
                if clause != 'N/A':
                    sources.append(f"{doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page_number', 'Unknown')} (Clause: {clause})")
                else:
                    sources.append(f"{doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page_number', 'Unknown')}")
            import markdown
            answer = markdown.markdown(answer)
        return render_template("index.html", query=user_query, answer=answer, sources=sources)
    return render_template("index.html", query="", answer="", sources=None)