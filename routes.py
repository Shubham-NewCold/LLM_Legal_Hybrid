from flask import Blueprint, render_template, request, jsonify
import langchain_utils.qa_chain as qa_module
import markdown
from langchain_core.callbacks.manager import CallbackManager
from email_tracer import EmailLangChainTracer
from langchain_openai import AzureChatOpenAI  # Add this import
from config import (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, 
                    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, 
                    TEMPERATURE, MAX_TOKENS)  # Add these imports

main_blueprint = Blueprint("main", __name__)

@main_blueprint.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if the request is JSON (AJAX request)
        if request.is_json:
            data = request.get_json()
            user_query = data.get("query", "")
            user_email = data.get("email", "")
        else:
            user_query = request.form.get("query", "")
            user_email = request.form.get("email", "")

        if not user_query.strip():
            answer = "Please enter a valid query."
            sources = None
        else:
            # Create fresh callback manager for each request
            tracer = EmailLangChainTracer(project_name="pr-new-molecule-89")  # Added project_name
            callback_manager = CallbackManager([tracer])
            
            # Create LLM instance with callbacks for this specific request
            request_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                openai_api_version=AZURE_OPENAI_API_VERSION,
                deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
                openai_api_key=AZURE_OPENAI_API_KEY,
                temperature=TEMPERATURE,
                model_name=AZURE_OPENAI_DEPLOYMENT_NAME,
                max_tokens=MAX_TOKENS,
                callback_manager=callback_manager,
            )

            # Create a new QA chain for this request
            request_qa_chain = qa_module.setup_qa_chain(qa_module.vectorstore)
            request_qa_chain.combine_documents_chain.llm_chain.llm = request_llm
            
            
            result = qa_module.qa_chain.invoke(
                {"query": user_query, "user_email": user_email},
                config={"callbacks": callback_manager}
            )
            
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            sources = []
            for doc in source_docs:
                clause = doc.metadata.get('clause', 'N/A')
                if clause != 'N/A':
                    sources.append(f"{doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page_number', 'Unknown')} (Clause: {clause})")
                else:
                    sources.append(f"{doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page_number', 'Unknown')}")
            answer = markdown.markdown(answer)
        
        if request.is_json:
            return jsonify({"answer": answer, "sources": sources})
        else:
            return render_template("index.html", query=user_query, answer=answer, sources=sources)
    return render_template("index.html", query="", answer="", sources=None)
