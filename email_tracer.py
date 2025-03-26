from langsmith import Client
from langchain_core.tracers import LangChainTracer
import os

class EmailLangChainTracer(LangChainTracer):
    """Custom tracer that captures and tracks user email information."""
    
    def __init__(self, project_name=None, **kwargs):
        # Ensure LangSmith environment variables are set
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if project_name:
            os.environ["LANGCHAIN_PROJECT"] = project_name
            
        super().__init__(**kwargs)
        self.client = Client()
        
    def on_chain_start(self, serialized, inputs, run_id=None, parent_run_id=None, **kwargs):
        # Initialize metadata and tags if not present
        metadata = kwargs.get("metadata", {})
        tags = kwargs.get("tags", [])
        
        # Capture user email from inputs
        user_email = inputs.get("user_email")
        if user_email:
            metadata["user_email"] = user_email
            tags.append(f"user:{user_email}")
            
            # Update kwargs
            kwargs["metadata"] = metadata
            kwargs["tags"] = tags
            
        # Call parent implementation
        super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs
        )
