import os
import uuid # Import uuid
from typing import Any, Dict, List, Optional, Sequence, Union # Import necessary types

from langsmith import Client
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run
from langchain_core.outputs import LLMResult # Import LLMResult
from langchain_core.documents import Document # Import Document

class EmailLangChainTracer(LangChainTracer):
    """
    Custom LangChainTracer that ensures user_email from invoke metadata
    is added to the run's metadata in LangSmith.
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        example_id: Optional[Union[str, uuid.UUID]] = None,
        tags: Optional[List[str]] = None,
        client: Optional[Any] = None,
        # Add any other specific args your tracer needs, but don't use **kwargs for super()
    ):
        # Ensure LangSmith environment variables are set for default tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if project_name:
            os.environ["LANGCHAIN_PROJECT"] = project_name
        else:
            # Ensure project name is set, either via arg or env var
            project_name = os.getenv("LANGCHAIN_PROJECT", "Default Project")

        # Call LangChainTracer's __init__ with only the arguments it expects
        super().__init__(
            example_id=example_id,
            tags=tags,
            project_name=project_name,
            client=client
            # DO NOT pass **kwargs here
        )
        # Initialize the client if not provided (LangChainTracer might do this too, but being explicit is safe)
        if self.client is None:
             try:
                 self.client = Client()
             except Exception as e:
                 print(f"WARN [EmailTracer]: Failed to initialize LangSmith client: {e}. Tracing might not work.")
                 self.client = None # Ensure client is None if init fails

        print(f"DEBUG [EmailTracer]: Initialized for project '{project_name}'.")


    def _get_user_email_from_metadata(self, run: Run) -> Optional[str]:
        """Helper to safely extract user_email from run metadata."""
        # Metadata might be passed down through parent runs
        current_run = run
        while current_run:
            if current_run.extra and "metadata" in current_run.extra:
                user_email = current_run.extra["metadata"].get("user_email")
                if user_email:
                    return user_email
            # Check parent run if not found in current
            if current_run.parent_run_id and current_run.parent_run_id in self.run_map:
                 current_run = self.run_map.get(current_run.parent_run_id)
            else:
                 current_run = None # Stop if no parent or parent not in map
        return None

    # Override _persist_run or _start_trace to add metadata early
    def _start_trace(self, run: Run) -> None:
        """Process the run at the start."""
        user_email = self._get_user_email_from_metadata(run)
        if user_email:
            # Ensure 'metadata' exists in run.extra
            if run.extra is None:
                run.extra = {}
            if "metadata" not in run.extra:
                run.extra["metadata"] = {}

            # Add user_email if not already present
            if "user_email" not in run.extra["metadata"]:
                print(f"DEBUG [EmailTracer]: Adding user_email '{user_email}' to metadata for run {run.id}")
                run.extra["metadata"]["user_email"] = user_email

            # Optionally add tags here too if needed, ensuring tags list exists
            if run.tags is None:
                run.tags = []
            tag = f"user:{user_email}"
            if tag not in run.tags:
                 run.tags.append(tag)

        # Call the original _start_trace to handle standard LangSmith logic
        super()._start_trace(run)


    # _persist_run is called when the run ends. Metadata might be better added at the start.
    # def _persist_run(self, run: Run) -> None:
    #     """Persist run details."""
    #     user_email = self._get_user_email_from_metadata(run)
    #     if user_email:
    #         if run.extra is None: run.extra = {}
    #         if "metadata" not in run.extra: run.extra["metadata"] = {}
    #         if "user_email" not in run.extra["metadata"]:
    #             run.extra["metadata"]["user_email"] = user_email
    #             print(f"DEBUG [EmailTracer]: Adding user_email to metadata in _persist_run for run {run.id}")
    #         if run.tags is None: run.tags = []
    #         tag = f"user:{user_email}"
    #         if tag not in run.tags: run.tags.append(tag)

    #     # Call the parent method to handle actual persistence to LangSmith
    #     super()._persist_run(run)

    # You might need to override other methods if user_email isn't propagating
    # correctly via run.extra["metadata"] from the initial invoke call.
    # For example, on_llm_start:
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Run:
        # Try to get user_email from the metadata passed down
        user_email = None
        if metadata:
            user_email = metadata.get("user_email")

        # If not found, try getting from parent run in the run_map
        if not user_email and parent_run_id and parent_run_id in self.run_map:
             parent_run = self.run_map.get(parent_run_id)
             user_email = self._get_user_email_from_metadata(parent_run) # Use helper

        # Add user_email to the current run's metadata if found
        if user_email:
            if metadata is None: metadata = {}
            if "user_email" not in metadata:
                metadata["user_email"] = user_email
                print(f"DEBUG [EmailTracer]: Adding user_email to metadata in on_llm_start for run {run_id}")
            # Optionally add tags
            if tags is None: tags = []
            tag = f"user:{user_email}"
            if tag not in tags: tags.append(tag)


        # Call the parent implementation with potentially updated tags/metadata
        return super().on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    # Add similar logic for on_chain_start, on_retriever_start etc. if needed
    # to ensure metadata propagates correctly.

    # Remove the email sending logic for now to focus on fixing the tracer init
    # def send_trace_email(self, run_details: Dict):
    #     pass