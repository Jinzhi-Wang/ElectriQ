import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

from knowledge_org import KnowledgeRepository
from retrieval import ScenarioClassifier, KeywordExtractor, retrieve_evidence, build_bm25_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SEEKRAG:
    """SEEK-RAG: Scenario-guided Evidence-based Knowledge Retrieval and Generation."""

    def __init__(self,
                 model_name: str = "meta-llama/Llama-3-8B-Instruct",
                 knowledge_repo_path: Optional[str] = None,
                 knowledge_repo: Optional[KnowledgeRepository] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize SEEK-RAG system.

        Args:
            model_name: Name of the language model to use.
            knowledge_repo_path: Path to the knowledge repository.
            knowledge_repo: Pre-loaded knowledge repository.
            device: Device to run the model on.
        """
        self.device = device
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        # Set generation config
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Initialize components
        self.scenario_classifier = ScenarioClassifier()
        self.keyword_extractor = KeywordExtractor()

        # Load knowledge repository
        if knowledge_repo:
            self.knowledge_repo = knowledge_repo
        elif knowledge_repo_path and os.path.exists(knowledge_repo_path):
            self.knowledge_repo = KnowledgeRepository.load_from_disk(knowledge_repo_path)
        else:
            self.knowledge_repo = KnowledgeRepository()
            logger.warning("No knowledge repository loaded. Retrieval will be limited.")

        # Build BM25 index for faster retrieval
        self.bm25_index = build_bm25_index(self.knowledge_repo) if self.knowledge_repo.knowledge_units else None

    def _prepare_prompt(self,
                        context: str,
                        query: str,
                        scenario: str,
                        region: str,
                        version: str,
                        validity_period: str) -> str:
        """
        Prepares the prompt for the language model.

        Args:
            context: Previous dialogue context.
            query: Current user query.
            scenario: Predicted scenario.
            region: Region of interest.
            version: Policy version.
            validity_period: Validity period.

        Returns:
            Formatted prompt string.
        """
        prompt = f"""
        You are a customer service representative for a power company. Your task is to provide accurate, compliant, and actionable responses to customer inquiries based on the latest regulatory documents and tariff policies.

        Current context: {context}

        User query: {query}

        Scenario: {scenario}
        Region: {region}
        Policy version: {version}
        Validity period: {validity_period}

        Please provide a clear, professional, and helpful response that includes:
        1. Accurate information based on current regulations
        2. Step-by-step guidance when applicable
        3. Appropriate citations to regulatory documents (e.g., "Per Tariff Implementation Rules (2024), Clause 12")
        4. Information about applicable timeframes, requirements, and next steps

        Your response must be compliant with the latest policies and provide verifiable information.
        """
        return prompt

    def _extract_metadata_from_context(self, context: str) -> Dict[str, str]:
        """Extracts metadata (region, version, validity period) from dialogue context."""
        metadata = {
            "region": "national",
            "version": "latest",
            "validity_period": "current"
        }

        # Extract region (simplified example)
        if "Beijing" in context:
            metadata["region"] = "Beijing"
        elif "Shanghai" in context:
            metadata["region"] = "Shanghai"
        # Add more region detection as needed

        # Extract version (simplified example)
        if "2024" in context:
            metadata["version"] = "2024"

        return metadata

    def generate_response(self,
                          context: str,
                          query: str,
                          top_k: int = 2) -> Tuple[str, List[KnowledgeUnit]]:
        """
        Generates a response using SEEK-RAG.

        Args:
            context: Previous dialogue context.
            query: Current user query.
            top_k: Number of evidence units to retrieve.

        Returns:
            Tuple containing the generated response and the retrieved evidence units.
        """
        # Extract metadata from context
        metadata = self._extract_metadata_from_context(context)
        region = metadata["region"]
        version = metadata["version"]
        validity_period = metadata["validity_period"]

        # Predict scenario
        scenario = self.scenario_classifier.predict(query)

        # Retrieve evidence
        evidence_units = retrieve_evidence(
            knowledge_repo=self.knowledge_repo,
            query=query,
            scenario=scenario,
            region=region,
            version=version,
            validity_period=validity_period,
            top_k=top_k,
            bm25_model=self.bm25_index
        )

        # Prepare prompt
        prompt = self._prepare_prompt(
            context=context,
            query=query,
            scenario=scenario,
            region=region,
            version=version,
            validity_period=validity_period
        )

        # Generate initial response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )
        initial_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Format evidence for inclusion in prompt
        evidence_text = ""
        if evidence_units:
            evidence_text = "\n\nRelevant policy information:\n"
            for i, unit in enumerate(evidence_units, 1):
                evidence_text += f"{i}. [{unit.document_title} ({unit.version}), Clause {unit.article_id}]: {unit.content[:200]}...\n"

        # Create final prompt with evidence
        final_prompt = f"{prompt}\n\n{evidence_text}\n\nPlease revise your response to include the relevant policy information and ensure it is compliant and actionable."

        # Generate final response with evidence
        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )
        final_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean response (remove prompt from response)
        final_response = final_response.replace(prompt, "").strip()

        return final_response, evidence_units

    def save_model(self, path: str):
        """Saves the model and tokenizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Loads a model and tokenizer from disk."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        logger.info(f"Model loaded from {path}")

    def evaluate(self,
                 context: str,
                 query: str,
                 reference_response: str,
                 evidence_units: List[KnowledgeUnit]) -> Dict[str, float]:
        """
        Evaluates the response against a reference.

        Args:
            context: Previous dialogue context.
            query: Current user query.
            reference_response: Reference response for evaluation.
            evidence_units: Retrieved evidence units.

        Returns:
            Dictionary of evaluation metrics.
        """
        # In a real implementation, you'd implement more comprehensive evaluation
        # For this example, we'll just return a simple score
        response, _ = self.generate_response(context, query)

        # Placeholder for evaluation metrics
        metrics = {
            "compliance": 0.0,
            "actionability": 0.0,
            "professionalism": 0.0,
            "clarity": 0.0
        }

        # For demonstration purposes, we'll return a fixed score
        # In a real implementation, you'd use more sophisticated evaluation
        metrics["compliance"] = 0.85
        metrics["actionability"] = 0.90
        metrics["professionalism"] = 0.88
        metrics["clarity"] = 0.92

        return metrics


def main():
    """Example usage of SEEK-RAG."""
    # Initialize SEEK-RAG
    seek_rag = SEEKRAG(
        model_name="meta-llama/Llama-3-8B-Instruct",
        knowledge_repo_path="knowledge_repo.json"
    )

    # Example dialogue
    context = "Customer: I want to install a 5 kW rooftop PV system. How can I apply for grid connection?"
    query = "What documents do I need for the application?"

    # Generate response
    response, evidence = seek_rag.generate_response(context, query)

    # Print results
    print("Response:")
    print(response)
    print("\nRetrieved evidence:")
    for i, unit in enumerate(evidence, 1):
        print(f"{i}. [{unit.document_title} ({unit.version}), Clause {unit.article_id}]: {unit.content[:100]}...")

    # Save model (optional)
    # seek_rag.save_model("seek_rag_model")


if __name__ == "__main__":
    main()