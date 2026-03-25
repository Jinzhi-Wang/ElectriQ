import os
import json
import re
from typing import Dict, List, Tuple, Any, Optional


class KnowledgeUnit:
    """Represents a clause-level knowledge unit with structured metadata."""

    def __init__(self,
                 document_title: str,
                 version: str,
                 article_id: str,
                 region: str,
                 validity_period: str,
                 scenario_tag: str,
                 content: str,
                 id: Optional[str] = None):
        self.document_title = document_title
        self.version = version
        self.article_id = article_id
        self.region = region
        self.validity_period = validity_period
        self.scenario_tag = scenario_tag
        self.content = content
        self.id = id or f"{document_title}_{version}_{article_id}_{region}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_title": self.document_title,
            "version": self.version,
            "article_id": self.article_id,
            "region": self.region,
            "validity_period": self.validity_period,
            "scenario_tag": self.scenario_tag,
            "content": self.content,
            "id": self.id
        }


class KnowledgeRepository:
    """Organizes EPM knowledge sources into clause-level repositories."""

    def __init__(self, knowledge_units: List[KnowledgeUnit] = None):
        self.knowledge_units = knowledge_units or []
        self.index = {}  # For fast lookup by scenario and metadata
        self._build_index()

    def _build_index(self):
        """Builds an index for fast lookup by scenario, region, and validity period."""
        self.index = {}
        for unit in self.knowledge_units:
            # Create key for scenario-based lookup
            scenario_key = unit.scenario_tag
            if scenario_key not in self.index:
                self.index[scenario_key] = []
            self.index[scenario_key].append(unit)

    def add_unit(self, unit: KnowledgeUnit):
        """Adds a new knowledge unit to the repository."""
        self.knowledge_units.append(unit)
        self._build_index()

    def get_units_by_scenario(self, scenario: str) -> List[KnowledgeUnit]:
        """Retrieves all knowledge units for a specific scenario."""
        return self.index.get(scenario, [])

    def get_units_by_metadata(self,
                              region: Optional[str] = None,
                              version: Optional[str] = None,
                              validity_period: Optional[str] = None) -> List[KnowledgeUnit]:
        """Retrieves knowledge units matching specific metadata."""
        results = []
        for unit in self.knowledge_units:
            if region and unit.region != region:
                continue
            if version and unit.version != version:
                continue
            if validity_period and unit.validity_period != validity_period:
                continue
            results.append(unit)
        return results

    def save_to_disk(self, path: str):
        """Saves the knowledge repository to disk as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([unit.to_dict() for unit in self.knowledge_units], f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_disk(cls, path: str) -> 'KnowledgeRepository':
        """Loads a knowledge repository from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        units = [KnowledgeUnit(**item) for item in data]
        return cls(units)


def normalize_document(doc: Dict[str, str]) -> Dict[str, str]:
    """
    Normalizes a document entry for knowledge repository.

    Args:
        doc: A dictionary containing document metadata and content.

    Returns:
        A normalized dictionary with standardized keys.
    """
    # Standardize keys
    normalized = {
        "document_title": doc.get("document_title", ""),
        "version": doc.get("version", "latest"),
        "article_id": doc.get("article_id", "N/A"),
        "region": doc.get("region", "national"),
        "validity_period": doc.get("validity_period", "current"),
        "scenario_tag": doc.get("scenario_tag", "general"),
        "content": doc.get("content", "")
    }
    return normalized


def load_regulatory_documents(path: str) -> List[Dict[str, str]]:
    """
    Loads regulatory documents from a directory.

    Args:
        path: Path to directory containing document files.

    Returns:
        List of document dictionaries.
    """
    documents = []
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append(doc)
    return documents


def build_knowledge_repository(documents: List[Dict[str, str]]) -> KnowledgeRepository:
    """
    Builds a knowledge repository from regulatory documents.

    Args:
        documents: List of document dictionaries.

    Returns:
        A KnowledgeRepository instance.
    """
    units = []
    for doc in documents:
        normalized = normalize_document(doc)
        # Split content into clauses if necessary
        clauses = normalize_content_to_clauses(normalized["content"])
        for i, clause in enumerate(clauses):
            unit = KnowledgeUnit(
                document_title=normalized["document_title"],
                version=normalized["version"],
                article_id=f"{normalized['article_id']}-{i + 1}",
                region=normalized["region"],
                validity_period=normalized["validity_period"],
                scenario_tag=normalized["scenario_tag"],
                content=clause
            )
            units.append(unit)

    return KnowledgeRepository(units)


def normalize_content_to_clauses(content: str) -> List[str]:
    """
    Normalizes content to a list of clauses.

    Args:
        content: Raw content string.

    Returns:
        List of normalized clauses.
    """
    # Simple clause splitting based on numbered items
    clauses = []
    current_clause = ""

    # Split by numbered items or bullet points
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a number or bullet point
        if re.match(r'^\d+[\.\)]', line) or re.match(r'^[-*]', line):
            if current_clause:
                clauses.append(current_clause)
                current_clause = ""
            current_clause = line
        else:
            if current_clause:
                current_clause += " " + line
            else:
                current_clause = line

    if current_clause:
        clauses.append(current_clause)

    return clauses