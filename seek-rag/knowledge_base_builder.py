import os
import json
import re
from typing import Dict, List, Tuple, Optional


# This file is for building the knowledge base from source regulatory documents

def normalize_document(doc: Dict[str, str]) -> Dict[str, str]:
    """
    Normalizes a document entry for knowledge base construction.

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


def build_knowledge_base_from_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Builds a knowledge base from regulatory documents.

    Args:
        documents: List of document dictionaries.

    Returns:
        List of knowledge base entries.
    """
    knowledge_base = []

    for doc in documents:
        normalized = normalize_document(doc)
        # Split content into clauses if necessary
        clauses = normalize_content_to_clauses(normalized["content"])

        for i, clause in enumerate(clauses):
            kb_entry = {
                "document_title": normalized["document_title"],
                "version": normalized["version"],
                "article_id": f"{normalized['article_id']}-{i + 1}",
                "region": normalized["region"],
                "validity_period": normalized["validity_period"],
                "scenario_tag": normalized["scenario_tag"],
                "content": clause,
                "id": f"{normalized['document_title']}_{normalized['version']}_{normalized['article_id']}-{i + 1}_{normalized['region']}"
            }
            knowledge_base.append(kb_entry)

    return knowledge_base


def load_documents_from_directory(directory_path: str) -> List[Dict[str, str]]:
    """
    Loads regulatory documents from a directory.

    Args:
        directory_path: Path to directory containing document files.

    Returns:
        List of document dictionaries.
    """
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append(doc)
    return documents


def save_knowledge_base(knowledge_base: List[Dict[str, str]], output_path: str):
    """
    Saves the knowledge base to disk as JSON.

    Args:
        knowledge_base: Knowledge base entries.
        output_path: Path to save the knowledge base.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    print(f"Knowledge base saved to {output_path}")


def build_and_save_knowledge_base(source_dir: str, output_path: str):
    """
    Builds and saves the knowledge base from source documents.

    Args:
        source_dir: Directory containing source regulatory documents.
        output_path: Path to save the knowledge base.
    """
    print(f"Loading documents from {source_dir}")
    documents = load_documents_from_directory(source_dir)
    print(f"Loaded {len(documents)} documents")

    print("Building knowledge base...")
    knowledge_base = build_knowledge_base_from_documents(documents)
    print(f"Built knowledge base with {len(knowledge_base)} entries")

    print(f"Saving knowledge base to {output_path}")
    save_knowledge_base(knowledge_base, output_path)

    print("Knowledge base construction completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build SEEK knowledge base from regulatory documents')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Directory containing source regulatory documents (JSON files)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the knowledge base JSON file')

    args = parser.parse_args()

    build_and_save_knowledge_base(args.source_dir, args.output_path)