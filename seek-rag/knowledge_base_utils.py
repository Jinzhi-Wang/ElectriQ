import os
import json
from typing import Dict, List, Tuple, Optional


# Utility functions for knowledge base operations

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extracts top N keywords from text using a simple heuristic.

    Args:
        text: Text to extract keywords from.
        top_n: Number of top keywords to return.

    Returns:
        List of top keywords.
    """
    # Simple keyword extraction based on word frequency
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}

    for word in words:
        if len(word) > 3:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and get top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_words[:top_n]]

    return keywords


def calculate_similarity_score(query: str, document: str) -> float:
    """
    Calculates a similarity score between a query and a document.

    Args:
        query: User query.
        document: Knowledge base document content.

    Returns:
        Similarity score between 0 and 1.
    """
    # Simple similarity based on keyword overlap
    query_keywords = extract_keywords(query)
    document_keywords = extract_keywords(document)

    common_keywords = set(query_keywords) & set(document_keywords)
    similarity_score = len(common_keywords) / max(len(query_keywords), 1)

    return similarity_score


def filter_by_validity(knowledge_base: List[Dict[str, str]], validity_period: str) -> List[Dict[str, str]]:
    """
    Filters knowledge base entries by validity period.

    Args:
        knowledge_base: Knowledge base entries.
        validity_period: Validity period to filter by.

    Returns:
        Filtered knowledge base entries.
    """
    return [entry for entry in knowledge_base if entry["validity_period"] == validity_period]


def filter_by_region(knowledge_base: List[Dict[str, str]], region: str) -> List[Dict[str, str]]:
    """
    Filters knowledge base entries by region.

    Args:
        knowledge_base: Knowledge base entries.
        region: Region to filter by.

    Returns:
        Filtered knowledge base entries.
    """
    return [entry for entry in knowledge_base if entry["region"] == region]


def filter_by_scenario(knowledge_base: List[Dict[str, str]], scenario: str) -> List[Dict[str, str]]:
    """
    Filters knowledge base entries by scenario.

    Args:
        knowledge_base: Knowledge base entries.
        scenario: Scenario to filter by.

    Returns:
        Filtered knowledge base entries.
    """
    return [entry for entry in knowledge_base if entry["scenario_tag"] == scenario]


def get_top_k_relevant(knowledge_base: List[Dict[str, str]], query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Gets the top K most relevant knowledge base entries for a query.

    Args:
        knowledge_base: Knowledge base entries.
        query: User query.
        top_k: Number of top entries to return.

    Returns:
        List of top K relevant knowledge base entries.
    """
    # Calculate similarity scores
    scores = []
    for entry in knowledge_base:
        score = calculate_similarity_score(query, entry["content"])
        scores.append((score, entry))

    # Sort by score and return top K
    scores.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scores[:top_k]]


def get_relevant_entries(knowledge_base: List[Dict[str, str]],
                         query: str,
                         scenario: str,
                         region: str,
                         validity_period: str,
                         top_k: int = 3) -> List[Dict[str, str]]:
    """
    Gets relevant knowledge base entries based on query and metadata.

    Args:
        knowledge_base: Knowledge base entries.
        query: User query.
        scenario: Predicted scenario.
        region: Region of interest.
        validity_period: Validity period.
        top_k: Number of top entries to return.

    Returns:
        List of relevant knowledge base entries.
    """
    # Filter by metadata
    filtered = [entry for entry in knowledge_base
                if (scenario is None or entry["scenario_tag"] == scenario) and
                (region is None or entry["region"] == region) and
                (validity_period is None or entry["validity_period"] == validity_period)]

    # Get top K relevant
    return get_top_k_relevant(filtered, query, top_k)


def test_knowledge_base_utils():
    """Test function for knowledge base utility functions."""
    # Create sample knowledge base
    sample_kb = [
        {
            "document_title": "Tariff Implementation Rules",
            "version": "2024",
            "article_id": "12",
            "region": "national",
            "validity_period": "2024-01-01 to 2024-12-31",
            "scenario_tag": "TOU",
            "content": "Time-of-use (TOU) tariffs are implemented to encourage electricity consumption during off-peak hours. The off-peak period is from 23:00 to 7:00 daily, and customers can save approximately 18% on their electricity bills by using power during this time.",
            "id": "Tariff_Implementation_Rules_2024_12_national"
        },
        {
            "document_title": "NEA Distributed PV Interconnection Regulation",
            "version": "2023",
            "article_id": "6-9",
            "region": "national",
            "validity_period": "2023-06-01 to 2024-05-31",
            "scenario_tag": "DER",
            "content": "For rooftop PV systems up to 5 kW, the application process requires: 1) Application form, 2) Device certificate, 3) Wiring diagram. The review and inspection process typically takes about 10 working days.",
            "id": "NEA_Distributed_PV_Interconnection_Regulation_2023_6-9_national"
        },
        {
            "document_title": "Power Outage Management Guidelines",
            "version": "2024",
            "article_id": "5-8",
            "region": "Beijing",
            "validity_period": "2024-01-01 to 2024-12-31",
            "scenario_tag": "Outage",
            "content": "During planned outages in Beijing, customers should: 1) Prepare flashlights and spare batteries, 2) Unplug sensitive devices, 3) Monitor outage notifications via the City Power app, 4) Contact the service hotline if the outage exceeds the announced duration.",
            "id": "Power_Outage_Management_Guidelines_2024_5-8_Beijing"
        }
    ]

    # Test keyword extraction
    print("Testing keyword extraction:")
    print("-" * 50)
    query = "What are the off-peak hours for time-of-use tariffs?"
    keywords = extract_keywords(query)
    print(f"Keywords for '{query}': {keywords}")

    # Test similarity score
    print("\nTesting similarity score:")
    document = sample_kb[0]["content"]
    similarity = calculate_similarity_score(query, document)
    print(f"Similarity between query and document: {similarity:.2f}")

    # Test filtering
    print("\nTesting filtering:")
    filtered_by_scenario = filter_by_scenario(sample_kb, "TOU")
    print(f"Filtered by TOU scenario: {len(filtered_by_scenario)} entries")

    filtered_by_region = filter_by_region(sample_kb, "Beijing")
    print(f"Filtered by Beijing region: {len(filtered_by_region)} entries")

    # Test getting top K relevant
    print("\nTesting top K relevant:")
    relevant_entries = get_top_k_relevant(sample_kb, query, top_k=2)
    for i, entry in enumerate(relevant_entries, 1):
        print(f"Top {i}: {entry['document_title']} ({entry['article_id']})")

    # Test getting relevant entries with metadata
    print("\nTesting relevant entries with metadata:")
    relevant_entries = get_relevant_entries(
        sample_kb,
        query,
        scenario="TOU",
        region="national",
        validity_period="2024-01-01 to 2024-12-31",
        top_k=2
    )
    for i, entry in enumerate(relevant_entries, 1):
        print(f"Relevant entry {i}: {entry['document_title']} ({entry['article_id']})")


if __name__ == "__main__":
    test_knowledge_base_utils()