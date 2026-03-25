import os
import json
from typing import List, Dict, Tuple, Optional


# This file contains the offline knowledge base implementation for SEEK

class SEEKKnowledgeBase:
    """Offline knowledge base for SEEK system."""

    def __init__(self, knowledge_base_path: str = "seek_knowledge_base.json"):
        """
        Initialize the offline knowledge base.

        Args:
            knowledge_base_path: Path to the knowledge base JSON file.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()
        self.index = self._build_index()

    def _load_knowledge_base(self) -> List[Dict[str, str]]:
        """Loads the knowledge base from disk."""
        if os.path.exists(self.knowledge_base_path):
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Knowledge base file not found at {self.knowledge_base_path}. Creating empty knowledge base.")
            return []

    def _build_index(self) -> Dict[str, List[Dict[str, str]]]:
        """Builds an index for fast lookup by scenario, region, and validity period."""
        index = {}

        # Build index by scenario
        for entry in self.knowledge_base:
            scenario = entry["scenario_tag"]
            if scenario not in index:
                index[scenario] = []
            index[scenario].append(entry)

        # Build index by region
        for entry in self.knowledge_base:
            region = entry["region"]
            if region not in index:
                index[region] = []
            index[region].append(entry)

        # Build index by validity period
        for entry in self.knowledge_base:
            validity = entry["validity_period"]
            if validity not in index:
                index[validity] = []
            index[validity].append(entry)

        return index

    def get_entries_by_scenario(self, scenario: str) -> List[Dict[str, str]]:
        """Retrieves knowledge base entries for a specific scenario."""
        return self.index.get(scenario, [])

    def get_entries_by_region(self, region: str) -> List[Dict[str, str]]:
        """Retrieves knowledge base entries for a specific region."""
        return self.index.get(region, [])

    def get_entries_by_validity_period(self, validity_period: str) -> List[Dict[str, str]]:
        """Retrieves knowledge base entries for a specific validity period."""
        return self.index.get(validity_period, [])

    def get_entries_by_metadata(self,
                                scenario: Optional[str] = None,
                                region: Optional[str] = None,
                                validity_period: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Retrieves knowledge base entries matching specific metadata.

        Args:
            scenario: Scenario to filter by.
            region: Region to filter by.
            validity_period: Validity period to filter by.

        Returns:
            List of matching knowledge base entries.
        """
        results = set(range(len(self.knowledge_base)))

        if scenario:
            scenario_entries = set(
                [i for i, entry in enumerate(self.knowledge_base) if entry["scenario_tag"] == scenario])
            results = results & scenario_entries

        if region:
            region_entries = set([i for i, entry in enumerate(self.knowledge_base) if entry["region"] == region])
            results = results & region_entries

        if validity_period:
            validity_entries = set(
                [i for i, entry in enumerate(self.knowledge_base) if entry["validity_period"] == validity_period])
            results = results & validity_entries

        return [self.knowledge_base[i] for i in results]

    def save(self, path: str = None):
        """Saves the knowledge base to disk."""
        path = path or self.knowledge_base_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        print(f"Knowledge base saved to {path}")


def test_knowledge_base():
    """Test function for SEEKKnowledgeBase."""
    # Create a sample knowledge base
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

    # Save to temporary file for testing
    test_kb_path = "test_seek_knowledge_base.json"
    with open(test_kb_path, 'w', encoding='utf-8') as f:
        json.dump(sample_kb, f, indent=2, ensure_ascii=False)

    # Test loading and querying
    kb = SEEKKnowledgeBase(test_kb_path)

    print("Testing SEEK Knowledge Base:")
    print("-" * 50)

    # Test by scenario
    tou_entries = kb.get_entries_by_scenario("TOU")
    print(f"TOU entries: {len(tou_entries)}")
    for i, entry in enumerate(tou_entries, 1):
        print(f"  Entry {i}: {entry['document_title']} ({entry['article_id']})")

    # Test by region
    beijing_entries = kb.get_entries_by_region("Beijing")
    print(f"\nBeijing entries: {len(beijing_entries)}")
    for i, entry in enumerate(beijing_entries, 1):
        print(f"  Entry {i}: {entry['document_title']} ({entry['article_id']})")

    # Test by validity period
    current_entries = kb.get_entries_by_validity_period("2024-01-01 to 2024-12-31")
    print(f"\nCurrent validity period entries: {len(current_entries)}")
    for i, entry in enumerate(current_entries, 1):
        print(f"  Entry {i}: {entry['document_title']} ({entry['article_id']})")

    # Test by metadata
    der_entries = kb.get_entries_by_metadata(scenario="DER", region="national")
    print(f"\nDER entries in national region: {len(der_entries)}")
    for i, entry in enumerate(der_entries, 1):
        print(f"  Entry {i}: {entry['document_title']} ({entry['article_id']})")

    # Clean up
    os.remove(test_kb_path)
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_knowledge_base()