import os
import json
from datetime import datetime
import logging
from rag_system import SEEKRAG
from knowledge_org import KnowledgeRepository, build_knowledge_repository, load_regulatory_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='seek_rag.log'
)
logger = logging.getLogger('seek_rag_example')


def load_sample_knowledge_repo():
    """Loads a sample knowledge repository for demonstration."""
    # In a real application, this would be loaded from a file
    sample_docs = [
        {
            "document_title": "Tariff Implementation Rules",
            "version": "2024",
            "article_id": "12",
            "region": "national",
            "validity_period": "current",
            "scenario_tag": "TOU",
            "content": "Time-of-use (TOU) tariffs are implemented to encourage electricity consumption during off-peak hours. The off-peak period is from 23:00 to 7:00 daily, and customers can save approximately 18% on their electricity bills by using power during this time."
        },
        {
            "document_title": "NEA Distributed PV Interconnection Regulation",
            "version": "2023",
            "article_id": "6-9",
            "region": "national",
            "validity_period": "current",
            "scenario_tag": "DER",
            "content": "For rooftop PV systems up to 5 kW, the application process requires: 1) Application form, 2) Device certificate, 3) Wiring diagram. The review and inspection process typically takes about 10 working days."
        },
        {
            "document_title": "Power Outage Management Guidelines",
            "version": "2024",
            "article_id": "5-8",
            "region": "national",
            "validity_period": "current",
            "scenario_tag": "Outage",
            "content": "During planned outages, customers should: 1) Prepare flashlights and spare batteries, 2) Unplug sensitive devices, 3) Monitor outage notifications via the City Power app, 4) Contact the service hotline if the outage exceeds the announced duration."
        }
    ]

    return build_knowledge_repository(sample_docs)


def main():
    """Main example execution."""
    logger.info("Starting SEEK-RAG example")

    # Create sample knowledge repository
    logger.info("Building sample knowledge repository")
    knowledge_repo = load_sample_knowledge_repo()

    # Save knowledge repository to disk
    repo_path = "knowledge_repo.json"
    knowledge_repo.save_to_disk(repo_path)
    logger.info(f"Knowledge repository saved to {repo_path}")

    # Initialize SEEK-RAG
    logger.info("Initializing SEEK-RAG system")
    seek_rag = SEEKRAG(
        model_name="meta-llama/Llama-3-8B-Instruct",
        knowledge_repo_path=repo_path
    )

    # Example dialogues
    examples = [
        {
            "context": "Customer: I usually charge my EV at 22:00. Is there an off-peak tariff? Can I join a demand response event?",
            "query": "What are the off-peak hours and how can I join a demand response event?"
        },
        {
            "context": "Customer: I plan to install a 5 kW rooftop PV system. How can I apply for grid connection?",
            "query": "What documents do I need for the application?"
        },
        {
            "context": "Customer: There's a planned outage this afternoon. What should I prepare in advance?",
            "query": "What should I do before the outage?"
        }
    ]

    # Process examples
    for i, example in enumerate(examples, 1):
        logger.info(f"Processing example {i}/{len(examples)}")
        logger.info(f"Context: {example['context']}")
        logger.info(f"Query: {example['query']}")

        # Generate response
        response, evidence = seek_rag.generate_response(
            context=example["context"],
            query=example["query"]
        )

        # Print results
        print(f"\nExample {i}:")
        print(f"Context: {example['context']}")
        print(f"Query: {example['query']}")
        print(f"Response: {response}")
        print(f"Retrieved evidence ({len(evidence)} units):")
        for j, unit in enumerate(evidence, 1):
            print(f"  [{j}] {unit.document_title} ({unit.version}), Clause {unit.article_id}: {unit.content[:100]}...")

        # Save response to file
        output_dir = "example_responses"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{output_dir}/response_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(f"Context: {example['context']}\n")
            f.write(f"Query: {example['query']}\n\n")
            f.write(f"Response:\n{response}\n\n")
            f.write(f"Retrieved evidence:\n")
            for j, unit in enumerate(evidence, 1):
                f.write(
                    f"  [{j}] {unit.document_title} ({unit.version}), Clause {unit.article_id}: {unit.content[:100]}...\n")

        logger.info(f"Response saved to {output_dir}/response_{timestamp}.txt")

    logger.info("Example execution completed")


if __name__ == "__main__":
    main()