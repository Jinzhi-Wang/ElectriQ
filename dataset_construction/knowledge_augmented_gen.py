"""
Knowledge-Augmented Generation Module for ElectriQ
Generates synthetic dialogue data augmented with domain knowledge.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class DialogueType(Enum):
    """Types of dialogue scenarios."""
    TARIFF_INQUIRY = "tariff_inquiry"
    BILL_DISPUTE = "bill_dispute"
    PLAN_APPLICATION = "plan_application"
    OUTAGE_REPORT = "outage_report"
    METER_READING = "meter_reading"
    DER_INTERCONNECTION = "der_interconnection"
    ENERGY_CONSULTATION = "energy_consultation"
    COMPLAINT = "complaint"


@dataclass
class KnowledgeSnippet:
    """Represents a knowledge base snippet."""
    topic: str
    content: str
    source: str
    validity_period: Optional[str] = None
    region: Optional[str] = None


@dataclass
class GeneratedDialogue:
    """Represents a generated dialogue."""
    dialogue_id: str
    dialogue_type: DialogueType
    turns: List[Dict]
    knowledge_used: List[KnowledgeSnippet]
    metadata: Dict


class KnowledgeBase:
    """
    Domain knowledge base for dialogue generation.
    """

    def __init__(self, knowledge_path: Optional[str] = None):
        """
        Initialize knowledge base.

        Args:
            knowledge_path: Path to knowledge base JSON file
        """
        self.knowledge: Dict[str, List[KnowledgeSnippet]] = {}

        if knowledge_path:
            self.load_knowledge(knowledge_path)
        else:
            self._init_default_knowledge()

    def _init_default_knowledge(self):
        """Initialize default domain knowledge."""
        self.knowledge = {
            'tariff': [
                KnowledgeSnippet(
                    topic='tariff',
                    content='居民峰谷电价时段：峰段 8:00-23:00，谷段 23:00-次日 7:00',
                    source='电价目录 2024 年第 12 条',
                    region='city',
                    validity_period='2024-01-01 to 2024-12-31'
                ),
                KnowledgeSnippet(
                    topic='tariff',
                    content='峰谷电价套餐 A：谷段电价 0.3 元/度，峰段电价 0.6 元/度',
                    source='电价目录 2024 年第 15 条',
                    region='city',
                    validity_period='2024-01-01 to 2024-12-31'
                ),
            ],
            'application': [
                KnowledgeSnippet(
                    topic='application',
                    content='峰谷电价申请渠道：城市电力 APP、网上营业厅、服务热线 95598',
                    source='业务办理指南',
                    region='city'
                ),
                KnowledgeSnippet(
                    topic='application',
                    content='峰谷电价申请所需材料：身份证、房产证明、电表户号',
                    source='业务办理指南'
                ),
            ],
            'billing': [
                KnowledgeSnippet(
                    topic='billing',
                    content='电费计算方式：用电量×对应时段电价',
                    source='电费计算规则'
                ),
                KnowledgeSnippet(
                    topic='billing',
                    content='电费账单异议处理：可在收到账单后 30 天内申请复核',
                    source='客户服务规范'
                ),
            ]
        }

    def load_knowledge(self, knowledge_path: str):
        """
        Load knowledge from JSON file.

        Args:
            knowledge_path: Path to knowledge JSON file
        """
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for topic, snippets in data.items():
            self.knowledge[topic] = [
                KnowledgeSnippet(**snippet) for snippet in snippets
            ]

    def get_knowledge(
            self,
            topic: str,
            region: Optional[str] = None,
            date: Optional[str] = None
    ) -> List[KnowledgeSnippet]:
        """
        Get knowledge snippets for a topic.

        Args:
            topic: Knowledge topic
            region: Region filter
            date: Date for validity check

        Returns:
            list: Relevant knowledge snippets
        """
        snippets = self.knowledge.get(topic, [])

        # Filter by region
        if region:
            snippets = [
                s for s in snippets
                if s.region is None or s.region == region
            ]

        # Filter by validity period
        if date:
            filtered = []
            for s in snippets:
                if s.validity_period is None:
                    filtered.append(s)
                else:
                    # Check if date is within validity period
                    start, end = s.validity_period.split(' to ')
                    if start <= date <= end:
                        filtered.append(s)
            snippets = filtered

        return snippets

    def get_random_knowledge(
            self,
            topic: str,
            count: int = 1
    ) -> List[KnowledgeSnippet]:
        """
        Get random knowledge snippets.

        Args:
            topic: Knowledge topic
            count: Number of snippets

        Returns:
            list: Random knowledge snippets
        """
        snippets = self.knowledge.get(topic, [])
        return random.sample(snippets, min(count, len(snippets)))


class DialogueGenerator:
    """
    Generate synthetic dialogues using LLM.
    """

    def __init__(
            self,
            llm_model: str = "gpt-4o",
            llm_client: Optional[object] = None
    ):
        """
        Initialize dialogue generator.

        Args:
            llm_model: LLM model name
            llm_client: LLM client instance
        """
        self.llm_model = llm_model
        self.llm_client = llm_client

        if llm_client is None:
            self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI()
        except ImportError:
            self.llm_client = None
            logging.warning("OpenAI client not available")

    def _build_prompt(
            self,
            dialogue_type: DialogueType,
            knowledge_snippets: List[KnowledgeSnippet],
            num_turns: int = 6
    ) -> str:
        """
        Build prompt for dialogue generation.

        Args:
            dialogue_type: Type of dialogue
            knowledge_snippets: Knowledge to incorporate
            num_turns: Number of dialogue turns

        Returns:
            str: Generation prompt
        """
        knowledge_text = "\n".join([
            f"- {s.content} (来源：{s.source})"
            for s in knowledge_snippets
        ])

        prompt = f"""You are a dialogue generator for Electric Power Marketing (EPM) customer service.

Generate a realistic {num_turns}-turn dialogue between a customer and a service representative.

Dialogue Type: {dialogue_type.value}

Domain Knowledge to Incorporate:
{knowledge_text}

Requirements:
1. Natural conversation flow
2. Customer asks questions or expresses concerns
3. Representative provides accurate, helpful responses using the knowledge
4. Include appropriate greetings and closings
5. Use professional but friendly tone
6. Responses should be actionable and complete

Output format (JSON):
{{
    "turns": [
        {{"speaker": "customer", "text": "..."}},
        {{"speaker": "representative", "text": "..."}},
        ...
    ]
}}
"""
        return prompt

    def generate_dialogue(
            self,
            dialogue_type: DialogueType,
            knowledge_snippets: List[KnowledgeSnippet],
            num_turns: int = 6
    ) -> Optional[GeneratedDialogue]:
        """
        Generate a single dialogue.

        Args:
            dialogue_type: Type of dialogue
            knowledge_snippets: Knowledge to incorporate
            num_turns: Number of turns

        Returns:
            GeneratedDialogue: Generated dialogue or None
        """
        if self.llm_client is None:
            return self._generate_rule_based(dialogue_type, knowledge_snippets, num_turns)

        prompt = self._build_prompt(dialogue_type, knowledge_snippets, num_turns)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # Parse JSON
            import json
            dialogue_data = json.loads(content)

            # Create dialogue object
            dialogue_id = f"gen_{dialogue_type.value}_{random.randint(1000, 9999)}"

            dialogue = GeneratedDialogue(
                dialogue_id=dialogue_id,
                dialogue_type=dialogue_type,
                turns=dialogue_data.get('turns', []),
                knowledge_used=knowledge_snippets,
                metadata={
                    'model': self.llm_model,
                    'num_turns': num_turns,
                    'knowledge_count': len(knowledge_snippets)
                }
            )

            return dialogue

        except Exception as e:
            logging.error(f"Failed to generate dialogue: {e}")
            return self._generate_rule_based(dialogue_type, knowledge_snippets, num_turns)

    def _generate_rule_based(
            self,
            dialogue_type: DialogueType,
            knowledge_snippets: List[KnowledgeSnippet],
            num_turns: int = 6
    ) -> GeneratedDialogue:
        """
        Generate dialogue using rule-based templates (fallback).

        Args:
            dialogue_type: Type of dialogue
            knowledge_snippets: Knowledge to incorporate
            num_turns: Number of turns

        Returns:
            GeneratedDialogue: Generated dialogue
        """
        # Simple template-based generation
        templates = {
            DialogueType.TARIFF_INQUIRY: {
                'customer_opening': "请问峰谷电价的时间是怎么规定的？",
                'representative_response': f"根据{knowledge_snippets[0].source}，{knowledge_snippets[0].content}",
                'customer_followup': "那怎么申请这个套餐呢？",
                'representative_closing': f"您可以通过{knowledge_snippets[1].content}进行申请"
            }
        }

        template = templates.get(dialogue_type, templates[DialogueType.TARIFF_INQUIRY])

        turns = [
            {"speaker": "customer", "text": template['customer_opening']},
            {"speaker": "representative", "text": template['representative_response']},
            {"speaker": "customer", "text": template['customer_followup']},
            {"speaker": "representative", "text": template['representative_closing']}
        ]

        dialogue_id = f"gen_{dialogue_type.value}_{random.randint(1000, 9999)}"

        return GeneratedDialogue(
            dialogue_id=dialogue_id,
            dialogue_type=dialogue_type,
            turns=turns,
            knowledge_used=knowledge_snippets,
            metadata={'generation_method': 'rule_based'}
        )


class KnowledgeAugmentedDataGenerator:
    """
    Main pipeline for knowledge-augmented dialogue generation.
    """

    def __init__(
            self,
            knowledge_base: Optional[KnowledgeBase] = None,
            dialogue_generator: Optional[DialogueGenerator] = None
    ):
        """
        Initialize data generator.

        Args:
            knowledge_base: Knowledge base instance
            dialogue_generator: Dialogue generator instance
        """
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.dialogue_generator = dialogue_generator or DialogueGenerator()

    def generate_dataset(
            self,
            dialogue_types: List[DialogueType],
            samples_per_type: int = 100,
            num_turns: int = 6,
            output_path: Optional[str] = None
    ) -> List[GeneratedDialogue]:
        """
        Generate complete dataset.

        Args:
            dialogue_types: Types of dialogues to generate
            samples_per_type: Number of samples per type
            num_turns: Number of turns per dialogue
            output_path: Path to save dataset

        Returns:
            list: List of generated dialogues
        """
        all_dialogues = []

        for dtype in dialogue_types:
            logging.info(f"Generating {samples_per_type} samples for {dtype.value}")

            for i in range(samples_per_type):
                # Get relevant knowledge
                knowledge = self.knowledge_base.get_random_knowledge(
                    topic=dtype.value.split('_')[0],
                    count=2
                )

                # Generate dialogue
                dialogue = self.dialogue_generator.generate_dialogue(
                    dialogue_type=dtype,
                    knowledge_snippets=knowledge,
                    num_turns=num_turns
                )

                if dialogue:
                    all_dialogues.append(dialogue)

        # Save to file
        if output_path:
            self._save_dataset(all_dialogues, output_path)

        return all_dialogues

    def _save_dataset(
            self,
            dialogues: List[GeneratedDialogue],
            output_path: str
    ):
        """
        Save dataset to JSON file.

        Args:
            dialogues: List of generated dialogues
            output_path: Output file path
        """
        data = []
        for dialogue in dialogues:
            dialogue_dict = {
                'dialogue_id': dialogue.dialogue_id,
                'dialogue_type': dialogue.dialogue_type.value,
                'turns': dialogue.turns,
                'knowledge_used': [
                    {
                        'topic': k.topic,
                        'content': k.content,
                        'source': k.source
                    }
                    for k in dialogue.knowledge_used
                ],
                'metadata': dialogue.metadata
            }
            data.append(dialogue_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_dataset(self, input_path: str) -> List[GeneratedDialogue]:
        """
        Load dataset from JSON file.

        Args:
            input_path: Input file path

        Returns:
            list: List of GeneratedDialogue objects
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dialogues = []
        for item in data:
            dialogue = GeneratedDialogue(
                dialogue_id=item['dialogue_id'],
                dialogue_type=DialogueType(item['dialogue_type']),
                turns=item['turns'],
                knowledge_used=[
                    KnowledgeSnippet(**k) for k in item['knowledge_used']
                ],
                metadata=item['metadata']
            )
            dialogues.append(dialogue)

        return dialogues


if __name__ == '__main__':
    # Example usage
    generator = KnowledgeAugmentedDataGenerator()

    # Generate dataset
    dialogue_types = [
        DialogueType.TARIFF_INQUIRY,
        DialogueType.PLAN_APPLICATION,
        DialogueType.BILL_DISPUTE
    ]

    dialogues = generator.generate_dataset(
        dialogue_types=dialogue_types,
        samples_per_type=5,
        num_turns=6,
        output_path="./generated_dialogues.json"
    )

    print(f"Generated {len(dialogues)} dialogues")
    print(f"Sample dialogue:")
    print(f"  ID: {dialogues[0].dialogue_id}")
    print(f"  Type: {dialogues[0].dialogue_type.value}")
    print(f"  Turns: {len(dialogues[0].turns)}")
    for turn in dialogues[0].turns[:4]:
        print(f"    [{turn['speaker']}]: {turn['text'][:50]}...")