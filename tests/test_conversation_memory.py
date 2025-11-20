"""
test_conversation_memory.py

Test suite for evaluating conversation memory and follow-up handling.
Place in tests/ directory.
"""

import pytest
from src.conversation_memory import ConversationMemory, ConversationTurn
from src.followup_detector import FollowUpDetector, FollowUpAnalysis
from src.query_reformulator import QueryReformulator


class TestConversationMemory:
    """Unit tests for ConversationMemory."""
    
    def test_add_turn(self):
        memory = ConversationMemory(max_turns=3)
        
        memory.add_turn(
            query="What is ACID?",
            answer="ACID stands for...",
            chunks_used=["chunk1"],
            chunk_ids=[0]
        )
        
        assert len(memory) == 1
        assert memory.get_last_turn().query == "What is ACID?"
    
    def test_sliding_window(self):
        memory = ConversationMemory(max_turns=3)
        
        # Add 5 turns
        for i in range(5):
            memory.add_turn(
                query=f"Question {i}",
                answer=f"Answer {i}"
            )
        
        # Should only keep last 3
        assert len(memory) == 3
        recent = memory.get_recent_turns(3)
        assert recent[0].query == "Question 2"
        assert recent[-1].query == "Question 4"
    
    def test_clear(self):
        memory = ConversationMemory(max_turns=5)
        memory.add_turn("Q", "A")
        memory.clear()
        assert memory.is_empty()


class TestFollowUpDetector:
    """Unit tests for FollowUpDetector."""
    
    @pytest.fixture
    def detector(self):
        return FollowUpDetector()
    
    def test_pronoun_detection(self, detector):
        analysis = detector.analyze("How does it work?", has_history=True)
        assert analysis.is_followup
        assert "pronouns" in str(analysis.signals)
    
    def test_comparative_detection(self, detector):
        analysis = detector.analyze("What about hash indexes?", has_history=True)
        assert analysis.is_followup
        assert "comparative" in str(analysis.signals)
    
    def test_standalone_query(self, detector):
        analysis = detector.analyze("What are ACID properties?", has_history=True)
        assert not analysis.is_followup
    
    def test_no_history(self, detector):
        analysis = detector.analyze("What about this?", has_history=False)
        assert not analysis.is_followup
    
    def test_confidence_scoring(self, detector):
        # Strong follow-up signal
        analysis1 = detector.analyze("What about it?", has_history=True)
        
        # Weak follow-up signal (just short)
        analysis2 = detector.analyze("Why?", has_history=True)
        
        assert analysis1.confidence > analysis2.confidence


class TestQueryReformulator:
    """Unit tests for QueryReformulator."""
    
    @pytest.fixture
    def memory(self):
        mem = ConversationMemory(max_turns=5)
        mem.add_turn(
            query="What is a B+ tree?",
            answer="A B+ tree is a balanced tree data structure...",
            chunks_used=[],
            chunk_ids=[]
        )
        return mem
    
    @pytest.fixture
    def reformulator(self):
        return QueryReformulator()
    
    def test_pronoun_replacement(self, reformulator, memory):
        reformulated, method = reformulator.reformulate(
            "How does it work?",
            memory
        )
        
        assert "B+ tree" in reformulated or "B+tree" in reformulated
        assert method == "pronoun_replacement"
    
    def test_comparative_expansion(self, reformulator, memory):
        reformulated, method = reformulator.reformulate(
            "What about hash indexes?",
            memory
        )
        
        assert "hash" in reformulated.lower()
        assert "b+" in reformulated.lower() or "tree" in reformulated.lower()
        assert method == "comparative_expansion"
    
    def test_standalone_unchanged(self, reformulator, memory):
        original = "Explain database normalization"
        reformulated, method = reformulator.reformulate(original, memory)
        
        assert reformulated == original
        assert method == "none"


# ============ INTEGRATION TESTS ============

@pytest.mark.integration
class TestConversationFlow:
    """Integration tests for complete conversation flows."""
    
    def test_multi_turn_conversation(self):
        """Test a realistic multi-turn conversation."""
        memory = ConversationMemory(max_turns=5)
        detector = FollowUpDetector()
        reformulator = QueryReformulator()
        
        # Turn 1: Standalone query
        q1 = "What are ACID properties?"
        analysis1 = detector.analyze(q1, has_history=memory.is_empty())
        assert not analysis1.is_followup
        
        memory.add_turn(q1, "ACID stands for Atomicity, Consistency, Isolation, Durability...")
        
        # Turn 2: Follow-up with pronoun
        q2 = "How does it ensure atomicity?"
        analysis2 = detector.analyze(q2, has_history=True, previous_query=q1)
        assert analysis2.is_followup
        
        reformulated2, _ = reformulator.reformulate(q2, memory, analysis2)
        assert "ACID" in reformulated2 or "properties" in reformulated2
        
        memory.add_turn(q2, "Atomicity is ensured through transaction logging...")
        
        # Turn 3: Comparative follow-up
        q3 = "What about BASE properties?"
        analysis3 = detector.analyze(q3, has_history=True, previous_query=q2)
        assert analysis3.is_followup
        
        reformulated3, _ = reformulator.reformulate(q3, memory, analysis3)
        assert "BASE" in reformulated3
        # Should mention comparison to ACID
        assert "ACID" in reformulated3 or "compare" in reformulated3.lower()


# ============ BENCHMARK TESTS FOR EVALUATION ============

"""
conversation_benchmarks.yaml

Add this to tests/benchmarks.yaml to evaluate conversation memory:
"""

CONVERSATION_BENCHMARKS = """
conversation_tests:
  - id: "conv_acid_followup"
    conversation:
      - turn: 1
        query: "What are ACID properties?"
        expected_answer: "ACID stands for Atomicity, Consistency, Isolation, and Durability. These are properties that guarantee database transactions are processed reliably."
        
      - turn: 2
        query: "How does it ensure atomicity?"
        is_followup: true
        expected_reformulation: "How do ACID properties ensure atomicity?"
        expected_answer: "Atomicity is ensured through transaction logging and rollback mechanisms..."
        
      - turn: 3
        query: "What about consistency?"
        is_followup: true
        expected_reformulation: "What about consistency in ACID properties?"
        expected_answer: "Consistency ensures that transactions bring the database from one valid state to another..."
    
    evaluation:
      - followup_detection_accuracy: "Did system correctly detect turns 2 and 3 as follow-ups?"
      - reformulation_quality: "Did reformulated queries capture context properly?"
      - answer_relevance: "Were answers relevant to reformulated queries?"

  - id: "conv_index_comparison"
    conversation:
      - turn: 1
        query: "What is a B+ tree index?"
        expected_answer: "A B+ tree is a balanced tree data structure used for indexing..."
        
      - turn: 2
        query: "What about hash indexes?"
        is_followup: true
        expected_reformulation: "How do hash indexes compare to B+ trees?"
        expected_answer: "Hash indexes use a hash function for direct lookups..."
        
      - turn: 3
        query: "Which is faster for range queries?"
        is_followup: true
        expected_reformulation: "Which is faster for range queries: hash indexes or B+ trees?"
        expected_answer: "B+ trees are faster for range queries because they maintain sorted order..."

  - id: "conv_transaction_isolation"
    conversation:
      - turn: 1
        query: "What are isolation levels in databases?"
        expected_answer: "Isolation levels control the visibility of changes made by concurrent transactions..."
        
      - turn: 2
        query: "Explain read uncommitted"
        is_followup: false  # This is standalone - no pronoun or comparison
        expected_answer: "Read uncommitted is the lowest isolation level..."
        
      - turn: 3
        query: "How does it differ from serializable?"
        is_followup: true
        expected_reformulation: "How does read uncommitted differ from serializable isolation?"
        expected_answer: "Read uncommitted allows dirty reads while serializable prevents all anomalies..."
"""


# ============ EVALUATION SCRIPT ============

def evaluate_conversation_memory():
    """
    Script to evaluate conversation memory performance.
    
    Metrics:
    1. Follow-up detection accuracy
    2. Reformulation quality
    3. Answer improvement with context
    """
    
    print("="*60)
    print("CONVERSATION MEMORY EVALUATION")
    print("="*60)
    
    # Test cases: (query, has_history, expected_is_followup)
    test_cases = [
        ("What are ACID properties?", False, False),
        ("How does it ensure atomicity?", True, True),
        ("What about BASE properties?", True, True),
        ("Explain database normalization", True, False),
        ("Why is this important?", True, True),
    ]
    
    detector = FollowUpDetector()
    correct = 0
    total = len(test_cases)
    
    print("\n1. Follow-up Detection Accuracy:")
    print("-" * 60)
    
    for query, has_history, expected in test_cases:
        analysis = detector.analyze(query, has_history=has_history)
        is_correct = analysis.is_followup == expected
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {query}")
        print(f"   Expected: {expected}, Got: {analysis.is_followup} (conf: {analysis.confidence:.2f})")
    
    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")
    
    # Reformulation quality
    print("\n2. Reformulation Quality:")
    print("-" * 60)
    
    memory = ConversationMemory(max_turns=5)
    memory.add_turn(
        "What are ACID properties?",
        "ACID stands for Atomicity, Consistency, Isolation, Durability..."
    )
    
    reformulator = QueryReformulator()
    
    followup_queries = [
        "How does it ensure atomicity?",
        "What about BASE properties?",
        "Why is this important?"
    ]
    
    for query in followup_queries:
        reformulated, method = reformulator.reformulate(query, memory)
        print(f"Original:     {query}")
        print(f"Reformulated: {reformulated}")
        print(f"Method:       {method}\n")
    
    print("="*60)
    print("Evaluation complete!")
    print("\nTo run full benchmark suite:")
    print("  pytest tests/test_conversation_memory.py -v")


if __name__ == "__main__":
    # Run evaluation
    evaluate_conversation_memory()
