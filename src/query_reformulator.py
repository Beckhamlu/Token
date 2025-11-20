"""
query_reformulator.py

Reformulates follow-up queries by incorporating conversation context.
"""

import re
from typing import List, Optional, Tuple
from src.conversation_memory import ConversationMemory, ConversationTurn
from src.followup_detector import FollowUpDetector, FollowUpAnalysis


class QueryReformulator:
    """
    Reformulates follow-up queries into standalone queries using conversation context.
    
    Strategies:
    1. Replace pronouns with entities from previous turn
    2. Add context from previous question
    3. Expand comparative queries
    4. Add topic context for short queries
    """
    
    def __init__(self, use_llm: bool = False, model_path: Optional[str] = None):
        """
        Initialize query reformulator.
        
        Args:
            use_llm: Whether to use LLM for reformulation (more accurate but slower)
            model_path: Path to LLM model if use_llm=True
        """
        self.use_llm = use_llm
        self.model_path = model_path
        self.detector = FollowUpDetector()
    
    def reformulate(
        self,
        query: str,
        memory: ConversationMemory,
        analysis: Optional[FollowUpAnalysis] = None
    ) -> Tuple[str, str]:
        """
        Reformulate query if it's a follow-up.
        
        Args:
            query: Current query
            memory: Conversation memory
            analysis: Pre-computed follow-up analysis (optional)
            
        Returns:
            Tuple of (reformulated_query, reformulation_method)
            If not a follow-up, returns (original_query, "none")
        """
        # Analyze if not provided
        if analysis is None:
            analysis = self.detector.analyze(
                query,
                has_history=not memory.is_empty(),
                previous_query=memory.get_last_turn().query if memory.get_last_turn() else None
            )
        
        # If not a follow-up, return original
        if not analysis.is_followup or analysis.confidence < 0.5:
            return query, "none"
        
        # Get conversation context
        last_turn = memory.get_last_turn()
        if not last_turn:
            return query, "none"
        
        # Choose reformulation strategy based on signals
        reformulated = query
        method = "none"
        
        # Strategy 1: Pronoun replacement
        if any("pronouns" in s for s in analysis.signals):
            reformulated, success = self._replace_pronouns(query, last_turn)
            if success:
                method = "pronoun_replacement"
                return reformulated, method
        
        # Strategy 2: Comparative expansion
        if any("comparative" in s for s in analysis.signals):
            reformulated, success = self._expand_comparative(query, last_turn)
            if success:
                method = "comparative_expansion"
                return reformulated, method
        
        # Strategy 3: Topic augmentation (for short/vague queries)
        if any("short_query" in s or "references" in s for s in analysis.signals):
            reformulated, success = self._augment_with_topic(query, last_turn)
            if success:
                method = "topic_augmentation"
                return reformulated, method
        
        # Strategy 4: Context prepending (fallback)
        if analysis.confidence > 0.6:
            reformulated = self._prepend_context(query, last_turn)
            method = "context_prepending"
            return reformulated, method
        
        # If all strategies fail, try LLM if available
        if self.use_llm and self.model_path:
            reformulated = self._llm_reformulate(query, memory)
            method = "llm_reformulation"
            return reformulated, method
        
        return query, "none"
    
    def _replace_pronouns(self, query: str, last_turn: ConversationTurn) -> Tuple[str, bool]:
        """
        Replace pronouns with entities from previous turn.
        
        Args:
            query: Current query with pronouns
            last_turn: Previous conversation turn
            
        Returns:
            Tuple of (reformulated_query, success)
        """
        # Extract main topic/entity from previous query
        prev_query = last_turn.query
        entity = self._extract_main_entity(prev_query)
        
        if not entity:
            return query, False
        
        # Replace pronouns
        reformulated = query
        pronoun_map = {
            r'\bit\b': entity,
            r'\bthis\b': entity,
            r'\bthat\b': entity,
            r'\bthey\b': entity,
            r'\bthem\b': entity,
        }
        
        replaced = False
        for pronoun_pattern, replacement in pronoun_map.items():
            if re.search(pronoun_pattern, reformulated, re.IGNORECASE):
                reformulated = re.sub(pronoun_pattern, replacement, reformulated, flags=re.IGNORECASE)
                replaced = True
        
        return reformulated, replaced
    
    def _expand_comparative(self, query: str, last_turn: ConversationTurn) -> Tuple[str, bool]:
        """
        Expand comparative queries with context.
        
        Example:
        Prev: "What is a B+ tree?"
        Curr: "What about hash indexes?"
        Result: "What about hash indexes compared to B+ trees?"
        
        Args:
            query: Current comparative query
            last_turn: Previous conversation turn
            
        Returns:
            Tuple of (reformulated_query, success)
        """
        entity = self._extract_main_entity(last_turn.query)
        if not entity:
            return query, False
        
        # Detect "what about X?" pattern
        match = re.search(r'what about (.+?)[\?\.]*$', query, re.IGNORECASE)
        if match:
            new_entity = match.group(1).strip()
            reformulated = f"How do {new_entity} compare to {entity}?"
            return reformulated, True
        
        # Detect "how about X?" pattern
        match = re.search(r'how about (.+?)[\?\.]*$', query, re.IGNORECASE)
        if match:
            new_entity = match.group(1).strip()
            reformulated = f"What are the differences between {new_entity} and {entity}?"
            return reformulated, True
        
        # If already has "compared to" but incomplete
        if re.search(r'\bcompared to\b', query, re.IGNORECASE) and entity not in query.lower():
            reformulated = query.rstrip('?. ') + f" and {entity}?"
            return reformulated, True
        
        return query, False
    
    def _augment_with_topic(self, query: str, last_turn: ConversationTurn) -> Tuple[str, bool]:
        """
        Augment query with topic from previous turn.
        
        Args:
            query: Current query
            last_turn: Previous conversation turn
            
        Returns:
            Tuple of (reformulated_query, success)
        """
        entity = self._extract_main_entity(last_turn.query)
        if not entity:
            return query, False
        
        # Check if query is very short or vague
        if len(query.split()) <= 5:
            # Prepend topic context
            reformulated = f"{query} (in the context of {entity})"
            return reformulated, True
        
        return query, False
    
    def _prepend_context(self, query: str, last_turn: ConversationTurn) -> str:
        """
        Prepend previous context to query (fallback strategy).
        
        Args:
            query: Current query
            last_turn: Previous conversation turn
            
        Returns:
            Reformulated query with context
        """
        prev_query = last_turn.query
        return f"Following up on '{prev_query}': {query}"
    
    def _extract_main_entity(self, query: str) -> Optional[str]:
        """
        Extract main entity/topic from query.
        
        Simple heuristic: Look for capitalized terms or key technical terms.
        
        Args:
            query: Query to extract entity from
            
        Returns:
            Main entity or None
        """
        # Remove question words and punctuation
        cleaned = re.sub(r'^(what|how|why|when|where|explain|describe|define)\s+(is|are|do|does|the)?\s*', '', query, flags=re.IGNORECASE)
        cleaned = re.sub(r'[?.!]', '', cleaned)
        
        # Look for capitalized phrases (e.g., "B+ tree", "ACID properties")
        capitalized = re.findall(r'\b[A-Z][A-Za-z0-9+\-\s]*[A-Za-z0-9+\-]', cleaned)
        if capitalized:
            return capitalized[0].strip()
        
        # Look for multi-word technical terms
        words = cleaned.split()
        if len(words) >= 2:
            significant = [w for w in words[:3] if len(w) > 3]
            if len(significant) >= 2:
                return ' '.join(significant[:2])
            elif significant:
                return significant[0]
        
        for word in words:
            if len(word) > 4:
                return word
        
        return None
    
    def _llm_reformulate(self, query: str, memory: ConversationMemory) -> str:
        """
        Use LLM to reformulate query with conversation context.
        
        Args:
            query: Current query
            memory: Conversation memory
            
        Returns:
            Reformulated query
        """
        # This would use the LLM to reformulate
        # For now, return a placeholder
        # TODO: Implement LLM-based reformulation
        return query


# Example usage
if __name__ == "__main__":
    from conversation_memory import ConversationMemory
    
    # Create memory
    memory = ConversationMemory(max_turns=5)
    
    # Add first turn
    memory.add_turn(
        query="What are ACID properties?",
        answer="ACID stands for Atomicity, Consistency, Isolation, and Durability...",
        chunks_used=["chunk1", "chunk2"],
        chunk_ids=[0, 1]
    )
    
    # Create reformulator
    reformulator = QueryReformulator()
    
    # Test follow-up queries
    test_queries = [
        "What about BASE properties?",
        "How does it ensure consistency?",
        "Explain that in more detail",
        "Why is this important?",
        "Compared to eventual consistency, what are the trade-offs?"
    ]
    
    print("Testing Query Reformulator:\n")
    for query in test_queries:
        analysis = reformulator.detector.analyze(
            query,
            has_history=True,
            previous_query=memory.get_last_turn().query
        )
        
        reformulated, method = reformulator.reformulate(query, memory, analysis)
        
        print(f"Original:     {query}")
        print(f"Follow-up:    {analysis.is_followup} (confidence: {analysis.confidence:.2f})")
        print(f"Reformulated: {reformulated}")
        print(f"Method:       {method}")
        print()
