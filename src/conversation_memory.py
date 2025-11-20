"""
conversation_memory.py

Tracks conversation history and provides context for follow-up queries.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import deque


@dataclass
class ConversationTurn:
    query: str
    answer: str
    chunks_used: List[str]  # Actual text of chunks
    chunk_ids: List[int]    # Global indices
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "query": self.query,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "num_chunks": len(self.chunks_used),
            "chunk_ids": self.chunk_ids,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


class ConversationMemory:
    """
    Manages conversation history with a sliding window.
    """
    
    def __init__(self, max_turns: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of turns to keep in memory
        """
        self.max_turns = max_turns
        self.history: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._topics: List[str] = []  # Extracted topics from conversation
    
    def add_turn(
        self, 
        query: str, 
        answer: str, 
        chunks_used: List[str] = None,
        chunk_ids: List[int] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        
        Args:
            query: User's question
            answer: Generated answer
            chunks_used: Text of chunks used for generation
            chunk_ids: Global indices of chunks used
            metadata: Additional metadata (scores, etc.)
        """
        turn = ConversationTurn(
            query=query,
            answer=answer,
            chunks_used=chunks_used or [],
            chunk_ids=chunk_ids or [],
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.history.append(turn)
        
        # Extract and update topics
        self._update_topics(query)
    
    def get_recent_turns(self, n: int = 3) -> List[ConversationTurn]:
        """
        Get the N most recent conversation turns.
        
        Args:
            n: Number of turns to retrieve
            
        Returns:
            List of recent conversation turns (oldest to newest)
        """
        return list(self.history)[-n:] if self.history else []
    
    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the most recent conversation turn."""
        return self.history[-1] if self.history else None
    
    def get_conversation_context(self, max_turns: int = 2) -> str:
        """
        Format recent conversation as context string.
        
        Args:
            max_turns: Maximum number of turns to include
            
        Returns:
            Formatted conversation context
        """
        recent = self.get_recent_turns(max_turns)
        if not recent:
            return ""
        
        context_parts = []
        for i, turn in enumerate(recent, 1):
            context_parts.append(f"Previous Question {i}: {turn.query}")
            # Use first 200 chars of answer for context
            answer_preview = turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer
            context_parts.append(f"Previous Answer {i}: {answer_preview}")
        
        return "\n".join(context_parts)
    
    def get_previous_topics(self) -> List[str]:
        """Get list of topics discussed in conversation."""
        return self._topics.copy()
    
    def get_previous_chunk_ids(self, n_turns: int = 2) -> List[int]:
        """
        Get chunk IDs used in recent turns (for potential re-ranking).
        
        Args:
            n_turns: Number of recent turns to consider
            
        Returns:
            List of unique chunk IDs
        """
        chunk_ids = set()
        for turn in self.get_recent_turns(n_turns):
            chunk_ids.update(turn.chunk_ids)
        return list(chunk_ids)
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.history.clear()
        self._topics.clear()
    
    def is_empty(self) -> bool:
        """Check if conversation history is empty."""
        return len(self.history) == 0
    
    def _update_topics(self, query: str) -> None:
        """
        Extract and update topics from query.
        Simple keyword-based extraction (can be enhanced with NER later).
        
        Args:
            query: User's question
        """
        # Extract capitalized terms and important keywords
        # This is a simple heuristic - can be improved with NLP
        keywords = [
            "transaction", "acid", "isolation", "consistency", "durability",
            "b+tree", "b+ tree", "index", "hash", "join", "normalization",
            "sql", "query", "concurrency", "lock", "deadlock", "recovery",
            "relation", "schema", "key", "foreign key", "primary key",
            "er model", "entity", "attribute", "aggregation"
        ]
        
        query_lower = query.lower()
        for keyword in keywords:
            if keyword in query_lower and keyword not in self._topics:
                self._topics.append(keyword)
        
        # Keep only last 10 topics
        self._topics = self._topics[-10:]
    
    def __len__(self) -> int:
        """Return number of turns in memory."""
        return len(self.history)
    
    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self.history)}, max_turns={self.max_turns})"
