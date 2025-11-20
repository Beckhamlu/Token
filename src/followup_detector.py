"""
followup_detector.py

Detects whether a query is a follow-up to previous conversation or standalone.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class FollowUpAnalysis:
    is_followup: bool
    confidence: float  # 0.0 to 1.0
    signals: List[str]  # What signals indicated follow-up
    references: List[str]  # Specific references found (pronouns, etc.)
    
    def __repr__(self) -> str:
        return f"FollowUpAnalysis(is_followup={self.is_followup}, confidence={self.confidence:.2f})"


class FollowUpDetector:
    """    
    What I'm using to detect follow-up:
    1. Pronouns: "it", "that", "this", "they", "those"
    2. Comparative: "compared to", "versus", "what about", "how about"
    3. Continuation: "also", "additionally", "furthermore", "and"
    4. References: "the algorithm", "the property", "that approach"
    5. Short queries: < 5 words often assume context
    """
    
    # Pronoun patterns
    PRONOUN_PATTERNS = [
        r'\bit\b', r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
        r'\bthey\b', r'\bthem\b', r'\btheir\b'
    ]
    
    # Comparative patterns
    COMPARATIVE_PATTERNS = [
        r'\bcompared to\b', r'\bversus\b', r'\bvs\b', r'\bvs\.\b',
        r'\bwhat about\b', r'\bhow about\b', r'\binstead\b',
        r'\bdifference between\b', r'\bdiffers from\b'
    ]
    
    # Continuation patterns
    CONTINUATION_PATTERNS = [
        r'\balso\b', r'\badditionally\b', r'\bfurthermore\b',
        r'\bmoreover\b', r'\bbesides\b', r'^and\b'
    ]
    
    # Reference patterns (assumes context)
    REFERENCE_PATTERNS = [
        r'\bthe algorithm\b', r'\bthe property\b', r'\bthe approach\b',
        r'\bthe method\b', r'\bthe technique\b', r'\bthe structure\b',
        r'\bthe concept\b', r'\bthe system\b', r'\bthat concept\b',
        r'\bthis property\b'
    ]
    
    # Question starters that typically need context
    CONTEXT_QUESTION_STARTERS = [
        "why does it", "how does it", "when does it", "where is it",
        "why is that", "how is that", "what makes it", "why do they"
    ]
    
    def __init__(self, short_query_threshold: int = 5):
        """
        Initialize follow-up detector.
        
        Args:
            short_query_threshold: Word count below which query is considered short
        """
        self.short_query_threshold = short_query_threshold
        
        # Compile patterns
        self._pronoun_re = re.compile('|'.join(self.PRONOUN_PATTERNS), re.IGNORECASE)
        self._comparative_re = re.compile('|'.join(self.COMPARATIVE_PATTERNS), re.IGNORECASE)
        self._continuation_re = re.compile('|'.join(self.CONTINUATION_PATTERNS), re.IGNORECASE)
        self._reference_re = re.compile('|'.join(self.REFERENCE_PATTERNS), re.IGNORECASE)
    
    def analyze(
        self, 
        query: str, 
        has_history: bool = True,
        previous_query: Optional[str] = None
    ) -> FollowUpAnalysis:
        """
        Analyze whether query is a follow-up.
        
        Args:
            query: Current query to analyze
            has_history: Whether conversation history exists
            previous_query: Previous query for context (optional)
            
        Returns:
            FollowUpAnalysis with detection results
        """
        if not has_history:
            # No history = can't be a follow-up
            return FollowUpAnalysis(
                is_followup=False,
                confidence=1.0,
                signals=["no_conversation_history"],
                references=[]
            )
        
        query_lower = query.lower()
        signals = []
        references = []
        confidence_scores = []
        
        # 1. Check for pronouns
        pronoun_matches = self._pronoun_re.findall(query_lower)
        if pronoun_matches:
            signals.append(f"pronouns: {', '.join(set(pronoun_matches))}")
            references.extend(pronoun_matches)
            confidence_scores.append(0.8)
        
        # 2. Check for comparative language
        comparative_matches = self._comparative_re.findall(query_lower)
        if comparative_matches:
            signals.append(f"comparative: {', '.join(set(comparative_matches))}")
            confidence_scores.append(0.9)
        
        # 3. Check for continuation words
        continuation_matches = self._continuation_re.findall(query_lower)
        if continuation_matches:
            signals.append(f"continuation: {', '.join(set(continuation_matches))}")
            confidence_scores.append(0.7)
        
        # 4. Check for vague references
        reference_matches = self._reference_re.findall(query_lower)
        if reference_matches:
            signals.append(f"references: {', '.join(set(reference_matches))}")
            references.extend(reference_matches)
            confidence_scores.append(0.85)
        
        # 5. Check for context-dependent question starters
        for starter in self.CONTEXT_QUESTION_STARTERS:
            if query_lower.startswith(starter):
                signals.append(f"context_question: {starter}")
                confidence_scores.append(0.75)
                break
        
        # 6. Check query length (short queries often assume context)
        word_count = len(query.split())
        if word_count <= self.short_query_threshold:
            signals.append(f"short_query: {word_count} words")
            # Lower confidence for short queries alone
            confidence_scores.append(0.3)
        
        # 7. Check for topic continuation (if previous query provided)
        if previous_query:
            topic_overlap = self._check_topic_overlap(query_lower, previous_query.lower())
            if topic_overlap:
                signals.append("topic_continuation")
                confidence_scores.append(0.6)
        
        is_followup = len(signals) > 0
        
        # Calculate confidence
        if confidence_scores:
            max_conf = max(confidence_scores)
            avg_conf = sum(confidence_scores) / len(confidence_scores)
            confidence = max_conf if max_conf > 0.7 else avg_conf
        else:
            confidence = 0.0
        
        # If only short query signal, reduce confidence
        if signals == [f"short_query: {word_count} words"]:
            confidence = min(confidence, 0.4)
        
        return FollowUpAnalysis(
            is_followup=is_followup,
            confidence=confidence,
            signals=signals,
            references=references
        )
    
    def _check_topic_overlap(self, query: str, previous_query: str) -> bool:
        """
        Check if queries share topic-related terms.
        
        Args:
            query: Current query (lowercase)
            previous_query: Previous query (lowercase)
            
        Returns:
            True if significant topic overlap detected
        """
        stopwords = {'what', 'how', 'why', 'when', 'where', 'does', 'are', 'is',
                     'the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'about',
                     'that', 'this', 'these', 'those', 'from', 'can', 'you'}
        
        def extract_keywords(text: str) -> set:
            words = re.findall(r'\b\w+\b', text)
            return {w for w in words if len(w) > 4 and w not in stopwords}
        
        query_keywords = extract_keywords(query)
        prev_keywords = extract_keywords(previous_query)
        
        overlap = query_keywords & prev_keywords
        
        return len(overlap) > 0
    
    def is_followup(self, query: str, has_history: bool = True) -> bool:
        """
        Simple boolean check if query is a follow-up.
        
        Args:
            query: Query to check
            has_history: Whether conversation history exists
            
        Returns:
            True if likely a follow-up
        """
        analysis = self.analyze(query, has_history)
        return analysis.is_followup and analysis.confidence > 0.5


# Example usage and test cases
if __name__ == "__main__":
    detector = FollowUpDetector()
    
    test_cases = [
        # Follow-ups
        ("What about hash indexes?", True),
        ("How does it work?", True),
        ("Explain that concept in more detail", True),
        ("Compared to B+ trees, what are the advantages?", True),
        ("And what about durability?", True),
        ("Why is this important?", True),
        
        # Standalone
        ("What are ACID properties?", False),
        ("Explain the concept of database normalization", False),
        ("How do transactions work in databases?", False),
    ]
    
    print("Testing Follow-Up Detector:\n")
    for query, expected_followup in test_cases:
        analysis = detector.analyze(query, has_history=True)
        status = "✓" if analysis.is_followup == expected_followup else "✗"
        print(f"{status} Query: {query}")
        print(f"  Result: {analysis}")
        if analysis.signals:
            print(f"  Signals: {', '.join(analysis.signals)}")
        print()
