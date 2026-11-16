from typing import TypedDict, List


class GraphState(TypedDict):
    question: str
    search_query: str   # clear, standalone version of the question (step 1)
    search_keywords: str  # keyword-style version used as an extra search (step 1)
    answer_style: str   # "short" or "descriptive" (step 1)
    documents: List     # retrieved chunks, then only those with relevant facts
    extracts: List      # facts extracted from each kept chunk (step 3)
    answer: str
    evidence: List
    chat_history: List
