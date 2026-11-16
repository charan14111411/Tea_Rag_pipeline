from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is not set. Please check your .env file.")

from langchain_groq import ChatGroq


def get_llm(reasoning_effort=None):
    """
    Main model. Used with default reasoning for the final answer, and with
    reasoning_effort="low" for fact extraction (faster, fewer tokens, still accurate).
    """
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        reasoning_effort=reasoning_effort,
    )

    return llm


def get_fallback_llm():
    """
    Used automatically when the main model hits a long rate limit (e.g. the free
    tier's daily token quota). It has its own separate Groq quota.
    """
    return ChatGroq(
        model=os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
    )


def get_fast_llm():
    """
    Smaller, faster model for the simplest step (understanding the question).
    Groq rate limits are per model, so this also takes load off the main model.
    """
    return ChatGroq(
        model=os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        reasoning_effort="low",
    )
