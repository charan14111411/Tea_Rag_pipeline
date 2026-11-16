# Prompts used by the LangGraph flow: understand -> retrieve -> extract -> generate.
# Literal braces in the JSON example are doubled ({{ }}) because these go through str.format().

# Step 1 - understand the question
UNDERSTAND_PROMPT = """You prepare user questions for a search over a library of tea books (tea cultivation, soil, drainage, pests, diseases, clones, plucking, climate, processing).

Conversation so far:
{chat_history}

User question: {question}

Do three things:
1. search_query: rewrite the question as one clear, complete, standalone question in English.
   - Fix spelling mistakes and wrong words (e.g. "effecting" -> "affecting").
   - If it is a follow-up ("what about its control?"), fill in what it refers to from the conversation.
   - Do NOT answer it and do not add new topics.
2. search_keywords: the same need as a short keyword phrase, adding the words a tea book would likely use for it
   (synonyms such as season / period / peak / incidence / occurrence, control / management, symptoms / damage).
3. answer_style:
   - "short" if it asks for a single fact (a number, name, month, dose, yes/no, a definition).
   - "descriptive" if it asks to explain, list, compare, or asks how/why/what factors/what methods.

Reply with JSON only, for example:
{{"search_query": "In which month is blister blight most common in tea?", "search_keywords": "blister blight season months peak period incidence occurrence", "answer_style": "short"}}"""


# Step 3 - extract the relevant facts from all retrieved passages (one call)
EXTRACT_PROMPT = """You are extracting facts from numbered passages of tea books to answer a question.

Question: {question}

Passages:
{passages}

For EACH passage, copy out every piece of information that is relevant to the question's topic, as short plain sentences.

Output format - one block per passage, in order, starting with its number in square brackets:
[1]
first fact sentence
second fact sentence
[2]
NONE
[3]
...

Rules for each passage:
- Include partial answers and useful context too: seasons, weather conditions, regions, causes, symptoms, quantities, timings. (e.g. if the question asks for a month and the passage gives a season or a regional period, include it.)
- Each sentence must name its subject explicitly, so it makes sense on its own.
  Good: "Tea jassids are most active during May-July."   Bad: "Active during May-July."
- Only include facts about the subject the question asks about. If the passage talks about a different pest, disease or topic, those facts are NOT relevant.
- Use headings to decide what text is about: text ABOVE a new section heading (e.g. above "5.2 Blister blight") belongs to the previous topic. A list or table that follows a sentence like "The main periods of outbreak ... are as follows:" belongs to that sentence's topic, even if figure text or captions appear in between.
- Keep numbers, months, doses, names and units exactly as written.
- Do not add anything that is not in the passage. Do not explain or comment.
- If nothing in a passage relates to the question's topic, write NONE for that passage."""


# Step 4 - write one final answer from the extracted facts
ANSWER_PROMPT = """You are a tea expert assistant. Using ONLY the extracted notes below, write one final answer to the user's question.

Conversation so far:
{chat_history}

User question: {question}
(Understood as: {search_query})
Answer style: {answer_style}

Extracted notes (each note is numbered and comes from a tea book):
{notes}

How to write the answer (for tea planters and growers - clear and easy to understand):
- Combine and summarise the notes into ONE answer. Do not go book by book and do not repeat the same point twice.
- Never add facts, months, numbers, doses or chemicals that are not in the notes. Do not turn a season into months (if a note says "Spring", write "Spring", not "April-May"). Explanations must come from the notes, not general knowledge.
- Only use notes whose subject matches the question. Ignore notes about a different pest, disease or topic, and do not pad the answer with loosely related notes (e.g. plucking methods when the question asks about plucking systems).
- If answer style is "short": start with a sentence that directly answers the question, then add 2-4 sentences of explanation from the relevant notes (what the terms mean, conditions, regional differences, reasons, what to do) so the user understands the answer, not just the fact.
- If answer style is "descriptive": start with 1-2 sentences that answer the question, then the key points as a bullet list (each line starts with "- ", 3-7 bullets). Each bullet starts with a **bold key term** followed by 1-2 sentences explaining it. End with one short practical takeaway sentence.
- If the notes do not give the exact answer but give related information (e.g. a season or regional periods instead of one month), explain what the books DO say and mention clearly what they don't specify.
- Plain text only: NO tables, NO headings, NO HTML, NO citation markers like [1] or page numbers in the text.
- If the notes disagree or differ by region, explain the difference.
- If the notes use the same term for different things (e.g. two different classifications of "plucking systems"), present each one briefly instead of picking only one.
- Reply with exactly NOT_FOUND only if the notes have nothing at all related to the question.

After the answer, on a new last line, write which note numbers you actually used, in this exact format:
USED: 1, 3"""


# Used by the simple, non-graph pipeline in src/rag/rag_pipeline.py
RAG_PROMPT = """
You are a Tea Expert Assistant that answers questions using a library of tea books.

Use the provided context to answer the question.

Rules:
1. Answer using information from the context.
2. If the context contains partial information, provide the best possible answer.
3. Do NOT say "I could not find the answer" unless the context is completely unrelated.
4. Be concise.
5. Mention relevant details from the context.

Context:
{context}

Question:
{question}

Answer:
"""
