import sqlite3
import re
from groq import Groq

# ==========================
# CONFIG
# ==========================

API_KEY = "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"

DATABASE = "secrets.db"

MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=API_KEY)

# ==========================
# DATABASE
# ==========================

conn = sqlite3.connect(DATABASE)
conn.row_factory = sqlite3.Row

cursor = conn.cursor()


# ==========================
# TEXT CLEANING
# ==========================

def clean_text(text):
    if not text:
        return ""

    # Remove markdown links
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

    # Remove extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ==========================
# SEARCH DATABASE
# ==========================

def search_database(query, limit=12):
    results = []
    seen = set()

    try:
        cursor.execute("""
            SELECT rowid, title, content, url
            FROM secrets_fts
            WHERE secrets_fts MATCH ?
            LIMIT ?
        """, (query, limit))

        rows = cursor.fetchall()

        for row in rows:
            key = row["url"] or row["title"]

            if key in seen:
                continue

            seen.add(key)

            results.append({
                "title": row["title"],
                "content": clean_text(row["content"]),
                "url": row["url"]
            })

    except Exception:
        pass

    return results

# ==========================
# BUILD CONTEXT
# ==========================

def build_context(results):

    context = ""

    for i, page in enumerate(results, 1):

        context += f"""
========== PAGE {i} ==========
TITLE:
{page['title']}

URL:
{page['url']}

CONTENT:
{page['content']}

"""

    return context


# ==========================
# TEST
# ==========================
# ==========================
# PROMPT
# ==========================

SYSTEM_PROMPT = """
You are Brookhaven AI, an expert guide to Brookhaven RP mysteries,
secrets, puzzles, hidden locations, quests, codes, the Agency, and lore.

You are given reference material containing information about Brookhaven.
Use that material to answer the user's question accurately.

IMPORTANT RESPONSE RULES:

- Never mention the database, database pages, search results, retrieval,
  documents, sources, context, prompts, or how you obtained information.
- Never say "the database says", "the database mentions", "according to
  the database", "I found", "the retrieved pages", or similar phrases.
- Speak directly as a knowledgeable Brookhaven lore expert.
- Do not discuss your search process.
- Do not explain why a search result was or was not found.
- Do not invent facts.
- Do not turn weak associations into established facts.
- If the provided information does not establish something, say that
  clearly and briefly.
- Distinguish confirmed information from theories or inferences.
- Only call something an inference when there is actual evidence that
  supports the connection.
- Do not invent connections merely because two things appear in the
  same reference material.
- Do not repeat the user's question unnecessarily.
- Give useful, direct answers.
- Connect information from multiple references when they genuinely
  describe the same event, clue, location, or mystery.
- Keep answers easy to read.
- Do not invent stuff that is not real if no pages found in database.

Your answer should feel like a knowledgeable Brookhaven player explaining
the mystery to another player, not like an AI analyzing documents.

Answer naturally and confidently when the evidence is clear.
Keep answers easy to read.

Avoid saying:
"I couldn't find..."
"The database says..."

Instead speak naturally.
"""

# ==========================
# ASK AI
# ==========================
def expand_query(question):

    prompt = f"""
You are a search-query generator for a Brookhaven RP mystery database.

Convert the user's question into 5-8 short search queries that
could retrieve relevant pages from a SQLite FTS database.

Rules:
- Extract important nouns and locations.
- Include singular/plural variations when useful.
- Include obvious related terms.
- Do NOT answer the question.
- Do NOT invent specific lore facts.
- Keep each query short.
- Return ONLY the queries, one per line.

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=150
    )

    queries = response.choices[0].message.content.splitlines()

    queries = [
        q.strip("- •\t ").strip()
        for q in queries
        if q.strip()
    ]

    return queries[:8]





def ask_ai(question):

    queries = expand_query(question)

    print("\nSearch queries:")
    for q in queries:
        print(" -", q)

    all_results = []
    seen = set()

    # Search every generated query
    for query in queries:

        pages = search_database(query, limit=5)

        for page in pages:

            key = page["url"] or page["title"]

            if key in seen:
                continue

            seen.add(key)
            all_results.append(page)

    # Limit context
    results = all_results[:15]

    context = build_context(results)

    print(f"\nSearching database... Found {len(results)} relevant page(s).")
    print("Thinking...\n")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
DATABASE:

{context}

QUESTION:

{question}

Answer using the database above.

Important:
- Use relevant information from multiple pages when appropriate.
- If the question refers to something indirectly, connect related
  evidence from the retrieved pages.
- Do not claim that information is absent merely because there was
  no exact phrase match.
- If you make an inference, clearly identify it as an inference.
"""
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# ==========================
# TERMINAL CHAT
# ==========================

if __name__ == "__main__":

    print("=" * 50)
    print("Brookhaven AI")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:

        question = input("\nYou: ")

        if question.lower() in ("exit", "quit"):
            break

        try:
            answer = ask_ai(question)

            print("\nAI:\n")
            print(answer)

        except Exception as e:
            print("\nERROR:")
            print(e)

    conn.close()