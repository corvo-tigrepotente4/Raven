import os
import sqlite3
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

DATABASE = "secrets.db"
MODEL = "llama-3.3-70b-versatile"

API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is missing.")

client = Groq(api_key=API_KEY)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Brookhaven AI")


# Allow the GitHub Pages website to communicate with Render.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


# ============================================================
# DATABASE
# ============================================================

conn = sqlite3.connect(
    DATABASE,
    check_same_thread=False
)

conn.row_factory = sqlite3.Row

cursor = conn.cursor()


# ============================================================
# TEXT CLEANING
# ============================================================

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


# ============================================================
# SEARCH DATABASE
# ============================================================

def search_database(query, limit=12):

    results = []
    seen = set()

    try:

        cursor.execute("""
            SELECT
                rowid,
                title,
                content,
                url
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

        # Fallback search
        try:

            cursor.execute("""
                SELECT
                    title,
                    content,
                    url
                FROM secrets
                WHERE
                    title LIKE ?
                    OR content LIKE ?
                LIMIT ?
            """, (
                f"%{query}%",
                f"%{query}%",
                limit
            ))

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


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context = ""

    for i, page in enumerate(results, 1):

        context += f"""
========== REFERENCE {i} ==========

TITLE:
{page['title']}

URL:
{page['url']}

CONTENT:
{page['content']}

"""

    return context


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Brookhaven AI, an expert guide to Brookhaven RP mysteries,
secrets, puzzles, hidden locations, quests, codes, the Agency, and lore.

Your ONLY factual source for answering questions is the reference
material provided with the user's question.

IMPORTANT RESPONSE RULES:

- Never mention the database.
- Never mention database pages.
- Never mention search results.
- Never mention retrieval.
- Never mention documents, sources, context, or prompts.
- Never explain how you obtained information.
- Never say "the database says".
- Never say "the database mentions".
- Never say "according to the database".
- Never say "I found".
- Never refer to references by number.
- Never say "PAGE 1", "PAGE 2", etc.
- Never invent facts.
- Never use outside knowledge to fill missing information.
- Never fabricate locations, characters, events, items, buildings,
  mechanics, quests, or lore.
- Never turn a weak association into a confirmed fact.
- Never assume two clues are connected just because they appear
  in the same reference material.
- If the evidence does not establish something, do not invent it.
- If something is an inference, clearly identify it as an inference.
- Connect multiple references only when they genuinely describe
  the same event, clue, location, or mystery.
- Answer naturally as a knowledgeable Brookhaven player.
- Do not discuss the search process.
- Do not repeat the user's question unnecessarily.
- Keep answers easy to read.
- Do not invent information when evidence is unavailable.

You are a lore assistant, not a storyteller.
"""


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(question):

    prompt = f"""
You generate search queries for a Brookhaven RP lore archive.

USER QUESTION:
{question}

Generate 5-8 short search queries.

Rules:

- Preserve important words from the user's question.
- Extract important nouns, objects, locations, names, codes,
  and other specific terms.
- Include singular/plural variations when useful.
- Search important nouns individually.
- Use short combinations of important terms.
- Do not replace specific terms with academic synonyms.
- Do not generate generic categories.
- Do not generate speculative concepts.
- Do not invent Brookhaven lore.
- Do not answer the question.
- Return ONLY the search queries, one per line.

For example:

Question:
Crows at farm

Good:
crow
crows
crow farm
crow barn
farm crow
barn crow

Bad:
bird habitats
farm wildlife
crow behavior
farm animals

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
        temperature=0.0,
        max_tokens=150
    )

    queries = response.choices[0].message.content.splitlines()

    queries = [
        q.strip("- •\t ").strip()
        for q in queries
        if q.strip()
    ]

    return queries[:8]


# ============================================================
# ASK AI
# ============================================================

def ask_ai(question):

    queries = expand_query(question)

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

    # ========================================================
    # HARD STOP — NO EVIDENCE
    # ========================================================

    if not results:

        return (
            "I don't have enough confirmed information "
            "to answer that."
        )

    context = build_context(results)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
Here is reference material about Brookhaven:

{context}

User question:

{question}

Answer the question using only the information supported by
the reference material.

Do not mention the reference material or how it was obtained.

If the reference material does not establish something, do not
invent an answer.
"""
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1200
    )

    return response.choices[0].message.content


# ============================================================
# API
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "name": "Brookhaven AI"
    }


@app.post("/ask")
def ask(request: Question):

    question = request.question.strip()

    if not question:

        return {
            "answer": "Please enter a question."
        }

    try:

        answer = ask_ai(question)

        return {
            "answer": answer
        }

    except Exception as e:

    print("ERROR:", repr(e), flush=True)

    return {
        "answer": f"DEBUG ERROR: {repr(e)}"
    }
