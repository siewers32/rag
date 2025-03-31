import os
import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", 5432)
}

# Example data to embed
texts = [
    "Wat is AI?",
    "De rol van PostgreSQL in dataverwerking.",
    "Hoe werkt een taalmodel?"
]

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

def insert_documents(texts):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    rows = [(text, embedding) for text in texts if (embedding := get_embedding(text))]
    execute_values(cur,
        "INSERT INTO documents (content, embedding) VALUES %s",
        [(text, Json(embedding)) for text, embedding in rows]
    )
    conn.commit()
    cur.close()
    conn.close()

def search_similar(question, top_k=3):
    embedding = get_embedding(question)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Zorg dat het embedding-formaat een vector is
    cur.execute("""
        SELECT content
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    """, (embedding, top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return [r[0] for r in results]

def generate_answer(question):
    context = "\n\n".join(search_similar(question))
    prompt = f"""
    Gebruik de volgende context om de vraag te beantwoorden.

    Context:
    {context}

    Vraag:
    {question}

    Antwoord:
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    insert_documents(texts)
    vraag = "Wat doet PostgreSQL precies?"
    antwoord = generate_answer(vraag)
    print("Vraag:", vraag)
    print("Antwoord:", antwoord)
