import os
import re
import json
import requests
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).parent / ".env")
api = os.environ["API"]  # your .env must contain a line: API=your_key

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "dots-studio/dots-3-note-preview:free"


# ============================================================
# RAG SETUP  (indexing phase: load -> clean -> chunk -> embed -> store)
# ============================================================
SAFE_DIR = Path(__file__).parent / "docs"   # the only folder the agent may read / search

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60
MAX_DISTANCE = 1.3   # chunks with a distance above this are treated as "not relevant"
TOP_K = 3


def load_documents(folder):
    """Return a list of (text, source_filename, page_label) for every PDF/txt/md in the folder."""
    items = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() == ".pdf":
            for page in PyPDFLoader(str(f)).load():
                text = re.sub(r"\s+", " ", page.page_content).strip()  # fix the word-per-line problem
                if text:
                    items.append((text, f.name, page.metadata.get("page_label", "?")))
        elif f.suffix.lower() in (".txt", ".md"):
            text = re.sub(r"\s+", " ", f.read_text(encoding="utf-8")).strip()
            if text:
                items.append((text, f.name, "1"))
    return items


def build_index():
    SAFE_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(Path(__file__).parent / "db"))

    # rebuild from scratch each start so the index never goes stale
    # (fine for small docs; for big ones you'd only re-index changed files)
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    collection = client.create_collection("docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    ids, texts, metas = [], [], []
    for text, source, page in load_documents(SAFE_DIR):
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{source}-p{page}-{i}")
            texts.append(chunk)
            metas.append({"source": source, "page": str(page)})

    if texts:
        collection.add(ids=ids, documents=texts, metadatas=metas)
    print(f">>> RAG index ready: {len(texts)} chunks from {SAFE_DIR}")
    return collection


collection = build_index()


# ---------- the tools ----------
def calculator_tool(a, sign, b):
    print(f">>> TOOL CALLED: {a} {sign} {b}")
    if sign == "+":
        return a + b
    elif sign == "-":
        return a - b
    elif sign == "*":
        return a * b
    elif sign == "/":
        if b == 0:
            return "Error: cannot divide by zero."
        return a / b
    return "Error: unsupported operation."
def pokemon_tool(name):
    print(f">>> TOOL CALLED: {name}")
    url = f"https://pokeapi.co/api/v2/pokemon/{name}"
    r = requests.get(url)
    if r.status_code == 404:
        return f"Error: no Pokemon named '{name}' found."
    if r.status_code != 200:
        return f"Error: PokeAPI failed ({r.status_code})."
    d = r.json()
    types = ", ".join(t["type"]["name"] for t in d["types"])
    abilities = ", ".join(a["ability"]["name"] for a in d["abilities"])
    stats = ", ".join(f"{s['stat']['name']}: {s['base_stat']}" for s in d["stats"])
    return (
        f"{d['name']} | types: {types} | abilities: {abilities} | "
        f"height: {d['height']} | weight: {d['weight']} | stats: {stats}"
    )
def web_search(querry):
    Tavila_api = os.environ["T_api"]

    r = requests.post(
        "https://api.tavily.com/search",
        headers= {"Authorization" : f"Bearer {Tavila_api}"},
        json={"query" : querry }
    )
    output = r.json()
    return(output)

def read_file_tool(filename):
    print(f">>> TOOL CALLED: read_file '{filename}'")
    path = (SAFE_DIR / filename).resolve()
    if SAFE_DIR.resolve() not in path.parents:
        return "Error: access denied. Only files in the docs folder can be read."
    if not path.exists():
        return f"Error: file '{filename}' not found."
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "Error: this file isn't plain text."
    return text[:3000]   # trim so it doesn't flood the context window


# ---------- NEW: the RAG tool (query phase: embed question -> retrieve -> filter) ----------
def search_docs(question):
    print(f">>> TOOL CALLED: search_docs '{question}'")
    if collection.count() == 0:
        return "No documents are indexed. Put PDF/txt/md files in the docs folder and restart."

    res = collection.query(query_texts=[question], n_results=TOP_K)

    kept = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        keep = dist <= MAX_DISTANCE
        # ALWAYS log what was retrieved: most RAG debugging is just reading this
        print(f"    [{meta['source']} p{meta['page']}] distance={dist:.3f} {'KEEP' if keep else 'DROP'}")
        if keep:
            kept.append(f"[source: {meta['source']}, page {meta['page']}]\n{doc}")

    if not kept:
        return "No relevant information found in the documents."
    return "\n\n".join(kept)


# name (text the model sends) -> real function
TOOLS = {"calculator_tool": calculator_tool,
          "pokemon_tool": pokemon_tool,
          "web_search" : web_search,
          "file_read" : read_file_tool,
          "search_docs": search_docs}


def run_tool(name, args_json):
    """Safely run a tool. Always returns a string, never crashes."""
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError:
        return "Error: arguments were not valid JSON."
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available: {list(TOOLS)}"
    try:
        return str(TOOLS[name](**args))
    except Exception as e:
        return f"Error: tool failed ({e})."


# ---------- tell the model about the tools ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator_tool",  # must match the key in TOOLS exactly
            "description": "Perform basic calculations. Always use this for any maths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number"},
                    "sign": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/"],
                        "description": "The mathematical sign",
                    },
                    "b": {"type": "number", "description": "The second number"},
                },
                "required": ["a", "sign", "b"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "pokemon_tool",
        "description": "Look up a Pokemon's types, abilities, height, weight and base stats by its name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The Pokemon's name, e.g. pikachu"}
            },
            "required": ["name"],
        },
    },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current or general information that is NOT in the user's documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "querry": {"type": "string", "description": "what do you wanna search"}
                },
                "required": ["querry"],
            },
        },
        },
    {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read the full raw text of one plain-text file in the docs folder by exact filename. For answering questions about document content, prefer search_docs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "file name"}
                    },
                    "required": ["filename"],
                },
            },
            },
    # NEW
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the user's own documents (indexed PDFs, notes, scripts) and return the most "
                "relevant passages with their source and page. Use this FIRST for any question that "
                "could be answered by the user's documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or topic to look up, phrased as a clear standalone search query",
                    }
                },
                "required": ["question"],
            },
        },
    },
]

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant with access to tools.\n"
            "- For questions that could be answered by the user's documents, call search_docs first.\n"
            "- Answer document questions ONLY from the passages search_docs returns. "
            "If it returns nothing relevant, say \"I don't know based on the documents\" and do not guess.\n"
            "- When you use document passages, mention the source file and page.\n"
            "- Use calculator_tool for maths, pokemon_tool for Pokemon, and web_search only for "
            "things that are not in the documents."
        ),
    }
]

# ---------- chat loop ----------
try:
    while True:
        user = input("you: ")
        if user.strip().lower() == "exit":
            break

        start = len(messages)  # remember where this turn begins
        messages.append({"role": "user", "content": user})
        used_tool = False

        for step in range(5):  # limit so the model can't loop forever
            try:
                response = requests.post(
                    URL,
                    headers={"Authorization": f"Bearer {api}"},
                    json={"model": MODEL, "messages": messages, "tools": tools},
                    timeout=60,
                )
            except requests.exceptions.RequestException as e:
                print("Network problem:", e)
                del messages[start:]
                break

            if response.status_code != 200:
                print("Request failed:", response.status_code, response.text)
                del messages[start:]  # remove this turn so history stays clean
                break

            msg = response.json()["choices"][0]["message"]

            # 1) the model is asking us to run a tool
            if msg.get("tool_calls"):
                used_tool = True
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content") or "",
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": c["function"],
                            }
                            for c in msg["tool_calls"]
                        ],
                    }
                )
                for call in msg["tool_calls"]:
                    result = run_tool(
                        call["function"]["name"], call["function"]["arguments"]
                    )
                    print(f"    tool result: {result}")
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": result}
                    )
                continue  # send the tool result back to the model

            # 2) normal text reply
            text = msg.get("content") or ""
            if not text:
                print("bot: (empty reply)  RAW:", msg)
                del messages[start:]
                break

            print(f"bot: {text}")
            print("[tool used]" if used_tool else "[answered directly, no tool]")
            messages.append({"role": "assistant", "content": text})
            break
        else:
            print("Stopped: too many tool steps in one turn.")
            del messages[start:]

except KeyboardInterrupt:
    print("\nBye!")