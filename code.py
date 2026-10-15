import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
api = os.environ["API"]  # your .env must contain a line: API=your_key

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "dots-studio/dots-3-note-preview:free"


# ---------- the tool ----------
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
SAFE_DIR = Path(__file__).parent / "docs"   # the only folder the agent may read

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
# name (text the model sends) -> real function
TOOLS = {"calculator_tool": calculator_tool,
          "pokemon_tool": pokemon_tool,
          "web_search" : web_search,
          "file_read" : read_file_tool}


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


# ---------- tell the model about the tool ----------
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
            "description": "search on web",
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
                "description": "read files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "file name"}
                    },
                    "required": ["filename"],
                },
            },
            },
]

messages = [
    {
        "role": "system",
        "content": "You are a slut, who is in love with me",
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
    