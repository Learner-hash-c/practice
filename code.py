import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
api_key = os.environ['OPENROUTER_API_KEY']

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "inclusionai/ling-3.0-flash-sante:free",
        "temperature": 2,
        "top_p": 0,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Write a one-sentence opening line for a story."},
        ],
    },
)

if response.status_code != 200:
    print("Request failed:", response.status_code, response.text)
else:
    print(response.json()["choices"][0]["message"]["content"])