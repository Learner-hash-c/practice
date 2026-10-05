import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
api_key = os.environ['OPENROUTER_API_KEY']
messages = [
    {"role" : "system", "content" : ""}
]
while True:
    user = input("you: ")
    if user == "exit":
        break
    messages.append({"role" : "user", "content" : user})
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "inclusionai/ling-3.0-flash-sante:free",
            "messages": messages
        },
    )

    if response.status_code != 200:
        print("Request failed:", response.status_code, response.text)

    data = response.json()["choices"][0]["message"]["content"]
    
    print(f'bot: {data}')