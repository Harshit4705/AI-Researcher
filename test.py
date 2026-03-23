import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from main import app
from ai_researcher import APP

def test_ai_researcher():
    try:
        print("Invoking APP...")
        state = APP.invoke({
            "project_id": "test_project",
            "question": "Can you find papers about linear regression?",
            "chat_history": []
        })
        print("Success!")
        print(state)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_researcher()
