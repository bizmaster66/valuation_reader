import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def get_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY가 .env에 없습니다.")
    return genai.Client(api_key=api_key)

def google_search_tool():
    # Grounding(구글 검색) 도구
    return types.Tool(google_search=types.GoogleSearch())
