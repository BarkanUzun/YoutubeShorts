# OpenAI client initialization
import os
import openai

def get_openai_client():
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("OPENAI_API_KEY environment variable is not set")
	return openai.OpenAI(api_key=api_key)
