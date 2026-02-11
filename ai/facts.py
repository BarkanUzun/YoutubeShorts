# generate_animal_facts
from ai.openai_client import get_openai_client
import json
import logging

def generate_animal_facts(animal: str) -> list[str]:
	"""Generate 3 short, surprising facts about the given animal."""
	try:
		client = get_openai_client()
		prompt = (
			f"Give me EXACTLY 3 short, surprising facts about {animal}.\n"
			"Return ONLY a JSON array of 3 strings, no other text or formatting.\n"
			'Example format: ["Fact 1", "Fact 2", "Fact 3"]'
		)
		# The actual OpenAI API call would go here, but for now, simulate a response:
		# response = client.chat.completions.create(...)
		# content = response.choices[0].message.content
		# facts = json.loads(content)
		# For now, raise NotImplementedError to indicate this is a stub
		raise NotImplementedError("OpenAI API call not implemented in this stub.")
	except Exception as e:
		logging.warning(f"Falling back to template facts for '{animal}': {e}")
		a = animal.lower()
		return [
			f"{animal} is way stranger than it looks.",
			f"Every new detail about {a} sounds made up.",
			f"{animal} breaks the rules of what an animal should be.",
		]
