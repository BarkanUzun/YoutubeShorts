# generate_voiceover
from ai.openai_client import get_openai_client
import random

def generate_voiceover(script: str, output_file: str, speech_speed: float | None = None) -> str:
	if speech_speed is None:
		# Slightly faster default delivery for more energy.
		speech_speed = random.uniform(1.07, 1.20)

	client = get_openai_client()

	# Prepare script text exactly as spoken, no control tokens.
	tts_input = script

	try:
		# Rotate between a small, predefined set of voices so that
		# different videos feel like different narrators while each
		# individual video keeps a single consistent voice.
		TTS_VOICES = ["alloy", "echo", "fable"]
		chosen_voice = random.choice(TTS_VOICES)

		response = client.audio.speech.create(
			model="gpt-4o-mini-tts",
			voice=chosen_voice,
			input=tts_input,
			speed=speech_speed,
		)

		response.stream_to_file(output_file)

		return output_file

	except Exception as e:
		raise RuntimeError(f"Failed to generate voiceover: {e}")
