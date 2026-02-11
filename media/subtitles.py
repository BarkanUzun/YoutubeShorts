# subtitle + SRT logic
import tempfile
import os

def generate_final_video_with_subtitles(animal, script, audio_path, video_path, output_path):
	"""Generate SRT, burn subtitles, and output final video."""
	# Generate SRT file from script (one line per spoken line, simple timing)
	lines = [l.strip() for l in script.split("\n") if l.strip()]
	duration = _probe_media_duration(audio_path) or 15.0
	per_line = duration / max(1, len(lines))
	srt = ""
	for idx, line in enumerate(lines):
		start = idx * per_line
		end = min(duration, (idx + 1) * per_line)
		srt += f"{idx+1}\n"
		srt += f"{int(start//60):02}:{int(start%60):02},{int((start%1)*1000):03} --> {int(end//60):02}:{int(end%60):02},{int((end%1)*1000):03}\n"
		srt += line + "\n\n"
	with tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="w", encoding="utf-8") as f:
		f.write(srt)
		srt_path = f.name
	# Burn subtitles
	burn_subtitles(video_path, srt_path, output_path)
	os.remove(srt_path)
