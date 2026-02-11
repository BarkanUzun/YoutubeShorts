# Animal Facts Channel

Generate short animal fact videos for TikTok and YouTube Shorts with:
- AI-generated facts and scripts
- AI voiceover
- Auto subtitles
- Vertical video assembly
- Optional YouTube upload

## Features

- Generate one random animal video
- Generate a specific animal video
- Batch-generate multiple videos
- List videos ready to upload
- Upload generated videos to YouTube Shorts

## Requirements

- Python 3.10+ (3.11+ recommended)
- `ffmpeg` installed and available in your PATH
- API keys:
  - `OPENAI_API_KEY`
  - `PEXELS_API_KEY`

## Installation

```bash
cd /Users/barkanuzun/animal-facts-channel

python3 -m venv .venv
source .venv/bin/activate

pip install openai requests opencv-python google-auth google-auth-oauthlib google-api-python-client
```

Or:

```bash
pip install -r requirements.txt
```

## Configuration

You can adjust runtime defaults in:

`/Users/barkanuzun/animal-facts-channel/config.json`

Current keys:
- `output_dir`
- `speech_speed`
- `max_title_length`
- `ffmpeg_zoom_max`

## Environment Variables

Set your keys before running:

```bash
export OPENAI_API_KEY="your_openai_key"
export PEXELS_API_KEY="your_pexels_key"
```

## Usage

Run from project root:

```bash
python animal_facts_ai.py
```

Common commands:

```bash
# Generate one random animal video
python animal_facts_ai.py

# Generate one specific animal video
python animal_facts_ai.py --animal dolphin

# Batch generate 5 videos
python animal_facts_ai.py --batch 5

# List videos ready to upload
python animal_facts_ai.py --list-ready

# Upload one generated animal video to YouTube
python animal_facts_ai.py --upload dolphin
```

## YouTube Upload Setup

For upload commands, place Google OAuth client file in project root:

`/Users/barkanuzun/animal-facts-channel/client_secrets.json`

On first upload, the app creates:

`/Users/barkanuzun/animal-facts-channel/youtube_token.json`

Both are ignored by `.gitignore` and should never be committed.

## Output Files

Generated files are written under:

`/Users/barkanuzun/animal-facts-channel/outputs`

Typical output includes:
- `*_script.txt`
- `*_voice.mp3`
- `*.srt`
- `*_final.mp4`
- `*_meta.json`

## GitHub Safety

The project includes:

`/Users/barkanuzun/animal-facts-channel/.gitignore`

It excludes local secrets, tokens, virtual env, logs, and generated media.

Before committing:

```bash
git status
```

Verify no secrets or local-only files are staged.
