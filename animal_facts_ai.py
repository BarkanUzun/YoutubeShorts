"""
Animal Facts AI - Generate short animal fact scripts for TikTok/YouTube Shorts.
"""

# ============================================================================
# SECTION: Imports and global setup
# ============================================================================

import json
import logging
import os
import random
import subprocess
import sys
import textwrap
import re

import cv2
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from openai import OpenAI


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(console_format)

# File handler
file_handler = logging.FileHandler('animal_facts.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ============================================================================
# SECTION: Configuration and global constants
# ============================================================================
# Default configuration values
DEFAULT_CONFIG = {
    "output_dir": "outputs",
    # Faster default speech for a very excited tone
    "speech_speed": 1.3,
    "max_title_length": 60,
    "ffmpeg_zoom_max": 1.2,
    "require_video_background": True,
}

# Global config cache
_config_cache = None


def load_config(config_path: str = "config.json") -> dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the config file (default: "config.json").
    
    Returns:
        A dict with configuration values, using defaults for missing keys.
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            config.update(user_config)
            logger.info(f"Loaded config from {config_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {config_path}: {e}. Using defaults.")
        except IOError as e:
            logger.warning(f"Failed to read {config_path}: {e}. Using defaults.")
    else:
        logger.info(f"Config file not found at {config_path}. Using defaults.")
    
    _config_cache = config
    return config


# Canonical animal universe for video generation
ANIMALS = [
    "elephant", "octopus", "dolphin", "penguin", "cheetah", "owl", "koala", "kangaroo", "giraffe", "sloth", "chameleon", "hummingbird", "platypus", "axolotl", "mantis shrimp", "blobfish", "naked mole rat", "vampire bat", "star-nosed mole", "aye-aye", "pangolin", "cassowary", "shoebill", "proboscis monkey", "fossa", "leafy seadragon", "blue-ringed octopus", "goblin shark", "anglerfish", "frilled shark", "hagfish", "viperfish", "oarfish", "giant isopod", "pistol shrimp", "peacock mantis shrimp", "electric eel", "stonefish", "cone snail", "box jellyfish", "maned wolf", "sun bear", "binturong", "kakapo", "okapi", "saiga antelope", "markhor", "jerboa", "springhare", "sugar glider", "slow loris", "tarsier", "proboscis bat", "ghost bat", "colugo", "flying fox", "armadillo", "echidna", "tenrec", "tapir", "wombat", "quokka", "thorny devil", "frilled lizard", "marine iguana", "tuatara", "komodo dragon", "saltwater crocodile", "alligator snapping turtle", "mata mata turtle", "softshell turtle", "giant tortoise", "tokay gecko", "satanic leaf-tailed gecko", "glass frog", "poison dart frog", "surinam toad", "caecilian", "red-eyed tree frog", "hellbender", "olm", "giant salamander", "bombardier beetle", "atlas moth", "hercules beetle", "goliath beetle", "leaf insect", "stick insect", "orchid mantis", "giant weta", "antlion", "assassin bug", "velvet worm", "trapdoor spider", "camel spider", "tailless whip scorpion", "giant huntsman spider", "black widow spider", "brown recluse spider", "tarantula hawk wasp", "executioner wasp", "army ant", "bullet ant", "sea pig", "sea cucumber", "feather star", "sea angel", "sea butterfly", "narwhal", "beluga whale", "sperm whale", "orca", "dugong", "manatee", "blob sculpin", "mudskipper", "lungfish", "coelacanth", "arapaima", "pacu", "piranha", "electric catfish", "stargazer fish", "mola mola", "triggerfish", "parrotfish", "flying fish", "clownfish", "remora", "vampire squid", "giant squid", "colossal squid", "cuttlefish", "nautilus", "horseshoe crab", "king crab", "coconut crab", "giant spider crab", "lobster", "sea snake", "blue dragon nudibranch", "firefly squid", "sea wasp", "lyrebird", "bird-of-paradise", "hoatzin", "kookaburra", "horned screamer", "harpy eagle", "secretary bird", "marabou stork", "turkey vulture",
    # Tier 1 – algorithm & viewer safe
    "otter", "sea otter", "seal", "hamster", "hedgehog", "rabbit", "kitten", "cat", "puppy", "dog", "panda", "red panda", "capybara",
    # Tier 2 – friendly but surprisingly cool
    "parrot", "beluga whale", "manatee", "dugong",
    # Tier 3 – cute-weird
    "flying squirrel", "pufferfish", "sea angel",
]

TTS_VOICES: list[str] = ["verse"]
_LAST_TTS_VOICE: str | None = None


def get_random_animal() -> str:
    """Return a random animal name from the canonical list."""
    return random.choice(ANIMALS)


def generate_animal_facts(animal: str) -> list[str]:
    """Generate 3 short, surprising facts about the given animal.

    Uses the OpenAI API to return a JSON array of three strings.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    prompt = (
        f"Give me EXACTLY 3 short, surprising facts about {animal}.\n"
        "Return ONLY a JSON array of 3 strings, no other text or formatting.\n"
        'Example format: ["Fact 1", "Fact 2", "Fact 3"]'
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that returns clean JSON only, "
                        "with no extra commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("OpenAI facts response contained no choices")

        content = (choices[0].message.content or "").strip()

        try:
            facts = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse facts as JSON: {e}") from e

        if not isinstance(facts, list) or len(facts) != 3:
            raise ValueError("Expected a JSON array of exactly 3 fact strings")

        return [str(f).strip() for f in facts]

    except Exception as e:
        # Fall back to simple template-based facts so video generation
        # can continue even if the API response is malformed.
        logger.warning(f"Falling back to template facts for '{animal}': {e}")
        a = animal.lower()
        return [
            f"{animal} is way stranger than it looks.",
            f"Every new detail about {a} sounds made up.",
            f"{animal} breaks the rules of what an animal should be.",
        ]

def get_relevant_pexels_videos(animal: str) -> list[str]:
    """Return a list of Pexels video URLs visually relevant to the animal.

    Strategy:
    1. Exact animal queries.
    2. Semantic fallback map.
    3. Environment-based fallbacks.
    4. If all fail, raise RuntimeError.
    """
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY environment variable is not set")

    animal_lower = animal.lower()
    headers = {"Authorization": api_key}
    search_url = "https://api.pexels.com/videos/search"

    generic_tags = {
        "landscape",
        "forest",
        "river",
        "aerial",
        "timelapse",
        "mountains",
        "time lapse",
        "city",
        "people",
    }

    # Level 2: semantic fallbacks for specific animals
    semantic_fallbacks: dict[str, list[str]] = {
        "platypus": ["australian wildlife", "river wildlife"],
        "axolotl": ["aquarium amphibian", "underwater creature"],
        "mantis shrimp": ["reef creature", "ocean macro"],
        "sloth": ["rainforest mammal", "tree animal"],
    }

    # Level 3: environment-based keywords
    underwater_animals = {
        "axolotl",
        "leafy seadragon",
        "blue-ringed octopus",
        "immortal jellyfish",
        "goblin shark",
        "anglerfish",
        "frilled shark",
        "hagfish",
        "viperfish",
        "oarfish",
        "giant isopod",
        "pistol shrimp",
        "peacock mantis shrimp",
        "electric eel",
        "stonefish",
        "cone snail",
        "box jellyfish",
        "sea pig",
        "sea cucumber",
        "feather star",
        "sea angel",
        "sea butterfly",
        "narwhal",
        "beluga whale",
        "sperm whale",
        "orca",
        "dugong",
        "manatee",
        "blob sculpin",
        "mudskipper",
        "lungfish",
        "coelacanth",
        "arapaima",
        "pacu",
        "piranha",
        "electric catfish",
        "stargazer fish",
        "mola mola",
        "triggerfish",
        "parrotfish",
        "flying fish",
        "clownfish",
        "remora",
        "vampire squid",
        "giant squid",
        "colossal squid",
        "cuttlefish",
        "nautilus",
        "horseshoe crab",
        "king crab",
        "coconut crab",
        "giant spider crab",
        "lobster",
    }

    bird_animals = {
        "owl",
        "penguin",
        "cassowary",
        "shoebill",
        "kakapo",
        "hummingbird",
        "lyrebird",
        "bird-of-paradise",
        "hoatzin",
        "kookaburra",
        "horned screamer",
        "harpy eagle",
        "secretary bird",
        "marabou stork",
        "turkey vulture",
    }

    land_mammals = {
        "elephant",
        "cheetah",
        "koala",
        "kangaroo",
        "giraffe",
        "sloth",
        "maned wolf",
        "sun bear",
        "binturong",
        "okapi",
        "saiga antelope",
        "markhor",
        "jerboa",
        "springhare",
        "sugar glider",
        "slow loris",
        "tarsier",
        "armadillo",
        "echidna",
        "tenrec",
        "tapir",
        "wombat",
        "quokka",
    }

    def is_underwater() -> bool:
        return animal_lower in underwater_animals or any(tok in animal_lower for tok in ["shark", "fish", "squid", "jellyfish", "octopus"])

    def is_bird() -> bool:
        return animal_lower in bird_animals or any(tok in animal_lower for tok in ["bird", "eagle", "owl", "stork", "vulture"])

    def is_land_mammal() -> bool:
        return animal_lower in land_mammals or any(tok in animal_lower for tok in ["wolf", "bear", "monkey", "fox", "antelope", "bat"])

    def filter_relevant(videos: list[dict], required_token: str | None = None) -> list[str]:
        urls: list[str] = []
        for video in videos:
            video_id = video.get("id", "unknown")
            duration = float(video.get("duration") or 0.0)
            if duration < 1.5 or duration > 10.0:
                logger.info(
                    f"[PEXELS] Rejecting video {video_id} for '{animal}' due to duration {duration:.2f}s"
                )
                continue

            text_fields = [
                str(video.get("title", "")),
                str(video.get("description", "")),
                str(video.get("url", "")),
            ]
            tags = [str(t).lower() for t in video.get("tags", []) if isinstance(t, str)]
            combined_text = " ".join(text_fields + tags).lower()

            if any(tag in generic_tags for tag in tags):
                logger.info(
                    f"[PEXELS] Rejecting video {video_id} for '{animal}' due to generic tags {tags}"
                )
                continue

            if required_token and required_token not in combined_text:
                logger.info(
                    f"[PEXELS] Rejecting video {video_id} for '{animal}' because token '{required_token}' "
                    "is not present in metadata"
                )
                continue

            # For exact animal level, required_token is the animal name; for
            # fallbacks it's a same-family semantic keyword. Either way, we
            # require that at least one strong search word appears in metadata.
            if not required_token:
                if animal_lower not in combined_text:
                    logger.info(
                        f"[PEXELS] Rejecting video {video_id} for '{animal}' because animal name "
                        "is not present in metadata"
                    )
                    continue

            chosen_link: str | None = None
            for vf in video.get("video_files", []):
                link = vf.get("link")
                w = int(vf.get("width") or 0)
                h = int(vf.get("height") or 0)
                if not link or not w or not h:
                    continue
                if min(w, h) < 1080:
                    logger.info(
                        f"[PEXELS] Rejecting a file of video {video_id} for '{animal}' due to "
                        f"resolution {w}x{h} < 1080p shortest side"
                    )
                    continue
                chosen_link = link
                break

            if chosen_link:
                urls.append(chosen_link)
            else:
                logger.info(
                    f"[PEXELS] Rejecting video {video_id} for '{animal}' because no suitable video_files were found"
                )
        return urls

    # Level 1: exact animal queries
    logger.info(f"[PEXELS] Using primary queries for '{animal}'")
    primary_queries = [animal, f"{animal} close up", f"{animal} in nature"]
    primary_results: list[str] = []
    for query in primary_queries:
        params = {"query": query, "per_page": 20}
        resp = requests.get(search_url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        videos = data.get("videos", []) or []
        logger.info(
            f"[PEXELS] Retrieved {len(videos)} candidate videos for '{animal}' with query='{query}'"
        )
        urls = filter_relevant(videos, required_token=None)
        primary_results.extend(urls)

    if primary_results:
        return list(dict.fromkeys(primary_results))[:10]

    # Level 2: semantic fallback map
    fallback_queries = semantic_fallbacks.get(animal_lower, [])
    semantic_results: list[str] = []
    if fallback_queries:
        logger.info(f"[PEXELS] Using semantic fallback queries for '{animal}'")
        for query in fallback_queries:
            params = {"query": query, "per_page": 20}
            resp = requests.get(search_url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            videos = data.get("videos", []) or []
            logger.info(
                f"[PEXELS] Retrieved {len(videos)} candidate videos for '{animal}' with semantic query='{query}'"
            )
            # Require at least one strong word from the query to appear
            key_token = query.split()[0].lower()
            urls = filter_relevant(videos, required_token=key_token)
            semantic_results.extend(urls)

    if semantic_results:
        return list(dict.fromkeys(semantic_results))[:10]

    # Level 3: environment-based fallbacks
    env_results: list[str] = []
    if is_bird():
        logger.info(f"[PEXELS] Using bird environment fallback for '{animal}'")
        env_queries = ["bird slow motion", "exotic bird close up"]
        required = "bird"
    elif is_underwater():
        logger.info(f"[PEXELS] Using underwater (fish-family) fallback for '{animal}'")
        env_queries = ["weird fish close up", "deep sea fish"]
        required = "fish"
    elif is_land_mammal():
        logger.info(f"[PEXELS] Using land mammal environment fallback for '{animal}'")
        env_queries = ["wild mammal close up", "strange mammal"]
        required = "mammal"
    else:
        env_queries = []
        required = None

    for query in env_queries:
        params = {"query": query, "per_page": 20}
        resp = requests.get(search_url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        videos = data.get("videos", []) or []
        logger.info(
            f"[PEXELS] Retrieved {len(videos)} candidate videos for '{animal}' with environment query='{query}'"
        )
        urls = filter_relevant(videos, required_token=required)
        env_results.extend(urls)

    if env_results:
        return list(dict.fromkeys(env_results))[:10]

    raise RuntimeError(f"No visually relevant Pexels videos found for '{animal}' after all fallbacks")


def download_animal_video(animal: str, download_dir: str = "background_videos") -> list[str]:
    """Deprecated simple downloader removed in favor of stricter validator.

    This earlier implementation has been removed because a later definition
    of download_animal_video in this file performs full metadata, duration,
    and resolution checks and is the one actually used by the pipeline.
    """
    raise RuntimeError("Deprecated placeholder; the stricter implementation below is used.")

def generate_metadata(animal: str, script: str) -> dict:
    """Generate emotional, human metadata (title, description, hashtags).

    The metadata is tuned to match the conversational, shocked tone of the
    Shorts script and to avoid spammy clickbait or SEO stuffing.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    examples = (
        "This Animal Shouldn't Exist; "
        "Nature Went Too Far With This One; "
        "I Thought This Was Fake Until I Checked"
    )

    prompt = f"""You are writing metadata for a TikTok/YouTube Short about the {animal}.

SCRIPT (for context, do not repeat verbatim):
{script}

Write a single JSON object with exactly these keys:
  * It should feel like spoken language, like something a friend texts you.
  * No emoji, no ALL CAPS, no brackets, no numbers list, no clickbait spam.
  * It must hint that the {animal} is unreal or extreme, similar in tone to:
    {examples}
  * Sound human, not SEO-stuffed.
  * Briefly react to what makes the {animal} so insane or unexpected.
  * No hashtag list, no long keyword chains, no "Subscribe for daily facts" CTA.
  * Each item must be a single hashtag starting with '#'.
  * Mix of: the animal, shorts/reels context, and curiosity hooks.
  * Avoid generic spam like #fyp, #viral, #subscribe, #follow, #explore.

Return ONLY valid JSON. Do not add any explanation, backticks, or text outside the JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a social media strategist who writes natural, emotional "
                        "metadata for short-form videos. You never write SEO-stuffed or spammy text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=300,
        )

        content = (response.choices[0].message.content or "").strip()

        try:
            metadata = json.loads(content)
        except json.JSONDecodeError as e:
            # Fallback: build minimal but on-tone metadata if the model returns non-JSON
            logger.warning(
                f"Metadata response was not valid JSON, using fallback metadata: {e}"
            )

            # Simple spoken-style title fallback
            fallback_title = f"This {animal.title()} Shouldn't Be Real"

            # Short, human description
            fallback_desc = (
                f"I genuinely thought this {animal} was a joke until I "
                "looked it up. Now I can't unsee it."
            )

            animal_tag = "#" + animal.replace(" ", "")
            base_tags = [
                animal_tag,
                "#animalfacts",
                "#weirdnature",
                "#wildshorts",
                "#mindblowing",
            ]

            metadata = {
                "title": fallback_title,
                "description": fallback_desc,
                "hashtags": base_tags,
            }

        # Validate required keys
        required_keys = ["title", "description", "hashtags"]
        missing_keys = [key for key in required_keys if key not in metadata]
        if missing_keys:
            logger.warning(
                f"Metadata missing keys {missing_keys}; applying fallback defaults."
            )

            fallback_title = f"This {animal.title()} Shouldn't Be Real"
            fallback_desc = (
                f"I genuinely thought this {animal} was a joke until I "
                "looked it up. Now I can't unsee it."
            )
            animal_tag = "#" + animal.replace(" ", "")
            fallback_tags = [
                animal_tag,
                "#animalfacts",
                "#weirdnature",
                "#wildshorts",
                "#mindblowing",
            ]

            if "title" not in metadata:
                metadata["title"] = fallback_title
            if "description" not in metadata:
                metadata["description"] = fallback_desc
            if "hashtags" not in metadata:
                metadata["hashtags"] = fallback_tags

        # Normalize and clamp title
        title = str(metadata["title"]).strip()
        config = load_config()
        max_title_length = int(config.get("max_title_length", 60))
        if len(title) > max_title_length:
            title = title[: max_title_length - 3].rstrip() + "..."
        metadata["title"] = title

        # Normalize description
        description = str(metadata["description"]).strip()
        metadata["description"] = description

        # Ensure hashtags is a list of 5–8 strings and filter spammy ones
        raw_tags = metadata.get("hashtags", [])
        if not isinstance(raw_tags, list):
            raise ValueError("hashtags must be a list")

        spammy = {"#fyp", "#foryou", "#viral", "#subscribe", "#follow", "#explore"}
        cleaned_tags: list[str] = []
        for tag in raw_tags:
            if not isinstance(tag, str):
                continue
            tag = tag.strip()
            if not tag:
                continue
            if not tag.startswith("#"):
                tag = "#" + tag
            lower_tag = tag.lower()
            if lower_tag in spammy:
                continue
            if tag not in cleaned_tags:
                cleaned_tags.append(tag)

        # Guarantee 5–8 tags by adding sane defaults if needed
        if len(cleaned_tags) < 5:
            animal_tag = "#" + animal.replace(" ", "")
            defaults = [
                animal_tag,
                "#animalfacts",
                "#weirdnature",
                "#wildshorts",
                "#mindblowing",
                "#strangeanimals",
            ]
            for tag in defaults:
                if tag not in cleaned_tags:
                    cleaned_tags.append(tag)
                if len(cleaned_tags) >= 8:
                    break

        metadata["hashtags"] = cleaned_tags[:8]

        return metadata

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse metadata response as JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate metadata: {e}")


_BANNED_PHRASES = {
    "this sounds fake",
    "no way this is real",
    "i had to double-check",
    "my brain just stops",
    "wait...",
    "wait . . .",
    "wait ,",
    # Generic filler reactions we never want in scripts.
    "it breaks my brain",
    "exists in a way",
    "peace of mind just expired",
    "this is escalating",
    # Explicitly banned chaotic filler from spec.
    "my brain can't handle this",
    "this feels illegal",
    "i'm spiraling",
    "i am spiraling",
        # Abstract filler we must reject.
        "under the surface",
        "specific way to keep a body running",
        "design would not stay stable",
        "that mechanism",
        "narrow conditions",
        "for most animals",
        "that same constraint",
        "this only works because",
        "time pressure builds",
        "the safe window closes",
        "movement lasts only until",
        # Disallowed scaffolding phrases for scripts.
        "that limit means",
        "that cap forces",
        "to handle that",
        "so to keep going",
        "this allows",
        "which means",
        # Vague failure language.
        "breaks down",
        "loses its edge",
        "fails quickly",
    " system ",
    " mechanism ",
    " design ",
}

_LAST_SIGNATURE: str | None = None
USED_PUNCHLINES: set[str] = set()
_PREVIOUS_SCRIPTS: list[str] = []
MAX_PUNCHLINE_HISTORY = 250
_PUNCHLINE_HISTORY: list[str] = []

EMOTIONAL_WORDS = {
    "insane",
    "unbelievable",
    "horrifying",
    "terrifying",
    "disgusting",
    "weird",
    "cursed",
    "annoyed",
    "offended",
    "worse",
    "wild",
    "stunned",
    "shocked",
    "hate",
    "love",
    "wait",
    "no",
    "what",
    "how",
}



# Hook system for absurd, pattern-breaking openings

ABSURD_HOOK_POOL: dict[str, list[str]] = {
    "survival": [
        "If you hesitate once, you lose instantly.",
        "This mistake ends it immediately for you.",
        "Knowing this changes the outcome for you.",
        "One second too late and it is over.",
        "Do this wrong and you do not recover.",
    ],
    "humiliation": [
        "You would lose instantly in this situation.",
        "This animal humiliates confidence in seconds.",
        "You are not built for this.",
        "You would regret underestimating this out loud.",
        "Your pride does not survive this.",
    ],
    "authority": [
        "You were taught this wrong your whole life.",
        "No one warns you about this.",
        "This breaks the rule you trust.",
        "You assume safety where none exists.",
        "What you were told does not apply here.",
    ],
    "comparison": [
        "This animal outperforms you effortlessly every time.",
        "Your instincts fail here every time.",
        "You do not win this matchup.",
        "This is not a fair comparison.",
        "You lose the moment it decides.",
    ],
}

HOOK_CATEGORY_PRIORITY: list[str] = ["survival", "humiliation", "authority", "comparison"]

USED_HOOKS: set[str] = set()
RECENT_HOOKS: list[str] = []
RECENT_HOOK_CATEGORIES: list[str] = []
MAX_HOOK_CATEGORY_RUN = 2
_RECENT_HOOK_ARCHETYPES: list[str] = []
RECENT_OPENING_LINES: list[str] = []
_LAST_HOOK_ALIGNMENT_TAG: str | None = None
_RECENT_SCRIPT_STEMS: list[list[str]] = []
_RECENT_EXPLANATION_STYLES: list[str] = []

_LAST_HOOK_PREFIX: str | None = None
_LAST_HOOK_VERB: str | None = None
_LAST_HOOK_QUERY: str | None = None
_LAST_HOOK_REQUIRED_TOKEN: str | None = None

# Tag each hook with a category so we can avoid repeats.
HOOK_CATEGORIES: dict[str, str] = {
    hook: category
    for category, hooks in ABSURD_HOOK_POOL.items()
    for hook in hooks
}

BANNED_HOOK_WORDS: set[str] = {
    "heat",
    "oxygen",
    "energy",
    "system",
    "biology",
    "mechanism",
    "adaptation",
    "evolution",
    "movement",
    "speed",
    "sprint",
    "jump",
    "run",
}

# Retained for downstream query alignment when available.
HOOK_VERB_QUERY_MAP: dict[str, list[str]] = {}



CTA_OPTIONS: list[str] = [
    "Follow for more real animal facts that sound unreal.",
    "Follow for one wild but true animal fact at a time.",
    "Follow if you want more short facts like this.",
    "Follow for the next animal that makes no sense.",
]

_LAST_CTA_INDEX: int | None = None


def _prepare_tts_script(script: str) -> str:
    """Return script text exactly as spoken, with no control tokens.

    This layer must not inject any TTS directives (no prosody tags, no
    break markers, no percentages). All emotion is carried by the wording
    from `build_script`, not by inline instructions.
    """

    return script


def _contains_banned_phrase(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _BANNED_PHRASES)


def _normalize_punchline(text: str) -> str:
    # Lowercase and strip common punctuation so we can avoid
    # reusing essentially identical strong lines.
    import string

    t = text.lower()
    table = str.maketrans("", "", string.punctuation)
    t = t.translate(table)
    return " ".join(t.split())


def _register_punchline(text: str) -> bool:
    norm = _normalize_punchline(text)
    global _PUNCHLINE_HISTORY
    if norm in USED_PUNCHLINES:
        return False
    USED_PUNCHLINES.add(norm)
    _PUNCHLINE_HISTORY.append(norm)
    # Rolling window so we avoid repetition without starving future scripts.
    if len(_PUNCHLINE_HISTORY) > MAX_PUNCHLINE_HISTORY:
        evicted = _PUNCHLINE_HISTORY.pop(0)
        if evicted not in _PUNCHLINE_HISTORY:
            USED_PUNCHLINES.discard(evicted)
    return True


def _too_similar_to_previous(script: str) -> bool:
    import string

    words = [w.strip(string.punctuation).lower() for w in script.split() if w.strip()]
    if not words:
        return False
    this_set = set(words)

    # Only consider the most recent 50 scripts for similarity checks.
    recent = _PREVIOUS_SCRIPTS[-50:]

    for prev in recent:
        p_words = [w.strip(string.punctuation).lower() for w in prev.split() if w.strip()]
        if not p_words:
            continue
        p_set = set(p_words)
        if not this_set:
            continue
        overlap = len(this_set & p_set) / max(1, len(this_set))
        if overlap > 0.3:
            return True
    return False








def build_script(animal: str, facts: list[str]) -> str:
    """Build a natural short-form narration from fact lines.

    Goals:
    - Keep the voice human and clear.
    - Avoid brittle token-level consequence templates.
    - Preserve short-form pacing and anti-repetition behavior.
    """

    animal_low = animal.lower()
    animal_tokens = {tok for tok in re.split(r"[\s\-]+", animal_low) if tok}
    animal_tokens |= {f"{tok}s" for tok in animal_tokens}

    def _pick_cta(allow_repeat: bool = False) -> tuple[int, str]:
        global _LAST_CTA_INDEX
        indices = list(range(len(CTA_OPTIONS)))
        random.shuffle(indices)
        for idx in indices:
            if not allow_repeat and idx == _LAST_CTA_INDEX and len(indices) > 1:
                continue
            text = CTA_OPTIONS[idx]
            _LAST_CTA_INDEX = idx
            _register_punchline(text)
            return idx, text
        idx = random.randrange(len(CTA_OPTIONS))
        text = CTA_OPTIONS[idx]
        _LAST_CTA_INDEX = idx
        _register_punchline(text)
        return idx, text

    def _cleanup_sentence(text: str, max_words: int = 28) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = cleaned.replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = cleaned.replace("..", ".")
        cleaned = re.sub(r"([!?])\1+", r"\1", cleaned)
        cleaned = re.sub(r"\.{2,}", ".", cleaned)
        cleaned = re.sub(r",\.+", ".", cleaned)

        words = cleaned.split()
        if len(words) > max_words:
            words = words[:max_words]
            while words and words[-1].lower().strip(".,;:!?") in {
                "and",
                "or",
                "but",
                "because",
                "while",
                "which",
                "that",
                "to",
                "of",
                "in",
                "for",
                "with",
                "from",
                "up",
                "than",
            }:
                words.pop()
            cleaned = " ".join(words).rstrip(",;:")

        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    def _pluralize_animal_name(name: str) -> str:
        lower = name.lower().strip()
        if not lower:
            return lower
        parts = lower.split()
        last = parts[-1]
        invariant = {"fish", "squid", "deer", "sheep", "moose", "salmon", "trout"}
        if last in invariant:
            plural_last = last
        elif re.search(r"[^aeiou]y$", last):
            plural_last = last[:-1] + "ies"
        elif last.endswith(("s", "x", "z", "ch", "sh")):
            plural_last = last + "es"
        else:
            plural_last = last + "s"
        parts[-1] = plural_last
        return " ".join(parts)

    def _ensure_animal_subject(text: str) -> str:
        line = _cleanup_sentence(text)
        if not line:
            return ""
        low = line.lower()
        if any(tok in low for tok in animal_tokens):
            return line

        if low.startswith(("they ", "their ", "them ", "it ", "its ")):
            first, _, rest = line.partition(" ")
            plural_name = _pluralize_animal_name(animal_low)
            if first.lower() == "they":
                return f"{plural_name.capitalize()} {rest}"
            if first.lower() == "their":
                possessive = f"{plural_name}'" if plural_name.endswith("s") else f"{plural_name}'s"
                return f"{possessive.capitalize()} {rest}"
            if first.lower() == "them":
                return f"{plural_name.capitalize()} {rest}"
            if first.lower() == "it":
                return f"The {animal_low} {rest}"
            if first.lower() == "its":
                return f"The {animal_low}'s {rest}"
            return f"The {animal_low} {rest}"

        if line.startswith("Unlike "):
            return f"Unlike many animals, the {animal_low} {line[7:].lstrip()}"

        if line[:1].isupper():
            line = line[0].lower() + line[1:]
        return f"The {animal_low} {line}"

    def _pick_hook(topic: str, allow_repeat: bool = False) -> str:
        topic_hooks = {
            "speed": [
                f"The {animal_low} moves in ways that feel impossible.",
                f"The {animal_low} turns speed into a survival tool.",
            ],
            "danger": [
                f"The {animal_low} is built for situations most animals avoid.",
                f"The {animal_low} looks calm until the risk shows up.",
            ],
            "longevity": [
                f"The {animal_low} plays a much longer game than most animals.",
                f"The timeline of a {animal_low} is hard to wrap your head around.",
            ],
            "intelligence": [
                f"The {animal_low} solves problems in surprisingly smart ways.",
                f"The {animal_low} is far more strategic than people expect.",
            ],
            "scale": [
                f"The numbers behind the {animal_low} are hard to believe.",
                f"With the {animal_low}, the scale is what shocks people first.",
            ],
            "biology": [
                f"The {animal_low} sounds made up, but these facts are real.",
                f"The {animal_low} is stranger than it looks at first glance.",
            ],
        }
        options = topic_hooks.get(topic, topic_hooks["biology"]) + [
            f"Every detail about the {animal_low} changes the picture.",
            f"The {animal_low} looks familiar until you see the details.",
        ]
        random.shuffle(options)
        for hook in options:
            if not allow_repeat and hook in RECENT_OPENING_LINES:
                continue
            RECENT_OPENING_LINES.append(hook)
            if len(RECENT_OPENING_LINES) > 48:
                del RECENT_OPENING_LINES[0 : len(RECENT_OPENING_LINES) - 48]
            _register_punchline(hook)
            return hook
        return options[0]

    def _pick_bridge() -> str:
        return random.choice(
            [
                "Now add this to the picture.",
                "And there is another twist.",
                "The next fact makes this even wilder.",
                "Here is the part most people miss.",
            ]
        )

    def _pick_context_line(topic: str) -> str:
        options = {
            "speed": [
                f"Together, that shows how the {animal_low} trades precision for bursts of speed.",
                f"Put together, these details explain how the {animal_low} handles fast decisions.",
            ],
            "danger": [
                f"Taken together, this is why the {animal_low} can survive high-risk moments.",
                f"All of that points to a {animal_low} built to manage danger efficiently.",
            ],
            "longevity": [
                f"That combination helps the {animal_low} stay effective over a long lifespan.",
                f"Put together, those traits explain how the {animal_low} lasts so long.",
            ],
            "intelligence": [
                f"Taken together, these traits show how deliberate the {animal_low} can be.",
                f"That mix is why the {animal_low} adapts so well to changing conditions.",
            ],
            "scale": [
                f"Those details show how much the {animal_low}'s size shapes its daily life.",
                f"Put together, these facts explain the tradeoffs that come with that scale.",
            ],
            "biology": [
                f"Taken together, these facts show how specialized the {animal_low} really is.",
                f"All of that is why the {animal_low} works so differently from what people expect.",
            ],
        }
        return random.choice(options.get(topic, options["biology"]))

    def _sentence_stem(text: str) -> str:
        tokens = [t.strip(".,!?:;\"").lower() for t in text.split() if t.strip()]
        return " ".join(tokens[:4])

    def _script_self_check(lines: list[str]) -> bool:
        if len(lines) < 5 or len(lines) > 8:
            return False
        if any(not line.strip() for line in lines):
            return False
        if any(len(line.split()) > 30 for line in lines):
            return False
        if "?" in lines[0]:
            return False
        if not lines[-1].lower().startswith("follow"):
            return False
        joined = " ".join(lines).lower()
        if _contains_banned_phrase(joined):
            return False
        total_words = sum(len(l.split()) for l in lines)
        if total_words < 52 or total_words > 120:
            return False
        for idx in range(1, len(lines)):
            prev = (lines[idx - 1].split() or [""])[0].lower()
            curr = (lines[idx].split() or [""])[0].lower()
            if prev and curr and prev == curr:
                return False
        return True

    clean_facts = [_cleanup_sentence(f) for f in facts if str(f).strip()]
    clean_facts = [f for f in clean_facts if f]
    fallback = f"Scientists are still learning how the {animal_low} fits into its ecosystem."
    if not clean_facts:
        clean_facts = [fallback]
    deduped: list[str] = []
    for fact in clean_facts:
        if fact not in deduped:
            deduped.append(fact)
    while len(deduped) < 3:
        deduped.append(fallback)
    fact_lines = [_ensure_animal_subject(f) for f in deduped[:3]]

    logger.debug(f"Using facts for {animal}: {fact_lines}")

    global _LAST_SIGNATURE, _PREVIOUS_SCRIPTS, _RECENT_SCRIPT_STEMS
    global _LAST_HOOK_ALIGNMENT_TAG, _LAST_HOOK_QUERY, _LAST_HOOK_VERB, _LAST_HOOK_REQUIRED_TOKEN

    primary_topic = _classify_fact_topic(fact_lines[0])
    _LAST_HOOK_ALIGNMENT_TAG = primary_topic
    _LAST_HOOK_QUERY = None
    _LAST_HOOK_VERB = None
    _LAST_HOOK_REQUIRED_TOKEN = None

    attempts = 0
    while True:
        attempts += 1
        relax_uniqueness = attempts >= 8
        hook = _pick_hook(primary_topic, allow_repeat=relax_uniqueness)
        cta_index, cta = _pick_cta(allow_repeat=relax_uniqueness)

        lines = [
            hook,
            fact_lines[0],
            _pick_bridge(),
            fact_lines[1],
            fact_lines[2],
            _pick_context_line(primary_topic),
            cta,
        ]

        # Remove accidental duplicates while preserving order.
        deduped_lines: list[str] = []
        for line in lines:
            if line not in deduped_lines:
                deduped_lines.append(line)
        lines = deduped_lines

        script_text = "\n".join(lines)

        recent_stems = {stem for script in _RECENT_SCRIPT_STEMS for stem in script}
        current_stems = [_sentence_stem(line) for line in lines if line.strip()]
        if any(stem in recent_stems for stem in current_stems) and attempts < 8:
            continue
        if _too_similar_to_previous(script_text) and attempts < 8:
            continue
        if not _script_self_check(lines) and attempts < 8:
            continue

        signature = f"{primary_topic}|{cta_index}"
        if _LAST_SIGNATURE == signature and attempts < 8:
            continue

        _LAST_SIGNATURE = signature
        _PREVIOUS_SCRIPTS.append(script_text)
        if len(_PREVIOUS_SCRIPTS) > 100:
            del _PREVIOUS_SCRIPTS[0 : len(_PREVIOUS_SCRIPTS) - 100]

        _RECENT_SCRIPT_STEMS.append(current_stems)
        if len(_RECENT_SCRIPT_STEMS) > 10:
            del _RECENT_SCRIPT_STEMS[0 : len(_RECENT_SCRIPT_STEMS) - 10]
        return script_text


def _line_looks_like_fact(text: str) -> bool:
    """Heuristic used for both script checks and subtitles.

    A line is treated as a fact if it clearly describes
    something the animal *does* or *is*, rather than a pure
    emotional reaction.
    """

    low = text.lower()
    # Any explicit number usually signals a factual detail.
    if any(ch.isdigit() for ch in low):
        return True
    # Focus on strong capability/trait markers; avoid generic
    # "is/are" so short explanations are not misclassified.
    markers = [
        " can ",
        " has ",
        " have ",
        " cannot ",
        " can't ",
        " lives ",
        " live ",
        " weigh",
        " weighs",
        " grows",
        " measures",
    ]
    return any(m in low for m in markers)


def _line_looks_like_fact_or_explanation(text: str) -> bool:
    """Return True for core facts and their micro-explanations.

    Used by subtitle rendering so we show both the concrete
    facts and the short explanatory sentences that interpret
    them, while still skipping hooks, CTAs, and closing
    reaction lines.
    """

    if _line_looks_like_fact(text):
        return True

    low = text.lower()
    explanation_markers = [
        "for a ",
        "for an ",
        "for most animals",
        "in simple terms",
        "biologically,",
        "biologically ",
        "at this size",
        "speed like this",
        "that level of scale",
        "that kind of speed",
        "details like that",
        "longevity like that",
        "lifespan like that",
        "put together, those details",
        "taken as a set",
        "as a package",
        "when you add it up",
        "taken together, those traits",
        "overall, the biology behind",
        "put simply, that entire set of traits",
        "that one detail quietly rewrites",
    ]
    return any(marker in low for marker in explanation_markers)


def _script_has_tts_artifacts(text: str) -> bool:
    """Detect obvious TTS control tokens or SSML artifacts in script text."""
    low = text.lower()
    artifacts = [
        "<speak",
        "</speak>",
        "<break",
        "<prosody",
        "</prosody>",
        "ssml",
        "[pause",
        "[silence",
        "{pause",
        "{silence",
    ]
    return any(token in low for token in artifacts)


def _classify_fact_topic(text: str) -> str:
    """Classify a fact line into a coarse topic bucket.

    Used to lightly align hooks and explanations with the
    kind of capability the facts are about, without ever
    looking at the specific animal name.
    """

    low = text.lower()
    if any(w in low for w in ["km/h", "kph", "mph", "fast", "faster", "speed"]):
        return "speed"
    if any(w in low for w in ["venom", "venomous", "toxic", "poison", "sting", "bite", "predator", "dangerous"]):
        return "danger"
    if any(w in low for w in ["years", "decades", "centuries", "lifespan", "lives up to", "age"]):
        return "longevity"
    if any(w in low for w in ["brain", "smart", "intelligent", "tool", "problem", "solve", "memory"]):
        return "intelligence"
    if any(w in low for w in ["meters", "meter", "feet", "foot", "inches", "pounds", "kilos", "kilograms", "tons", "huge", "giant", "tiny", "massive"]):
        return "scale"
    return "biology"


def generate_voiceover(script: str, output_file: str, speech_speed: float | None = None) -> str:
    if speech_speed is None:
        # Slightly faster default delivery for more energy.
        speech_speed = random.uniform(1.07, 1.20)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)

    # Prepare script text exactly as spoken, no control tokens.
    tts_input = _prepare_tts_script(script)

    try:
        # Rotate between a small, predefined set of voices so that
        # different videos feel like different narrators while each
        # individual video keeps a single consistent voice.
        global _LAST_TTS_VOICE
        choices = TTS_VOICES[:]
        if _LAST_TTS_VOICE in choices and len(choices) > 1:
            choices = [v for v in choices if v != _LAST_TTS_VOICE]
        chosen_voice = random.choice(choices)
        _LAST_TTS_VOICE = chosen_voice

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


def generate_subtitles(audio_path: str, output_srt: str = "subtitles.srt") -> str:
    """
    Transcribe audio to SRT subtitle format using OpenAI Whisper.
    
    Args:
        audio_path: Path to the audio file to transcribe.
        output_srt: Output SRT file path (default: "subtitles.srt").
    
    Returns:
        The output SRT file path.
    
    Raises:
        RuntimeError: If the transcription fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="srt",
            )
        
        # Write the SRT content to file
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write(response)
        
        return output_srt
    
    except FileNotFoundError:
        raise RuntimeError(f"Audio file not found: {audio_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate subtitles: {e}")


def _wrap_subtitle_sentence(text: str, max_words_per_line: int = 6) -> list[str]:
    """Wrap a spoken sentence into short visual lines for mobile.

    Splits only on word boundaries and prefers natural pauses such as
    commas or connectors ("and", "but", "because", "which"). Content is
    never dropped; all words are preserved across the returned lines.
    """

    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current: list[str] = []
    last_break_at: int | None = None

    def flush_line(until: int | None = None) -> None:
        nonlocal current, last_break_at
        if not current:
            return
        if until is None or until <= 0 or until > len(current):
            line_words = current
            rest: list[str] = []
        else:
            line_words = current[:until]
            rest = current[until:]
        lines.append(" ".join(line_words))
        current = rest
        last_break_at = None

    for w in words:
        current.append(w)
        core = w.strip(",.!?…").lower()
        if w.endswith(",") or core in {"and", "but", "because", "which"}:
            last_break_at = len(current)

        if len(current) >= max_words_per_line:
            if last_break_at is not None:
                flush_line(last_break_at)
            else:
                flush_line(None)

    if current:
        flush_line(None)

    return lines


def download_animal_image(animal: str, download_dir: str = "backgrounds") -> str:
    """
    Download a photo of the given animal from Pexels.
    
    Args:
        animal: The name of the animal to search for.
        download_dir: The directory to save the image (default: "backgrounds").
    
    Returns:
        The local file path to the downloaded image.
    
    Raises:
        RuntimeError: If the API call or download fails.
    """
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY environment variable is not set")
    
    try:
        # Search for photos
        headers = {"Authorization": api_key}
        search_url = "https://api.pexels.com/v1/search"
        params = {"query": animal, "per_page": 15}
        
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        photos = data.get("photos", [])
        
        if not photos:
            raise RuntimeError(f"No images found for '{animal}'")
        
        # Pick a random photo from results
        photo = random.choice(photos)
        image_url = photo["src"]["large"]
        photo_id = photo["id"]
        
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        # Save the image
        file_path = os.path.join(download_dir, f"{animal}_{photo_id}.jpg")
        with open(file_path, "wb") as f:
            f.write(image_response.content)
        
        return file_path
    
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download image: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to download animal image: {e}")


def download_animal_image_with_scale(
    animal: str,
    download_dir: str = "backgrounds",
    max_attempts: int = 6,
) -> str:
    """Download an animal image that satisfies subject scale constraints."""
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY environment variable is not set")

    headers = {"Authorization": api_key}
    search_url = "https://api.pexels.com/v1/search"
    params = {"query": animal, "per_page": 20}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    photos = data.get("photos", [])
    if not photos:
        raise RuntimeError(f"No images found for '{animal}'")

    os.makedirs(download_dir, exist_ok=True)

    attempts = 0
    random.shuffle(photos)
    best_path: str | None = None
    best_scale = 1.0
    for photo in photos:
        if attempts >= max_attempts:
            break
        attempts += 1
        image_url = photo["src"]["large"]
        photo_id = photo["id"]
        file_path = os.path.join(download_dir, f"{animal}_{photo_id}.jpg")
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(image_response.content)
            subject_scale = _estimate_image_subject_fill_ratio(file_path)
            if subject_scale <= 0.40:
                logger.info(
                    "[FRAMING] image=%s subject_scale=%.3f accepted",
                    file_path,
                    subject_scale,
                )
                return file_path
            logger.info(
                "[FRAMING] image=%s subject_scale=%.3f rejected",
                file_path,
                subject_scale,
            )
            if subject_scale < best_scale:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_path = file_path
                best_scale = subject_scale
            else:
                os.remove(file_path)
        except requests.RequestException as e:
            logger.warning(f"Failed to download candidate image: {e}")
        except OSError:
            pass

    if best_path:
        logger.warning(
            "[FRAMING] using least-zoomed image=%s subject_scale=%.3f",
            best_path,
            best_scale,
        )
        return best_path

    raise RuntimeError(
        f"No suitable images found for '{animal}' under subject scale constraints"
    )


def _is_too_zoomed(video_path: str, max_fill_ratio: float = 0.45) -> bool:
    """
    Reject videos where the foreground (animal) fills too much of the frame.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return True  # reject unreadable video

    filled_ratio = _estimate_subject_fill_ratio_from_frame(frame)

    return filled_ratio > max_fill_ratio


def download_animal_video(animal: str, download_dir: str = "background_videos") -> list[str]:
    """Download multiple validated Pexels clips for the given animal.
    
    Applies strict validation on metadata, duration, resolution, and tags.
    Returns paths to one or more local MP4 clips or an empty list if
    nothing suitable is found even after fallbacks.
    """
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY environment variable is not set")

    animal_lower = animal.lower()
    generic_tags = {"landscape", "forest", "river", "aerial", "timelapse", "mountains"}

    # Build strong search queries
    base_queries: list[str] = [
        f"{animal} animal",
        f"{animal} wildlife",
        f"{animal} close up",
        f"{animal} in nature",
    ]
    water_animals = {"shark", "dolphin", "whale", "orca", "seal", "penguin", "octopus", "fish"}
    bird_animals = {
        "owl",
        "penguin",
        "cassowary",
        "shoebill",
        "kakapo",
        "hummingbird",
        "lyrebird",
        "bird-of-paradise",
        "hoatzin",
        "kookaburra",
        "horned screamer",
        "harpy eagle",
        "secretary bird",
        "marabou stork",
        "turkey vulture",
    }
    if any(word in animal_lower for word in water_animals):
        base_queries.append(f"{animal} swimming")

    hook_queries: list[str] = []
    if _LAST_HOOK_QUERY:
        hook_queries.append(_LAST_HOOK_QUERY)
    if _LAST_HOOK_VERB and _LAST_HOOK_VERB in HOOK_VERB_QUERY_MAP:
        hook_queries.extend([q.format(animal=animal) for q in HOOK_VERB_QUERY_MAP[_LAST_HOOK_VERB]])
    hook_queries = list(dict.fromkeys(hook_queries))
    if hook_queries:
        base_queries = hook_queries + [q for q in base_queries if q not in hook_queries]

    headers = {"Authorization": api_key}
    search_url = "https://api.pexels.com/videos/search"
    os.makedirs(download_dir, exist_ok=True)

    valid_paths: list[str] = []
    used_queries: set[str] = set()

    def name_matches(combined_text: str) -> bool:
        """Return True if the Pexels metadata clearly refers to this animal.

        We first look for the full animal name (e.g. "box jellyfish"). If that
        is missing, we fall back to any strong, non-generic token from the
        animal name (e.g. "jellyfish"), so that cases where Pexels only tags a
        clip as "jellyfish" still pass for "box jellyfish".
        """
        text = combined_text.lower()
        if animal_lower in text:
            return True

        tokens = [t for t in animal_lower.replace("-", " ").split() if t]
        generic_name_tokens = {
            "animal",
            "wildlife",
            "nature",
            "common",
            "giant",
            "little",
            "small",
            "big",
            "sea",
            "river",
            "forest",
        }
        strong_tokens = [t for t in tokens if len(t) >= 4 and t not in generic_name_tokens]
        if not strong_tokens:
            return False
        return any(tok in text for tok in strong_tokens)

    def validate_and_download(videos: list[dict], query: str) -> None:
        nonlocal valid_paths
        for video in videos:
            vid_id = video.get("id")
            duration = float(video.get("duration") or 0.0)
            text_fields = [
                str(video.get("title", "")),
                str(video.get("description", "")),
                str(video.get("url", "")),
            ]
            tags = [str(t).lower() for t in video.get("tags", []) if isinstance(t, str)]

            # Reject by metadata
            combined_text = " ".join(text_fields + tags)
            if not name_matches(combined_text):
                logger.info(f"[PEXELS] Rejecting video {vid_id} (query='{query}') reason=no-animal-match")
                continue

            if duration < 1.5 or duration > 10.0:
                logger.info(
                    f"[PEXELS] Rejecting video {vid_id} (query='{query}') reason=duration {duration:.2f}s"
                )
                continue

            if any(tag in generic_tags for tag in tags):
                logger.info(f"[PEXELS] Rejecting video {vid_id} (query='{query}') reason=generic-tags")
                continue

            # Find a file with sufficient resolution
            chosen_link: str | None = None
            chosen_w = 0
            chosen_h = 0
            for vf in video.get("video_files", []):
                link = vf.get("link")
                w = int(vf.get("width") or 0)
                h = int(vf.get("height") or 0)
                if not link or not w or not h:
                    continue
                if min(w, h) < 1080:
                    continue
                chosen_link = link
                chosen_w = w
                chosen_h = h
                break

            if not chosen_link:
                logger.info(f"[PEXELS] Rejecting video {vid_id} (query='{query}') reason=resolution-too-small")
                continue

            try:
                video_response = requests.get(chosen_link)
                video_response.raise_for_status()
                file_path = os.path.join(download_dir, f"{animal}_{vid_id}.mp4")
                with open(file_path, "wb") as f:
                    f.write(video_response.content)
                if _is_too_zoomed(file_path):
                    logger.info(f"[PEXELS] Rejecting video {vid_id} reason=too-zoomed")
                    os.remove(file_path)
                    continue
                subject_scale = _estimate_subject_fill_ratio_video(file_path)
                edge_density = _edge_density_video(file_path)
                valid_paths.append(file_path)
                logger.info(
                    f"[PEXELS] Accepted video {vid_id} (query='{query}') duration={duration:.2f}s "
                    f"size={chosen_w}x{chosen_h} subject_scale={subject_scale:.3f} edge_density={edge_density:.3f} -> {file_path}"
                )
            except requests.RequestException as e:
                logger.warning(f"[PEXELS] Failed to download video {vid_id} (query='{query}'): {e}")

            if len(valid_paths) >= 10:
                return

    try:
        attempts = 0
        # First, try to find at least one good clip for this exact animal
        while not valid_paths and attempts < len(base_queries):
            # Pick an unused query, prioritizing hook-aligned queries first
            remaining = [q for q in base_queries if q not in used_queries]
            if not remaining:
                break
            hook_remaining = [q for q in hook_queries if q not in used_queries]
            if hook_remaining:
                query = hook_remaining[0]
            else:
                query = random.choice(remaining)
            used_queries.add(query)
            attempts += 1
            logger.info(
                f"[PEXELS] Searching videos for '{animal}' with query='{query}' (attempt {attempts})"
            )
            params = {"query": query, "per_page": 20}
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            videos = data.get("videos", []) or []
            logger.info(
                f"[PEXELS] Retrieved {len(videos)} candidate videos for '{animal}' with query='{query}'"
            )
            validate_and_download(videos, query)
            logger.info(
                f"[PEXELS] After query='{query}', accepted={len(valid_paths)} clips for '{animal}'"
            )

        if not valid_paths:
            # Fallback: try a more generic, visually similar category so we
            # still get a reasonable background (e.g., any bird for a rare bird).
            fallback_query: str | None = None
            generic_token: str | None = None

            if animal_lower in bird_animals or any(word in animal_lower for word in ["bird", "eagle", "owl"]):
                fallback_query = "exotic bird"
                generic_token = "bird"
            elif any(word in animal_lower for word in water_animals):
                fallback_query = "ocean animal"
                generic_token = "ocean"

            if not fallback_query or not generic_token:
                # Last-resort generic fallback for any animal
                fallback_query = "weird animal"
                generic_token = "animal"

            if fallback_query and generic_token:
                try:
                    logger.info(
                        f"[PEXELS] Falling back to generic query '{fallback_query}' for '{animal}'"
                    )
                    params = {"query": fallback_query, "per_page": 20}
                    response = requests.get(search_url, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()
                    videos = data.get("videos", []) or []

                    for video in videos:
                        vid_id = video.get("id")
                        duration = float(video.get("duration") or 0.0)
                        text_fields = [
                            str(video.get("title", "")),
                            str(video.get("description", "")),
                            str(video.get("url", "")),
                        ]
                        tags = [str(t).lower() for t in video.get("tags", []) if isinstance(t, str)]
                        combined_text = " ".join(text_fields + tags).lower()

                        if generic_token not in combined_text:
                            continue
                        if duration < 1.5 or duration > 10.0:
                            continue

                        chosen_link: str | None = None
                        chosen_w = 0
                        chosen_h = 0
                        for vf in video.get("video_files", []):
                            link = vf.get("link")
                            w = int(vf.get("width") or 0)
                            h = int(vf.get("height") or 0)
                            if not link or not w or not h:
                                continue
                            if min(w, h) < 1080:
                                continue
                            chosen_link = link
                            chosen_w = w
                            chosen_h = h
                            break

                        if not chosen_link:
                            continue

                        try:
                            video_response = requests.get(chosen_link)
                            video_response.raise_for_status()
                            file_path = os.path.join(download_dir, f"{animal}_fallback_{vid_id}.mp4")
                            with open(file_path, "wb") as f:
                                f.write(video_response.content)
                            if _is_too_zoomed(file_path):
                                logger.info(f"[PEXELS] Rejecting video {vid_id} reason=too-zoomed")
                                os.remove(file_path)
                                continue
                            subject_scale = _estimate_subject_fill_ratio_video(file_path)
                            edge_density = _edge_density_video(file_path)
                            valid_paths.append(file_path)
                            logger.info(
                                f"[PEXELS] Accepted FALLBACK video {vid_id} for '{animal}' "
                                f"query='{fallback_query}' duration={duration:.2f}s size={chosen_w}x{chosen_h} "
                                f"subject_scale={subject_scale:.3f} edge_density={edge_density:.3f} -> {file_path}"
                            )
                            break
                        except requests.RequestException:
                            continue
                except requests.RequestException as e:
                    logger.warning(
                        f"[PEXELS] Request error during fallback download for '{animal}': {e}"
                    )

        if not valid_paths:
            logger.warning(
                f"[PEXELS] Not enough valid clips for '{animal}' (found {len(valid_paths)}). Skipping animal."
            )
            return []

        # Cap to at most 10 clips
        return valid_paths[:10]

    except requests.RequestException as e:
        logger.warning(f"[PEXELS] Request error while downloading videos for '{animal}': {e}")
        return []
    except Exception as e:
        logger.warning(f"[PEXELS] Unexpected error downloading animal videos for '{animal}': {e}")
        return []


def create_video_from_image_and_audio(
    image_path: str,
    audio_path: str,
    output_path: str = "animal_short.mp4"
) -> str:
    """
    Create a vertical video from an image and audio file using ffmpeg.
    
    Args:
        image_path: Path to the background image.
        audio_path: Path to the audio file.
        output_path: Output video file path (default: "animal_short.mp4").
    
    Returns:
        The output video file path.
    
    Raises:
        RuntimeError: If ffmpeg fails or is not installed.
    """
    subject_scale = _estimate_image_subject_fill_ratio(image_path)
    zoom_max = 1.06
    if subject_scale >= 0.35:
        zoom_max = 1.0

    scale_factor = 1.0
    if subject_scale > 0.40:
        scale_factor = min(1.0, (0.40 / subject_scale) / max(zoom_max, 1.0))
        logger.warning(
            "[FRAMING] image subject_scale=%.3f exceeds 0.40, applying scale_factor=%.3f",
            subject_scale,
            scale_factor,
        )

    # Static framing on a true 9:16 frame.
    # 1) Crop to 9:16 using input height
    # 2) Then scale to 1080x1920 without zoom.
    video_filter = (
        "crop=ih*9/16:ih,"
        "scale=1080:1920,"
        f"scale=1080*{scale_factor:.3f}:1920*{scale_factor:.3f}:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
        "fps=30"
    )

    logger.info(
        "[FRAMING] image=%s subject_scale=%.3f zoom_max=%.2f scale_factor=%.3f",
        image_path,
        subject_scale,
        zoom_max,
        scale_factor,
    )
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if exists
        "-loop", "1",  # Loop the image
        "-i", image_path,  # Input image
        "-i", audio_path,  # Input audio
        "-vf", video_filter,  # Video filter for Ken Burns effect
        "-c:v", "libx264",  # H.264 video codec
        "-tune", "stillimage",  # Optimize for still image input
        "-c:a", "aac",  # AAC audio codec
        "-b:a", "192k",  # Audio bitrate
        "-pix_fmt", "yuv420p",  # Pixel format for compatibility
        "-shortest",  # Stop when the shortest input ends (audio)
        output_path
    ]
    
    try:
        logger.info(f"Running ffmpeg to create video: {output_path}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return output_path
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not found in PATH")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}")


def _adaptive_vertical_filter(dampen: float = 1.0, y_bias: float = 0.0) -> str:
    """
    Dynamically scale and pad to 9:16 without over-zooming.
    """
    safe_dampen = max(0.85, min(dampen, 1.0))
    safe_bias = max(-0.08, min(y_bias, 0.08))
    return (
        f"scale=1080*{safe_dampen:.3f}:1920*{safe_dampen:.3f}:force_original_aspect_ratio=decrease,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2+{safe_bias:+.3f}*oh"
    )


def _estimate_subject_fill_ratio_from_frame(frame: "cv2.Mat") -> float:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    row_counts = (edges > 0).sum(axis=1)
    if row_counts.max() == 0:
        return 1.0

    active_rows = [idx for idx, cnt in enumerate(row_counts) if cnt > (0.02 * w)]
    if not active_rows:
        return 1.0

    min_row = min(active_rows)
    max_row = max(active_rows)
    span = max_row - min_row + 1
    return min(1.0, span / max(1, h))


def _estimate_subject_fill_ratio_video(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return 1.0
    return _estimate_subject_fill_ratio_from_frame(frame)


def _estimate_image_subject_fill_ratio(image_path: str) -> float:
    frame = cv2.imread(image_path)
    if frame is None:
        return 1.0
    return _estimate_subject_fill_ratio_from_frame(frame)


def _edge_density_video(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return cv2.countNonZero(edges) / max(1, (edges.shape[0] * edges.shape[1]))


def _infer_context_from_path(video_path: str) -> str:
    name = os.path.basename(video_path).lower()
    if any(tok in name for tok in ["sea", "ocean", "reef", "fish", "shark", "whale", "dolphin", "octopus", "jellyfish", "squid", "eel"]):
        return "underwater"
    if any(tok in name for tok in ["bird", "wing", "flight", "flying", "sky"]):
        return "air"
    return "ground"


def _vertical_bias_from_context(video_path: str) -> float:
    context = _infer_context_from_path(video_path)
    if context == "underwater" or context == "air":
        return 0.06
    return -0.04


def _compute_crop_dampener(subject_scale: float, edge_density: float, video_path: str) -> float:
    name = os.path.basename(video_path).lower()
    keyword_hit = any(tok in name for tok in ["shell", "carapace", "armor", "armour", "giant", "tortoise", "turtle", "whale", "elephant", "reptile", "croc", "alligator"])
    if subject_scale >= 0.35 or edge_density < 0.02 or keyword_hit:
        return 0.94
    return 1.0


def _compute_zoom_range(subject_scale: float, edge_density: float, video_path: str) -> tuple[float, float]:
    name = os.path.basename(video_path).lower()
    keyword_hit = any(tok in name for tok in ["shell", "carapace", "armor", "armour", "giant", "tortoise", "turtle", "whale", "elephant", "reptile", "croc", "alligator"])
    if subject_scale >= 0.35 or edge_density < 0.02 or keyword_hit:
        return (1.0, 1.03)
    return (1.02, 1.08)


def create_video_from_clip_and_audio(
    video_path: str,
    audio_path: str,
    output_path: str = "animal_short.mp4"
) -> str:
    """Create a vertical Short from an existing video clip and audio.
    
    Center-crops the clip into a true 9:16 frame and combines it with
    the narration audio, without any padding or black bars.
    """
    subject_scale = _estimate_subject_fill_ratio_video(video_path)
    edge_density = _edge_density_video(video_path)
    if subject_scale > 0.40:
        logger.error(
            "Clip subject scale too large (%.2f > 0.40) for %s",
            subject_scale,
            video_path,
        )
        raise RuntimeError("Clip subject scale exceeds 40% of frame height")

    dampener = _compute_crop_dampener(subject_scale, edge_density, video_path)
    y_bias = _vertical_bias_from_context(video_path)
    logger.info(
        "[FRAMING] clip=%s subject_scale=%.3f edge_density=%.3f dampen=%.2f y_bias=%.3f zoom=1.00 crop=adaptive",
        video_path,
        subject_scale,
        edge_density,
        dampener,
        y_bias,
    )

    # 1) Scale/pad to 9:16 with dampening + bias (no zoom)
    filter_complex = (
        f"[0:v]{_adaptive_vertical_filter(dampen=dampener, y_bias=y_bias)},setsar=1[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop", "-1",  # loop clip so video is never shorter than audio
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",  # cut both when narration ends
        output_path,
    ]
    try:
        logger.info(f"Running ffmpeg to create video from clip: {output_path}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not found in PATH")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}")


def _probe_media_duration(path: str) -> float | None:
    """Return media duration in seconds using ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def create_video_from_clips_and_audio(
    clips: list[str],
    audio_path: str,
    output_path: str = "animal_short.mp4",
) -> str:
    """Create a vertical Short from multiple clips and narration audio."""
    if not clips:
        raise RuntimeError("No clips provided to create_video_from_clips_and_audio")


    audio_duration = _probe_media_duration(audio_path) or 15.0
    out_dir = os.path.dirname(output_path) or "."
    out_base = os.path.basename(output_path)
    segment_paths: list[str] = []
    total_est = 0.0

    # Randomize clip order for every run to avoid repetition
    clips = clips[:]
    random.shuffle(clips)

    clip_stats: dict[str, dict] = {}
    for clip in clips:
        subject_scale = _estimate_subject_fill_ratio_video(clip)
        edge_density = _edge_density_video(clip)
        if subject_scale > 0.40:
            logger.error(
                "Clip subject scale too large (%.2f > 0.40) for %s",
                subject_scale,
                clip,
            )
            raise RuntimeError("Clip subject scale exceeds 40% of frame height")
        dampener = _compute_crop_dampener(subject_scale, edge_density, clip)
        zoom_min, zoom_max = _compute_zoom_range(subject_scale, edge_density, clip)
        y_bias = _vertical_bias_from_context(clip)
        clip_stats[clip] = {
            "subject_scale": subject_scale,
            "edge_density": edge_density,
            "dampener": dampener,
            "zoom_min": zoom_min,
            "zoom_max": zoom_max,
            "y_bias": y_bias,
        }
        logger.info(
            "[FRAMING] clip=%s subject_scale=%.3f edge_density=%.3f dampen=%.2f zoom_range=%.2f-%.2f y_bias=%.3f crop=adaptive",
            clip,
            subject_scale,
            edge_density,
            dampener,
            zoom_min,
            zoom_max,
            y_bias,
        )

    seg_index = 0
    clip_index = 0
    max_segments = 40
    while total_est < audio_duration and seg_index < max_segments:
        clip_path = clips[clip_index % len(clips)]
        clip_index += 1
        clip_duration = _probe_media_duration(clip_path) or 3.0
        if clip_duration <= 0.3:
            logger.info(f"Skipping clip '{clip_path}' due to invalid duration {clip_duration:.2f}s")
            continue
        # Choose a random segment length between ~1.8–2.6 seconds
        raw_len = random.uniform(1.8, 2.6)
        segment_len = min(raw_len, clip_duration)
        # Ensure we stay within the desired range for strong pacing
        if segment_len < 1.8:
            logger.info(
                f"Skipping clip '{clip_path}' because usable duration {segment_len:.2f}s < 1.8s"
            )
            continue

        start = 0.0
        if clip_duration > segment_len + 0.5:
            max_start = clip_duration - segment_len - 0.25
            if max_start > 0:
                start = random.uniform(0, max_start)

        stats = clip_stats.get(clip_path, {})
        zoom_min = stats.get("zoom_min", 1.02)
        zoom_max = stats.get("zoom_max", 1.08)
        y_bias = stats.get("y_bias", 0.0)
        dampener = stats.get("dampener", 1.0)
        zoom = round(random.uniform(zoom_min, zoom_max), 3)

        # 1) Crop to 9:16 using input height
        # 2) Then scale to 1080x1920 with a gentle, center-based zoom;
        # horizontal and vertical pan are effectively kept centered.
        vf = (
            f"{_adaptive_vertical_filter(dampen=dampener, y_bias=y_bias)},"
            "fps=30"
        )

        seg_path = os.path.join(out_dir, f"{out_base}.seg{seg_index}.mp4")
        seg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(segment_len),
            "-i",
            clip_path,
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset","veryfast",
            seg_path,
        ]
        try:
            subprocess.run(seg_cmd, capture_output=True, text=True, check=True)
            segment_paths.append(seg_path)
            total_est += segment_len
            seg_index += 1
            logger.info(
                "Created segment %s from '%s' start=%.2fs len=%.2fs crop=adaptive dampen=%.2f y_bias=%.3f",
                seg_path,
                clip_path,
                start,
                segment_len,
                dampener,
                y_bias,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"ffmpeg failed while creating segment from '{clip_path}': {e.stderr}")

    if not segment_paths:
        raise RuntimeError("Failed to create any segments from clips for video assembly")

    # Build concat list file in the same directory as the output video
    concat_list = os.path.join(out_dir, f"{out_base}.concat.txt")
    try:
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg in segment_paths:
                # ffmpeg concat demuxer uses paths relative to the list file
                f.write(f"file '{os.path.basename(seg)}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            output_path,
        ]
        logger.info(
            f"Running ffmpeg to create video from {len(segment_paths)} clips into: {output_path}"
        )
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not found in PATH")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed during clip assembly: {e.stderr}")
    finally:
        # Cleanup temp files
        for seg in segment_paths:
            try:
                if os.path.exists(seg):
                    os.remove(seg)
            except OSError:
                pass
        try:
            if os.path.exists(concat_list):
                os.remove(concat_list)
        except OSError:
            pass


def burn_subtitles(
    input_video: str,
    subtitle_file: str,
    output_video: str = "animal_short_final.mp4"
) -> str:
    """Burn subtitles directly into the video frames (hard subtitles).

    This avoids relying on ffmpeg subtitle filters and uses
    `_wrap_subtitle_sentence` to keep each visual line short and
    phone-readable while preserving the full spoken content.

    Args:
        input_video: Path to the input video file.
        subtitle_file: Path to the SRT subtitle file.
        output_video: Output video file path (default: "animal_short_final.mp4").

    Returns:
        The output video file path with burned-in subtitles.

    Raises:
        RuntimeError: If processing fails.
    """

    # Parse SRT into timed cues (in seconds)
    try:
        with open(subtitle_file, "r", encoding="utf-8") as f:
            srt_text = f.read().replace("\r\n", "\n").replace("\r", "\n").strip()
    except OSError as e:
        raise RuntimeError(f"Failed to read subtitles file: {e}")

    if not srt_text:
        logger.warning("Subtitle file is empty; copying video without subtitles.")
        temp_output = output_video
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-c:v", "copy",
            "-c:a", "copy",
            temp_output,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return temp_output
        except FileNotFoundError:
            raise RuntimeError("ffmpeg is not installed or not found in PATH")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed while copying video: {e.stderr}")

    def _srt_time_to_seconds(t: str) -> float:
        hms, ms = t.split(",")
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    cues: list[tuple[float, float, list[str]]] = []
    for block in srt_text.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line = lines[1]
        if "-->" not in time_line:
            continue
        start_str, end_str = [p.strip() for p in time_line.split("-->")]
        start = _srt_time_to_seconds(start_str)
        end = _srt_time_to_seconds(end_str)
        if len(lines) < 3:
            continue
        text = " ".join(lines[2:])
        if not text:
            continue
        # Subtitle concrete facts and their short explanations,
        # but skip pure reactions, hooks, and CTA-only lines.
        if not _line_looks_like_fact_or_explanation(text):
            continue
        wrapped = _wrap_subtitle_sentence(text)
        if not wrapped:
            continue
        cues.append((start, end, wrapped))

    if not cues:
        logger.warning("No valid cues parsed from subtitles; copying video without subtitles.")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-c:v", "copy",
            "-c:a", "copy",
            output_video,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_video
        except FileNotFoundError:
            raise RuntimeError("ffmpeg is not installed or not found in PATH")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed while copying video: {e.stderr}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video = output_video + ".nosound.mp4"
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter for subtitle rendering")

    # Precompute cue frame ranges
    cue_frames: list[tuple[int, int, list[str]]] = []
    for start, end, lines in cues:
        start_f = int(start * fps)
        end_f = int(end * fps)
        cue_frames.append((start_f, end_f, lines))

    frame_idx = 0
    cue_index = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Advance cue index if current cue has passed
        while cue_index < len(cue_frames) and frame_idx > cue_frames[cue_index][1]:
            cue_index += 1

        lines_to_draw: list[str] | None = None
        if cue_index < len(cue_frames):
            start_f, end_f, lines = cue_frames[cue_index]
            if start_f <= frame_idx <= end_f:
                lines_to_draw = lines

        if lines_to_draw:
            # Huge, high-contrast white text centered in the frame.
            # Ensure subtitles never overlap the bottom 20% of the frame.
            font_scale = 1.9
            thickness = 5
            line_height = 80
            total_height = line_height * len(lines_to_draw)
            center_y = height // 2
            max_text_bottom = int(height * 0.8)
            # Start from vertical center, but clamp so the bottom of the block
            # never drops into the bottom 20% of the frame.
            base_y = center_y - total_height // 2
            if base_y + total_height > max_text_bottom:
                base_y = max(0, max_text_bottom - total_height)

            for i, line in enumerate(lines_to_draw):
                text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
                text_x = max(10, (width - text_size[0]) // 2)
                text_y = base_y + (i + 1) * line_height
                # Strong shadow + pure white text
                cv2.putText(
                    frame,
                    line,
                    (text_x + 4, text_y + 4),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    line,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Mux original audio back in using ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_video,
        "-i", input_video,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        output_video,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not found in PATH")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed while muxing audio: {e.stderr}")
    finally:
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except OSError:
            pass

    return output_video


def upload_to_youtube(
    video_path: str,
    title: str,
    description: str,
    tags: list[str]
) -> dict:
    """
    Upload a video to YouTube as a public Short.
    
    Args:
        video_path: Path to the video file to upload.
        title: The video title.
        description: The video description.
        tags: List of tags for the video.
    
    Returns:
        A dict with "video_id" and "url" keys.
    
    Raises:
        RuntimeError: If the upload fails or credentials are missing.
    """
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    CLIENT_SECRETS_FILE = "client_secrets.json"
    TOKEN_FILE = "youtube_token.json"
    
    try:
        creds = None
        
        # Load existing credentials
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        # Refresh or get new credentials if needed
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(CLIENT_SECRETS_FILE):
                    raise RuntimeError(f"Missing {CLIENT_SECRETS_FILE}. Download it from Google Cloud Console.")
                
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for future use
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())
        
        # Build YouTube API client
        youtube = build("youtube", "v3", credentials=creds)
        
        # Video metadata
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": "15",  # Pets & Animals category
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        }
        
        # Upload the video
        media = MediaFileUpload(
            video_path,
            mimetype="video/mp4",
            resumable=True
        )
        
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info(f"Upload progress: {int(status.progress() * 100)}%")
        
        video_id = response["id"]
        video_url = f"https://www.youtube.com/shorts/{video_id}"
        
        logger.info(f"Video uploaded successfully: {video_url}")
        
        return {
            "video_id": video_id,
            "url": video_url
        }
    
    except FileNotFoundError as e:
        raise RuntimeError(f"Video file not found: {video_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload to YouTube: {e}")


def upload_generated_video_to_youtube(animal: str, output_dir: str = "outputs") -> None:
    """
    Upload a previously generated animal video to YouTube using its metadata.
    
    Args:
        animal: The name of the animal (used to find the metadata file).
        output_dir: Directory where the video and metadata are stored (default: "outputs").
    
    Raises:
        RuntimeError: If metadata file is missing or upload fails.
    """
    meta_path = os.path.join(output_dir, f"{animal}_meta.json")
    
    # Load metadata
    if not os.path.exists(meta_path):
        raise RuntimeError(f"Metadata file not found: {meta_path}")
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse metadata file: {e}")
    
    # Extract required fields
    video_path = metadata.get("video_path")
    title = metadata.get("title")
    description = metadata.get("description")
    hashtags = metadata.get("hashtags", [])
    
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")
    
    if not title:
        raise RuntimeError("Missing 'title' in metadata")
    
    if not description:
        raise RuntimeError("Missing 'description' in metadata")
    
    # Convert hashtags to tags (remove # prefix if present)
    tags = [tag.lstrip("#") for tag in hashtags]
    
    # Upload to YouTube
    logger.info(f"Uploading {animal} video to YouTube...")
    result = upload_to_youtube(video_path, title, description, tags)
    
    logger.info(f"Upload complete! Title: {title}, URL: {result['url']}")


def list_ready_videos(output_dir: str = "outputs") -> list[dict]:
    """
    List all generated videos that are ready to upload.
    
    Args:
        output_dir: Directory to scan for metadata files (default: "outputs").
    
    Returns:
        A list of dicts containing: animal, video_path, title, description, hashtags.
    """
    ready_videos = []
    
    if not os.path.exists(output_dir):
        return ready_videos
    
    # Find all metadata files
    meta_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith("_meta.json")
    ])
    
    for meta_file in meta_files:
        meta_path = os.path.join(output_dir, meta_file)
        
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Only include if video file exists
            video_path = metadata.get("video_path", "")
            if video_path and os.path.exists(video_path):
                ready_videos.append({
                    "animal": metadata.get("animal", ""),
                    "video_path": video_path,
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "hashtags": metadata.get("hashtags", []),
                })
        except (json.JSONDecodeError, IOError):
            # Skip invalid or unreadable files
            continue
    
    return ready_videos


def pick_oldest_ready_videos(limit: int, output_dir: str = "outputs") -> list[dict]:
    """
    Pick the oldest ready videos based on metadata file modification time.
    
    Args:
        limit: Maximum number of videos to return.
        output_dir: Directory to scan for metadata files (default: "outputs").
    
    Returns:
        A list of at most `limit` dicts (oldest first), each containing:
        animal, video_path, title, description, hashtags.
    """
    ready_videos = []
    
    if not os.path.exists(output_dir):
        return ready_videos
    
    # Find all metadata files with their modification times
    meta_files = [
        f for f in os.listdir(output_dir)
        if f.endswith("_meta.json")
    ]
    
    # Build list of (mtime, meta_path, metadata) tuples
    videos_with_mtime = []
    
    for meta_file in meta_files:
        meta_path = os.path.join(output_dir, meta_file)
        
        try:
            mtime = os.path.getmtime(meta_path)
            
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Only include if video file exists
            video_path = metadata.get("video_path", "")
            if video_path and os.path.exists(video_path):
                videos_with_mtime.append((
                    mtime,
                    {
                        "animal": metadata.get("animal", ""),
                        "video_path": video_path,
                        "title": metadata.get("title", ""),
                        "description": metadata.get("description", ""),
                        "hashtags": metadata.get("hashtags", []),
                    }
                ))
        except (json.JSONDecodeError, IOError, OSError):
            # Skip invalid or unreadable files
            continue
    
    # Sort by modification time (oldest first) and take up to limit
    videos_with_mtime.sort(key=lambda x: x[0])
    ready_videos = [video for _, video in videos_with_mtime[:limit]]
    
    return ready_videos


def upload_next_ready_videos(count: int, output_dir: str = "outputs") -> None:
    """
    Upload the oldest ready videos to YouTube.
    
    Args:
        count: Number of videos to upload.
        output_dir: Directory where videos and metadata are stored (default: "outputs").
    """
    videos = pick_oldest_ready_videos(count, output_dir)
    
    if not videos:
        logger.warning("No ready videos found to upload.")
        return
    
    logger.info(f"Uploading {len(videos)} video(s) to YouTube...")
    
    for i, video in enumerate(videos, 1):
        animal = video["animal"]
        video_path = video["video_path"]
        title = video["title"]
        description = video["description"]
        hashtags = video["hashtags"]
        
        # Convert hashtags to tags (remove # prefix if present)
        tags = [tag.lstrip("#") for tag in hashtags]
        
        try:
            logger.info(f"[{i}/{len(videos)}] Uploading: {animal} - {title}")
            result = upload_to_youtube(video_path, title, description, tags)
            logger.info(f"[{i}/{len(videos)}] Success: {result['url']}")
        except RuntimeError as e:
            logger.error(f"[{i}/{len(videos)}] Failed to upload {animal}: {e}")


def generate_animal_video(animal: str, output_dir: str | None = None) -> str:
    """
    Generate a complete animal facts video with voiceover and subtitles.
    
    Args:
        animal: The name of the animal to create a video about.
        output_dir: Directory to save all output files (default: from config).
    
    Returns:
        The path to the final MP4 video file.
    
    Raises:
        RuntimeError: If any step in the video generation fails.
    """
    if output_dir is None:
        config = load_config()
        output_dir = config["output_dir"]
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        script_path = os.path.join(output_dir, f"{animal}_script.txt")
        voice_path = os.path.join(output_dir, f"{animal}_voice.mp3")
        image_path = os.path.join(output_dir, f"{animal}_image.jpg")
        raw_video_path = os.path.join(output_dir, f"{animal}_raw.mp4")
        srt_path = os.path.join(output_dir, f"{animal}.srt")
        final_video_path = os.path.join(output_dir, f"{animal}_final.mp4")
        meta_path = os.path.join(output_dir, f"{animal}_meta.json")
        
        # Generate facts
        logger.info(f"[STEP] Generating facts for {animal}")
        facts = generate_animal_facts(animal)
        
        # Build script
        script = build_script(animal, facts)
        
        # Save script to file (pure spoken text only)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        
        # Generate metadata
        metadata = generate_metadata(animal, script)
        metadata["animal"] = animal
        metadata["video_path"] = final_video_path
        
        # Save metadata to JSON file
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Ensure the script does not contain any TTS control artifacts
        if _script_has_tts_artifacts(script):
            logger.warning(
                "Script for '%s' contained TTS-like control text; regenerating.",
                animal,
            )
            # Try a few times to get a clean script
            for _ in range(4):
                script = build_script(animal, facts)
                if not _script_has_tts_artifacts(script):
                    with open(script_path, "w", encoding="utf-8") as f:
                        f.write(script)
                    break
            else:
                raise RuntimeError(
                    f"Could not generate a clean script for '{animal}' without TTS control text."
                )

        # Generate voiceover
        logger.info(f"[STEP] Generating voiceover for {animal}")
        generate_voiceover(script, voice_path)

        # Download a Pexels clip and build the background video
        logger.info(f"[STEP] Downloading media for {animal}")
        clips = download_animal_video(animal, output_dir)

        # If no suitable video clips were found, fall back to a single image
        if not clips:
            config = load_config()
            if config.get("require_video_background", True):
                raise RuntimeError(
                    f"No valid Pexels video clips found for '{animal}' and video-only backgrounds are required."
                )
            logger.warning(f"No valid Pexels video clips found for '{animal}'; attempting image fallback.")
            try:
                image_file = download_animal_image_with_scale(animal, download_dir=output_dir)
                create_video_from_image_and_audio(image_file, voice_path, raw_video_path)
            except Exception as e:
                raise RuntimeError(
                    f"No valid Pexels video clips found for '{animal}' and image fallback failed: {e}"
                )
        else:
            # Pick one good clip and loop it under the narration so
            # the background stays consistent and on-topic.
            chosen_clip = random.choice(clips)
            logger.info(f"[STEP] Rendering single looping clip for {animal}: {chosen_clip}")
            create_video_from_clip_and_audio(chosen_clip, voice_path, raw_video_path)
        
        # Generate subtitles
        generate_subtitles(voice_path, srt_path)
        
        # Burn subtitles into final video
        burn_subtitles(raw_video_path, srt_path, final_video_path)
        
        return final_video_path

    except ValueError as e:
        logger.exception("Value error while generating video for '%s'", animal)
        raise RuntimeError(f"Failed to generate animal video for '{animal}': {e}")
    except Exception as e:
        logger.exception("Unexpected error while generating video for '%s'", animal)
        raise RuntimeError(f"Failed to generate animal video for '{animal}': {e}")


def generate_batch(num_videos: int, output_dir: str | None = None) -> list[str]:
    """
    Generate multiple animal fact videos in batch.
    
    Args:
        num_videos: The number of videos to generate.
        output_dir: Directory to save all output files (default: from config).
    
    Returns:
        A list of paths to the generated video files.
    """
    if output_dir is None:
        config = load_config()
        output_dir = config["output_dir"]
    
    video_paths = []
    available_animals = ANIMALS.copy()
    random.shuffle(available_animals)
    
    for i in range(num_videos):
        # If we've exhausted all animals, reset the list
        if not available_animals:
            logger.warning("Animal list exhausted, restarting from the beginning.")
            available_animals = ANIMALS.copy()
            random.shuffle(available_animals)
        
        animal = available_animals.pop()
        logger.info(f"[{i + 1}/{num_videos}] Generating video for: {animal}")
        
        try:
            video_path = generate_animal_video(animal, output_dir)
            video_paths.append(video_path)
            logger.info(f"[{i + 1}/{num_videos}] Generated video for: {animal}")
        except RuntimeError as e:
            logger.error(f"[{i + 1}/{num_videos}] Failed to generate video for {animal}: {e}")
    
    return video_paths


def generate_weekly_batch(output_dir: str | None = None) -> list[str]:
    """
    Generate a week's worth of videos (14 videos for 2 per day for 7 days).
    
    Args:
        output_dir: Directory to save all output files (default: from config).
    
    Returns:
        A list of paths to the generated video files.
    """
    logger.info("Starting weekly batch generation (14 videos for 7 days)")
    
    video_paths = generate_batch(14, output_dir)
    
    logger.info(f"Weekly batch complete: {len(video_paths)}/14 videos generated successfully")
    
    return video_paths


def main() -> None:
    """Main entry point with CLI argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate short animal fact videos for TikTok/YouTube Shorts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python animal_facts_ai.py                    # Generate video for random animal
  python animal_facts_ai.py --animal dolphin   # Generate video for dolphin
  python animal_facts_ai.py --batch 5          # Generate 5 videos
  python animal_facts_ai.py --list-ready       # List all ready videos
  python animal_facts_ai.py --upload dolphin   # Upload dolphin video to YouTube
        """
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=int,
        metavar="N",
        help="Generate N videos in batch mode"
    )
    parser.add_argument(
        "--animal", "-a",
        type=str,
        metavar="NAME",
        help="Generate a video for a specific animal"
    )
    parser.add_argument(
        "--upload", "-u",
        type=str,
        metavar="NAME",
        help="Upload a previously generated video for the given animal to YouTube"
    )
    parser.add_argument(
        "--list-ready", "-l",
        action="store_true",
        help="List all videos that are ready to upload"
    )
    
    args = parser.parse_args()
    
    # Count how many action arguments were provided
    actions = sum([
        args.batch is not None,
        args.animal is not None,
        args.upload is not None,
        args.list_ready
    ])
    
    # If multiple conflicting actions, show error
    if actions > 1:
        parser.error("Options --batch, --animal, --upload, and --list-ready are mutually exclusive.")
    
    try:
        if args.list_ready:
            # List all ready videos
            config = load_config()
            ready = list_ready_videos(config["output_dir"])
            
            if not ready:
                logger.info("No videos ready for upload.")
            else:
                logger.info(f"Found {len(ready)} video(s) ready for upload:")
                for video in ready:
                    logger.info(f"  - {video['animal']}: {video['title']}")
        
        elif args.upload:
            # Upload a specific animal video
            config = load_config()
            upload_generated_video_to_youtube(args.upload, config["output_dir"])
        
        elif args.batch is not None:
            # Batch generation
            if args.batch < 1:
                parser.error("Batch size must be at least 1.")
            
            logger.info(f"Generating {args.batch} videos in batch mode")
            video_paths = generate_batch(args.batch)
            
            logger.info(f"Generated {len(video_paths)} videos:")
            for path in video_paths:
                logger.info(f"  - {path}")
        
        elif args.animal:
            # Generate video for specific animal
            logger.info(f"Generating video about: {args.animal}")
            final_video = generate_animal_video(args.animal)
            logger.info(f"Final video created: {final_video}")
        
        else:
            # Default: generate video for random animal
            animal = get_random_animal()
            logger.info(f"Generating video about: {animal}")
            final_video = generate_animal_video(animal)
            logger.info(f"Final video created: {final_video}")
    
    except RuntimeError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
