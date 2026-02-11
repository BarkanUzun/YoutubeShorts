# load_config
import json
import os
import logging

_config_cache = None
DEFAULT_CONFIG = {
	"output_dir": "outputs",
	"speech_speed": 1.3,
	"max_title_length": 60,
	"ffmpeg_zoom_max": 1.2
}

def load_config(config_path: str = "config.json") -> dict:
	global _config_cache
	if _config_cache is not None:
		return _config_cache
	config = DEFAULT_CONFIG.copy()
	if os.path.exists(config_path):
		try:
			with open(config_path, "r", encoding="utf-8") as f:
				user_config = json.load(f)
			config.update(user_config)
			logging.info(f"Loaded config from {config_path}")
		except json.JSONDecodeError as e:
			logging.warning(f"Failed to parse {config_path}: {e}. Using defaults.")
		except IOError as e:
			logging.warning(f"Failed to read {config_path}: {e}. Using defaults.")
	else:
		logging.info(f"Config file not found at {config_path}. Using defaults.")
	_config_cache = config
	return config
