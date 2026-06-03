from pathlib import Path
import yaml

_ROOT = Path(__file__).parent

with open(_ROOT / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

with open(_ROOT / "prompts.yaml", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)
