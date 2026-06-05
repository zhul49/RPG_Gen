import gzip
import json
import re
from pathlib import Path

from src.data_prep.template import to_messages

# A research dataset of real Discord D&D game sessions captured via the Avrae bot

# FIREBALL has no per-session setting info so a generic blub is used
SETTING_DEFAULT = (
    "A Dungeons & Dragons campaign. Adventurers move through a fantasy world "
    "of combat, dungeons, and intrigue, each turn taking mechanical actions "
    "the GM narrates the outcome of."
)

def strip_markdown(text):
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^>\s*", "", cleaned)
    cleaned = cleaned.strip("*_ ").strip()
    return cleaned


# bulleted list of just NPC names from the combat state
def format_character_names(combat_state, max_chars=600):
    if not combat_state:
        return "(none)"
    lines = []
    for character in combat_state:
        if not isinstance(character, dict):
            continue
        name = character.get("name", "?")
        lines.append(f"- {name}")
    text = "\n".join(lines)
    # cap long rosters
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n- ..."
    return text or "(none)"


# combines the GM's pre-narration and the bot command into one player_action line
def format_player_action(triple):
    commands = triple.get("commands_norm") or []
    pre_narration = triple.get("before_utterances") or []
    speaker = triple.get("current_actor") or "Player"

    parts = []
    if pre_narration:
        narration_text = " ".join(strip_markdown(u) for u in pre_narration if u).strip()
        if narration_text:
            parts.append(narration_text)
    if commands:
        parts.append("(command: " + ", ".join(str(c) for c in commands) + ")")

    if not parts:
        return None
    return f"[{speaker}] " + " ".join(parts)

# builds one training example from a FIREBALL triple
def triple_to_example(triple, min_narrator_chars=80):
    # pull and clean the GM's response
    after_utterances = triple.get("after_utterances") or []
    narrator = " ".join(strip_markdown(u) for u in after_utterances if u).strip()
    if len(narrator) < min_narrator_chars:
        return None

    player_action = format_player_action(triple)
    if not player_action:
        return None

    # format just the NPC names for the characters slot
    characters_text = format_character_names(triple.get("combat_state_before"))

    # put the record together
    slots = {
        "setting": SETTING_DEFAULT,
        "characters": characters_text,
        "world_state": "(none)",
        "history": "(none)",
        "player_action": player_action,
    }
    return {
        "messages": to_messages(slots, narrator_response=narrator),
        "source": "fireball",
    }


# yields one parsed triple
def read_triples(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except (OSError, EOFError):
        return

# walks every FIREBALL session file and yields training examples
def iter_examples(fireball_dir, min_narrator_chars=80):
    fireball_dir = Path(fireball_dir)
    seen_paths = set()

    for gz_path in fireball_dir.rglob("*.jsonl.gz"):
        if "anonymized/data/" in str(gz_path):
            continue
        seen_paths.add(gz_path)
        for triple in read_triples(gz_path):
            example = triple_to_example(triple, min_narrator_chars=min_narrator_chars)
            if example is not None:
                yield example

    for triple_path in fireball_dir.rglob("filtered_triples.jsonl"):
        if triple_path in seen_paths:
            continue
        for triple in read_triples(triple_path):
            example = triple_to_example(triple, min_narrator_chars=min_narrator_chars)
            if example is not None:
                yield example
