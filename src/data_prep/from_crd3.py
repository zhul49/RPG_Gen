import json
import re
from pathlib import Path

from src.data_prep.template import to_messages

#A research dataset of the Critical Role livestreamed D&D campaign

# no character action is labelled, for examples when the DM drives the scene
PROACTIVE_PLAYER_ACTION = "(none — advance the scene)"


# Critical Role has two campaigns, this helps give the LLM context of the scene
CAMPAIGN_DEFAULTS = {
    "C1": {
        "setting": (
            "Tal'Dorei — a continent of human kingdoms, elven holdings, and "
            "dwarven citadels. The adventuring party Vox Machina travels its "
            "roads taking on quests for the realm's powers."
        ),
        "characters": (
            "The party (Vox Machina):\n"
            "- Grog Strongjaw — goliath barbarian.\n"
            "- Keyleth — half-elven druid of the Air Ashari.\n"
            "- Percival de Rolo — gunslinger and gunsmith of Whitestone.\n"
            "- Scanlan Shorthalt — gnome bard.\n"
            "- Tiberius Stormwind — dragonborn sorcerer of Draconia.\n"
            "- Vax'ildan — half-elven rogue, twin to Vex'ahlia.\n"
            "- Vex'ahlia — half-elven ranger, twin to Vax'ildan.\n"
            "- Pike Trickfoot — gnome cleric."
        ),
    },
    "C2": {
        "setting": (
            "Wildemount — a continent of city-states, scarred coasts, and "
            "wild frontier. The adventuring party 'the Mighty Nein' wanders "
            "the land taking on uneasy jobs and stranger ones."
        ),
        "characters": (
            "The party (the Mighty Nein):\n"
            "- Caleb Widogast — human wizard with a troubled past.\n"
            "- Beauregard Lionett — human monk of the Cobalt Soul.\n"
            "- Fjord Stone — half-orc warlock, sea-traveler.\n"
            "- Jester Lavorre — tiefling cleric of the Traveler.\n"
            "- Mollymauk Tealeaf — tiefling blood hunter (early episodes).\n"
            "- Nott the Brave — goblin rogue, sharp tongue and crossbow.\n"
            "- Yasha Nydoorin — aasimar barbarian with a great-sword."
        ),
    },
}
DEFAULT_CAMPAIGN = "C1"

FILLER_WORDS_RE = re.compile(
    r'^(okay|ok|yeah|yes|no|uh|um|hmm|right|sure|all right|alright|'
    r'sounds good|gotcha|nope|yep|yup|aye|fine|cool|nice|wait|hold on|'
    r'i wait|we wait|i look around|i look|continue|continues|same|me too|'
    r'agreed|let.?s go|let.?s do it|i nod|nods|laughs|chuckles)\b',
    re.IGNORECASE,
)

RECAP_OPENING_RE = re.compile(
    r'^(hello[, ]+(welcome|and welcome)|welcome (back|to)|hi[,!]?\s+welcome|'
    r'picking up where|last (we left|time we|episode)|previously[,\s]+on|'
    r'so[,!]?\s+when we last|when we last (left|saw|met)|'
    r'before (we (begin|start)|getting)|so[,!]+\s*last)',
    re.IGNORECASE,
)

# checks for switch campaign and episode
def campaign_for_episode(episode_id):
    if not episode_id or len(episode_id) < 2:
        return DEFAULT_CAMPAIGN
    prefix = episode_id[:2].upper()
    if prefix in CAMPAIGN_DEFAULTS:
        return prefix
    return DEFAULT_CAMPAIGN

def is_filler_trigger(player_text):
    # strip speaker prefix
    body = re.sub(r'^\[[^\]]*\]\s*', '', player_text).strip()
    # strip parentheticals
    body_without_parens = re.sub(r'\s*\([^)]*\)\s*', ' ', body).strip()
    if not body_without_parens:
        return True

    # checks for filler words
    if len(body_without_parens) <= 25 and FILLER_WORDS_RE.match(body_without_parens):
        return True

    # checks for how long it is
    if len(body_without_parens) <= 12:
        return True

    return False

# joins strings into a single string
def join_utterances(utterances):
    if isinstance(utterances, list):
        cleaned = [u.strip() for u in utterances if u and u.strip()]
        return " ".join(cleaned)
    return str(utterances).strip()

# formats the recent history block for the json of the last 5 sentences before DM description
def format_history(turns, end_index, num_turns_back=5, max_chars_per_turn=220):
    start_index = max(0, end_index - num_turns_back)
    lines = []
    for turn in turns[start_index:end_index]:
        speakers = ", ".join(turn.get("NAMES", ["?"]))
        text = join_utterances(turn.get("UTTERANCES", []))
        if len(text) > max_chars_per_turn:
            text = text[:max_chars_per_turn].rsplit(" ", 1)[0] + "..."
        lines.append(f"[{speakers}] {text}")
    return "\n".join(lines) if lines else "(none)"

# grabs the most recent non-DM voice as the player action
def find_preceding_player_turn(turns, matt_turn_index, look_back=10):
    earliest = max(0, matt_turn_index - look_back) - 1
    for j in range(matt_turn_index - 1, earliest, -1):
        names = turns[j].get("NAMES")
        if names and names != ["MATT"]:
            return j
    return None

# formats the player turn as '[SPEAKER] text'
def build_player_action_line(turns, player_turn_index):
    speakers = ", ".join(turns[player_turn_index].get("NAMES", ["?"]))
    text = join_utterances(turns[player_turn_index].get("UTTERANCES", []))
    return f'[{speakers}] {text}'


# reactive requires real actions and proactive bucket requires filler
def matches_mode(player_action, mode):
    is_filler = is_filler_trigger(player_action)
    if mode == "reactive":
        return not is_filler
    if mode == "proactive":
        return is_filler
    return False


# assembles one finished training example record
def build_record(mode, episode_id, turn_index, narration,
                 history_text, original_player_action):
    campaign = campaign_for_episode(episode_id)
    defaults = CAMPAIGN_DEFAULTS[campaign]

    # in proactive mode the player line gets relabeled with the trigger
    if mode == "proactive":
        player_action_slot = PROACTIVE_PLAYER_ACTION
        source_tag = "crd3_proactive"
    else:
        player_action_slot = original_player_action
        source_tag = "crd3"

    slots = {
        "setting": defaults["setting"],
        "characters": defaults["characters"],
        "world_state": "(none)",
        "history": history_text,
        "player_action": player_action_slot,
    }

    record = {
        "messages": to_messages(slots, narrator_response=narration),
        "source": source_tag,
        "episode": episode_id,
        "turn_idx": turn_index,
    }
    if mode == "proactive":
        record["original_player_action"] = original_player_action
    return record


# walks one episode JSON and yields training examples
def episode_to_examples(
    episode_path,
    skip_first_turns=50,
    min_matt_chars=300,
    mode="reactive",
):
    episode = json.loads(episode_path.read_text())
    turns = episode.get("TURNS", [])
    episode_id = episode_path.stem

    for turn_index, turn in enumerate(turns):
        if turn_index < skip_first_turns:
            continue
        if turn.get("NAMES") != ["MATT"]:
            continue
        narration = join_utterances(turn.get("UTTERANCES", []))
        if len(narration) < min_matt_chars:
            continue
        if RECAP_OPENING_RE.match(narration.strip()):
            continue

        player_turn_index = find_preceding_player_turn(turns, turn_index)
        if player_turn_index is None:
            continue
        original_player_action = build_player_action_line(turns, player_turn_index)

        if not matches_mode(original_player_action, mode):
            continue

        yield build_record(
            mode=mode,
            episode_id=episode_id,
            turn_index=turn_index,
            narration=narration,
            history_text=format_history(turns, player_turn_index),
            original_player_action=original_player_action,
        )

def iter_examples(crd3_cleaned_dir, **kwargs):
    kwargs.setdefault("mode", "reactive")
    for episode_path in sorted(Path(crd3_cleaned_dir).glob("*.json")):
        yield from episode_to_examples(episode_path, **kwargs)


def iter_proactive_candidates(crd3_cleaned_dir, **kwargs):
    kwargs["mode"] = "proactive"
    for episode_path in sorted(Path(crd3_cleaned_dir).glob("*.json")):
        yield from episode_to_examples(episode_path, **kwargs)
