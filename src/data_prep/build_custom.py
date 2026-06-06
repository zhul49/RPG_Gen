# merge the hand authored buckets into custom.jsonl, can also generate curated examples
import json
from pathlib import Path

from src.data_prep.template import SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "processed" / "full"
CURATED_CACHE = OUT_DIR / "authored_gm_examples.jsonl"
CONSEQUENCES_FILE = OUT_DIR / "authored_consequences.jsonl"
V6_GAPS_FILE = OUT_DIR / "authored_npc_fixes.jsonl"
V7_SLOT_COHERENCE_FILE = OUT_DIR / "authored_scene_coherence.jsonl"
CUSTOM_COMBINED_FILE = OUT_DIR / "custom.jsonl"

# skip flag is True so a run makes no api calls
SKIP_CURATED = True       # dont generate new curated examples
CURATED_N = 500           # curated examples to generate
CURATED_BATCH = 10        # examples per curated api call
CURATED_MODEL = "gpt-4o-mini"

# the JSON braces in here are doubled up so str.format does not choke on them
CURATED_PROMPT = """\
Generate {n} tabletop fantasy RPG game master training examples. The model \
being trained sees (setting, characters, world_state, history, player_action) \
and produces the GM's narration. Each example should demonstrate ONE of two \
valid modes:

  MODE A - DIALOGUE+ACTION: voice an NPC with action wrapped around dialogue.
  Example narrator: "Garrick wipes his hands on his apron, looking you over. \
'Aye, what'll it be?' He gestures toward a row of mugs on the shelf behind him."

  MODE B - PURE NARRATION: scene / environment / sensory description with NO \
NPC dialogue. Use when the player is exploring, observing, or alone.
  Example narrator: "The path narrows as the trees close overhead. The air \
grows damp; somewhere ahead you hear water dripping into stone. A faint glow \
shows through the leaves to your left."

Aim for ROUGHLY HALF AND HALF across the {n} examples. Constraints for ALL:
  - Second-person, present-tense GM voice
  - 3 to 6 sentences
  - Concrete sensory detail, no flowery generalities
  - End at a moment the player can react to; do NOT resolve the player's action
  - Vary setting (tavern, forest, cave, royal court, harbor, sewer, frozen \
pass, ruin, market, library), NPC type (when applicable), and tone (grim, \
comic, tense, eerie, mundane).

Each example must have these fields:
  - setting: one-paragraph location description
  - characters: who's present, e.g. "You are voicing: X / Persona: ... / \
Other present: Y", OR "(none - the party is alone)" for narration mode with \
no NPCs
  - history: 0-3 prior turns in transcript form, or "(none)" for first-turn \
examples (use "(none)" for at least half)
  - player_action: a specific in-character player action
  - mode: "dialogue" or "narration"
  - narrator: the gold GM response demonstrating the chosen mode

Return JSON: {{"examples": [<{n} objects>]}}. JSON ONLY, no commentary.\
"""

USER_TEMPLATE = """\
## Setting
{setting}

## Characters present
{characters}

## World state
(none)

## Recent history
{history}

## Player action
{player_action}"""


# turn one generated example into a training record with the standard messages
def _to_curated_record(ex):
    history = ex.get("history")
    if not history:
        history = "(none)"
    user_msg = USER_TEMPLATE.format(
        setting=ex["setting"],
        characters=ex["characters"],
        history=history,
        player_action=ex["player_action"],
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ex["narrator"]},
        ],
        "source": "curated",
        "mode": ex.get("mode", "unknown"),
    }


# generate curated examples in batches and append them to the cache file
def generate_curated():
    from openai import OpenAI
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = 0
    if CURATED_CACHE.exists():
        with CURATED_CACHE.open() as f:
            for line in f:
                existing += 1
    needed = CURATED_N - existing
    if needed <= 0:
        return
    client = OpenAI()
    f = CURATED_CACHE.open("a")
    batch_i = 0
    while needed > 0:
        n = min(CURATED_BATCH, needed)
        batch_i += 1
        print(f"curated batch {batch_i}: requesting {n} examples")
        try:
            resp = client.chat.completions.create(
                model=CURATED_MODEL,
                messages=[{"role": "user", "content": CURATED_PROMPT.format(n=n)}],
                response_format={"type": "json_object"},
                max_completion_tokens=8000,
            )
            text = resp.choices[0].message.content
            if not text:
                text = "{}"
            data = json.loads(text)
            examples = data.get("examples")
            if examples is None:
                # sometimes the model uses a different key so grab the first list we find
                examples = []
                for v in data.values():
                    if isinstance(v, list):
                        examples = v
                        break
            for ex in examples:
                try:
                    rec = _to_curated_record(ex)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                except KeyError as e:
                    print(f"  skip malformed example, missing {e}")
            needed -= n
        except Exception as e:
            print(f"  batch {batch_i} failed: {type(e).__name__}: {e}")
            needed -= n
    f.close()


# read a jsonl file into a list, missing file just means an empty list
def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# load curated examples and swap in the current system prompt from template.py
def load_curated():
    out = load_jsonl(CURATED_CACHE)
    for rec in out:
        if rec.get("messages") and rec["messages"][0].get("role") == "system":
            rec["messages"][0]["content"] = SYSTEM_PROMPT
    return out


# merge every hand authored bucket into one list and rewrite custom.jsonl
def load_custom():
    combined = []
    combined.extend(load_curated())
    combined.extend(load_jsonl(CONSEQUENCES_FILE))
    combined.extend(load_jsonl(V6_GAPS_FILE))
    combined.extend(load_jsonl(V7_SLOT_COHERENCE_FILE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CUSTOM_COMBINED_FILE.open("w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"custom: {len(combined)} rows -> {CUSTOM_COMBINED_FILE.name}")
    return combined


# generate curated examples if asked then return all the custom buckets merged
def build():
    if not SKIP_CURATED:
        generate_curated()
    custom = load_custom()
    return custom
