import os
import re
from openai import OpenAI

MODEL = "gpt-4o-mini"

PROMPT_TEMPLATE = """\
Below is a tabletop RPG GM's narration of one turn. Extract only DURABLE, \
CONCRETE facts about the world that should be remembered for future turns.

Focus on:
- Named NPCs, with a short identifying detail
- Named locations, with a short detail
- Important physical items, gifts, threats, promises, deadlines
- Established relationships or plot facts

Use the entity's BARE CANONICAL NAME as the key — "Dax", "MAB", "Halsey Station" \
— NOT generic labels like "NPC", "Location", "Item", "Plot thread", "Tension", \
"Atmosphere", "Threat", or "Mood". One entity = one line.

Do NOT extract:
- The player's own actions, claims, or backstory
- Hypothetical, imagined, or "what if" statements (e.g. a character musing about \
what they would do IF they had a body) — these are not real world facts
- Figures of speech, metaphors, or sensory mood
- Transient emotional state, tension, or atmosphere
- Generic objects or scenery with no lasting significance
- Game mechanics or dice

Output ONE FACT PER LINE in this format:
  Name — short concrete fact

Aim for 0–6 entries. If the turn contains no durable world info, output nothing.

--- GM TURN ---
{gm_turn}
--- END ---
"""

# generic labels we drop instead of saving as real facts
NOISE_KEYS = {
    "npc", "unknown npc", "an npc", "plot thread", "tension", "atmosphere",
    "threat", "threat of death", "threat of consequences", "threat of takeover",
    "item", "location", "mood", "setting", "narrator", "player", "scene",
}

_client = None


def _client_lazy():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def extract_facts(gm_turn):
    # pull a list of durable facts out of a GM turn
    if not gm_turn or not gm_turn.strip():
        return []
    # no API key so we cannot call the model
    if not os.environ.get("OPENAI_API_KEY"):
        return []

    response = _client_lazy().chat.completions.create(
        model=MODEL,
        max_completion_tokens=500,
        messages=[{"role": "user",
                   "content": PROMPT_TEMPLATE.format(gm_turn=gm_turn)}],
    )
    text = response.choices[0].message.content or ""

    META_PHRASES = (
        "no durable", "no facts", "no new", "nothing to extract",
        "no world info", "no important", "no information",
    )

    facts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line:
            continue
        if line.endswith(":") and len(line) < 40:
            continue
        low = line.lower()
        # skip lines where the model says there are no facts
        skip = False
        for p in META_PHRASES:
            if p in low:
                skip = True
                break
        if skip:
            continue
        # real facts have a dash between the name and the description
        if "—" not in line and " - " not in line:
            continue
        # drop generic labels like NPC or Tension
        key = re.split(r"\s—\s|—| - ", line, maxsplit=1)[0].strip().lower()
        if key in NOISE_KEYS:
            continue
        facts.append(line)
    return facts
