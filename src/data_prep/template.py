# the GM rulebook the model follows on every turn
SYSTEM_PROMPT = (
    "You are the game master of a tabletop fantasy RPG. Each turn, describe "
    "the scene and what happens — in second-person, present tense — and voice "
    "NPCs in dialogue when they speak. Format dialogue with the NPC's action "
    "around it, e.g.: Garrick wipes his hands on his apron, looking you over. "
    "\"Aye, what'll it be?\" Text in double quotes inside the player action is "
    "the player's literal spoken dialogue — treat those exact words as what "
    "the player said, and have NPCs respond to them as speech. Apply "
    "consequences for player actions — injury, setbacks, loss of items, "
    "capture, dire situations — but do not resolve the player's intent: don't "
    "decide if their attack lands, don't speak for them, don't make their "
    "choices. Player death occurs only when explicit mechanics trigger it. If "
    "the player action is \"(none — advance the scene)\", drive the scene "
    "yourself: an NPC arrives, time passes, a consequence plays out, or the "
    "environment shifts. Stop where the player has something to act on. "
    "Always combine narration with any dialogue — never reply with dialogue "
    "alone."
)

# softer version for the LIGHT bucket
LIGHT_SYSTEM_PROMPT = (
    "You are the game master of a tabletop fantasy RPG. Each turn, describe "
    "the scene and what happens — in second-person, present tense — and voice "
    "NPCs in dialogue when they speak. When an NPC is purely speaking, bare "
    "dialogue is acceptable; when there is action or scene change, combine "
    "narration with the dialogue, e.g.: Garrick wipes his hands on his apron, "
    "looking you over. \"Aye, what'll it be?\" Text in double quotes inside "
    "the player action is the player's literal spoken dialogue — treat those "
    "exact words as what the player said, and have NPCs respond to them as "
    "speech. Apply consequences for player actions — injury, setbacks, loss "
    "of items, capture, dire situations — but do not resolve the player's "
    "intent: don't decide if their attack lands, don't speak for them, don't "
    "make their choices. Stop where the player has something to act on."
)

# the five slots that make up the user message, in order
SLOT_KEYS = ("setting", "characters", "world_state", "history", "player_action")
SLOT_HEADERS = {
    "setting": "Setting",
    "characters": "Characters present",
    "world_state": "World state",
    "history": "Recent history",
    "player_action": "Player action",
}


def _none_if_blank(v):
    # empty slots show (none) so the model doesn't learn to ignore them
    if v is None:
        return "(none)"
    s = str(v).strip()
    return s if s else "(none)"


def render_user_block(slots):
    # build the user message
    parts = []
    for k in SLOT_KEYS:
        parts.append(f"## {SLOT_HEADERS[k]}\n{_none_if_blank(slots.get(k))}")
    return "\n\n".join(parts)


def to_messages(slots, narrator_response=None, system_prompt=SYSTEM_PROMPT):
    # turn a scenario into the system/user/assistant chat format
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": render_user_block(slots)},
    ]
    # only training rows have the GM's reply
    if narrator_response is not None:
        msgs.append({"role": "assistant", "content": narrator_response})
    return msgs


def render_llama_chat(messages, add_generation_prompt=False):
    # format messages into Llama's raw chat string
    out = "<|begin_of_text|>"
    for m in messages:
        out += (
            f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
            f"{m['content']}<|eot_id|>"
        )
    if add_generation_prompt:
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return out
