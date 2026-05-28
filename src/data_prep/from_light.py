from pathlib import Path

from datasets import load_from_disk

from src.data_prep.template import LIGHT_SYSTEM_PROMPT, to_messages


# LIGHT is a Facebook fantasy dialog dataset. Each row is a 2-character conversation with persona text. We treat one character the "self" as the NPC the GM is voicing, and the other is "partner" is the player

# turns LIGHT's setting dict into one line for the setting slot
def format_setting(setting):
    name = (setting.get("name") or "").strip()
    desc = (setting.get("description") or "").strip()
    if name and desc:
        return f"{name}. {desc}"
    return desc or name or "(none)"


# builds the personas
def format_characters(characters):
    return (
        f"You are voicing: {characters['self_name']}\n"
        f"Persona: {characters['self_persona'].strip()}\n"
        f"Other present: {characters['partner_name']}"
    )


# formats dialogue[0..end_index-1] as a transcript for the history slot
def format_history(dialogue, characters, end_index):
    self_name = characters["self_name"]
    partner_name = characters["partner_name"]
    lines = []
    for j in range(end_index):
        speaker = self_name if j % 2 == 0 else partner_name
        lines.append(f'{speaker}: "{dialogue[j].strip()}"')
    return "\n".join(lines) if lines else "(none)"


# yields training examples from one LIGHT conversation row
def row_to_examples(row):
    dialogue = [d.strip() for d in row.get("dialogue", []) if d and d.strip()]
    if len(dialogue) < 3:
        return

    characters = row["characters"]
    setting_text = format_setting(row.get("setting", {}))
    characters_text = format_characters(characters)

    for self_turn_index in range(2, len(dialogue), 2):
        history_text = format_history(dialogue, characters, self_turn_index - 1)
        player_action = (
            f'{characters["partner_name"]} says: "{dialogue[self_turn_index - 1]}"'
        )
        narrator = dialogue[self_turn_index]

        slots = {
            "setting": setting_text,
            "characters": characters_text,
            "world_state": "(none)",
            "history": history_text,
            "player_action": player_action,
        }
        yield {
            "messages": to_messages(
                slots,
                narrator_response=narrator,
                system_prompt=LIGHT_SYSTEM_PROMPT,
            ),
            "source": "light",
        }


# walks the LIGHT dataset and yields training examples from every row
def iter_examples(light_dir, splits=("train",)):
    ds = load_from_disk(str(light_dir))
    for split in splits:
        if split not in ds:
            continue
        for row in ds[split]:
            yield from row_to_examples(row)
