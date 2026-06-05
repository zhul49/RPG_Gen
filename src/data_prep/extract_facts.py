import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FACTS_DIR = REPO_ROOT / "data" / "processed" / "crd3_facts"

# gpt model for checking
MODEL = "gpt-5.5"

PROMPT_TEMPLATE = """\
Below is a transcript from a Dungeons & Dragons campaign (Critical Role).
Extract a list of important world entities someone playing in this world \
would need to remember: NPCs, locations, items, organizations, and ongoing \
plot threads. SKIP the player characters themselves (Grog, Keyleth, Percy, \
Scanlan, Tiberius, Vax, Vex, Pike).

Output ONE FACT PER LINE in this format:
  Name — short description (one sentence, present tense, concrete)

Aim for 12–20 entries. Output the list ONLY — no numbering, no header, no \
explanation. If the transcript is mostly housekeeping/intros, return an \
empty response.

--- TRANSCRIPT ---
{transcript}
--- END ---
"""


# build a short transcript for the prompt from the synopsis and a few of Matt's longer turns
def _episode_transcript(episode_dict, max_chars=12000):
    meta = episode_dict.get("METADATA", {})
    pieces = []
    blurb = meta.get("Wiki Blurb")
    if isinstance(blurb, str) and blurb.strip():
        pieces.append("WIKI:\n" + blurb.strip())

    syn = meta.get("Synopsis")
    if isinstance(syn, list):
        syn_text = []
        for section in syn:
            if isinstance(section, dict):
                head = section.get("heading", "")
                contents = section.get("content", [])
                if isinstance(contents, list):
                    for c in contents:
                        if isinstance(c, dict):
                            syn_text.append(f"[{head}] {c.get('content', '')}")
                        else:
                            syn_text.append(f"[{head}] {c}")
        if syn_text:
            pieces.append("SYNOPSIS:\n" + "\n".join(syn_text))

    # A few long Matt turns past the sponsorship stuff
    turns = episode_dict.get("TURNS", [])
    matt_long = []
    for i, t in enumerate(turns):
        if i < 50:
            continue
        if t.get("NAMES") != ["MATT"]:
            continue
        text = " ".join(t.get("UTTERANCES", []))
        if len(text) >= 500:
            matt_long.append(text[:1500])
        if len(matt_long) >= 4:
            break
    if matt_long:
        pieces.append("MATT NARRATION SAMPLES:\n" + "\n\n".join(matt_long))

    transcript = "\n\n".join(pieces)
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n...[truncated]"
    return transcript


def _parse_fact_list(text):
    facts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line:
            continue
        # skip short lines that end in a colon since those are headers
        if line.endswith(":") and len(line) < 40:
            continue
        facts.append(line)
    return facts


# return the list of facts for one episode, using the cached file if we have it
def extract_episode_facts(episode_path, client=None):
    episode_path = Path(episode_path)
    episode_id = episode_path.stem
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = FACTS_DIR / f"{episode_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())["facts"]

    if client is None:
        from openai import OpenAI
        client = OpenAI()

    episode = json.loads(episode_path.read_text())
    transcript = _episode_transcript(episode)
    prompt = PROMPT_TEMPLATE.format(transcript=transcript)

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    facts = _parse_fact_list(text)

    cache.write_text(json.dumps({
        "episode": episode_id,
        "model": MODEL,
        "facts": facts,
        "raw_response": text,
    }, indent=2))
    return facts
