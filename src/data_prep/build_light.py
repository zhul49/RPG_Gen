# filter and judge the LIGHT examples for in character persona quality
import asyncio
import hashlib
import random
import re
from pathlib import Path

from src.data_prep import from_light
from src.data_prep.judge import (
    JUDGE_DIR,
    extract_player_action,
    filter_by_score,
    judge_all,
    load_judge_cache,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LIGHT_DIR = REPO_ROOT / "data" / "raw" / "light"
LIGHT_JUDGE_CACHE = JUDGE_DIR / "light_judge.jsonl"

# skip flag is True so a run uses the score cache and makes no api calls
SKIP_LIGHT_JUDGE = True   # dont make new LIGHT judge calls
LIGHT_CAP = 400           # LIGHT examples to keep after judging, 0 skips LIGHT
LIGHT_THRESHOLD = 5       # minimum LIGHT judge score to keep

# the JSON braces in here are doubled up so str.format does not choke on them
LIGHT_JUDGE_PROMPT = """\
You are scoring fantasy RPG training data for QUALITY. These examples teach a \
model to voice NPCs IN CHARACTER. The model sees a setting, a character \
description with persona, and a line spoken TO that character. It should \
respond AS that character — staying in voice and honoring the persona.

NOTE: unlike GM-narration examples, these responses are intentionally BARE \
DIALOGUE without narrative wrapping. Do NOT penalize the lack of scene \
description — that's not the target pattern for this bucket. Score only \
on whether the response is a high-quality in-character utterance.

QUALITY SIGNALS (high score):
  - Stays in character: response could only have been said by this specific \
persona, not by a generic person
  - Honors the persona: doesn't contradict the character's stated traits, \
role, or background
  - Substantive: actually responds to what was said; not just acknowledgement
  - Coherent: clear meaning, no typo-fragments or garbled text
  - Distinct voice: vocabulary, tone, or speech pattern that matches the \
persona (a king sounds different from a peasant)

DEMERITS (low score):
  - Pure acknowledgement only ("ok", "sure", "thanks", "yes please", "alright")
  - Contradicts the persona (a humble peasant using legal jargon; a violent \
bandit being polite for no reason; a child speaking in a scholarly register)
  - Typos, fragments, sentence cut off mid-clause
  - Generic response anyone could have given regardless of persona
  - Out-of-character meta-commentary ("good game", references to game mechanics)
  - Copy-pasted quotes, Bible verses, or known text dumped in
  - Crowdsource-worker filler ("I will help", "yes that is fine")

Score the RESPONSE from 1 to 5:
  5 = excellent. In-character, persona-honoring, substantive, distinctive voice.
  4 = good. Solid in-character response with minor weaknesses.
  3 = mixed. In-character but bland or partially generic.
  2 = poor. Weak persona-honoring or short and content-free.
  1 = unusable. Off-character, fragment/typo-riddled, or pure acknowledgement.

BE STRICT. When in doubt, score lower. We only want 4s and 5s in the final set.

Return JSON: {{"score": <integer 1-5>, "reason": "short phrase"}}

PERSONA OF THE CHARACTER BEING VOICED:
{persona}

INCOMING LINE (from another character):
{player_action}

RESPONSE TO SCORE:
{narrator}
"""

_LIGHT_MIN_GOLD_CHARS = 50
_LIGHT_PURE_ACK_RE = re.compile(
    r'^(ok|okay|yes|yeah|sure|alright|all right|thanks|thank you|nope|no|'
    r'fine|cool|nice|yep|yup|aye|agreed|of course|certainly|indeed)'
    r'[.!,]*\s*$',
    re.IGNORECASE,
)


# None if the example should be kept, else a short reason string
def _light_drop_reason(ex):
    gold = ex["messages"][2]["content"].strip()
    if len(gold) < _LIGHT_MIN_GOLD_CHARS:
        return "too_short"
    if _LIGHT_PURE_ACK_RE.match(gold):
        return "pure_ack"
    return None


# drop the obviously bad LIGHT examples before paying for the judge
def heuristic_filter(pool):
    kept = []
    for ex in pool:
        reason = _light_drop_reason(ex)
        if reason is None:
            kept.append(ex)
    print(f"light heuristic: kept {len(kept)} of {len(pool)}")
    return kept


# stable id from a hash of the gold text
def _light_id(ex):
    gold = ex["messages"][2]["content"]
    return "light#" + hashlib.sha1(gold.encode("utf-8")).hexdigest()[:16]


# pull the persona text out of the characters slot
def _extract_persona(ex):
    user_msg = ex["messages"][1]["content"]
    m = re.search(r'Persona: (.+?)\nOther present:', user_msg, re.DOTALL)
    if m:
        return m.group(1).strip()[:800]
    return "(unknown)"


# score every uncached LIGHT example then keep the good ones. the cap is
# applied after scoring so it can be tuned later without rejudging
def judge_light(light_pool):
    cache = load_judge_cache(LIGHT_JUDGE_CACHE)
    if not SKIP_LIGHT_JUDGE:
        todo = []
        for ex in light_pool:
            eid = _light_id(ex)
            if eid in cache and cache[eid][0] is not None:
                continue
            pers = _extract_persona(ex)
            pa = extract_player_action(ex)
            nar = ex["messages"][2]["content"][:2000]
            prompt = LIGHT_JUDGE_PROMPT.format(
                persona=pers, player_action=pa, narrator=nar,
            )
            todo.append((eid, prompt))
        if todo:
            print(f"light: judging {len(todo)} new candidates")
            JUDGE_DIR.mkdir(parents=True, exist_ok=True)
            with LIGHT_JUDGE_CACHE.open("a") as cf:
                results = asyncio.run(judge_all(todo, cf, LIGHT_THRESHOLD))
            for eid in results:
                cache[eid] = results[eid]

    kept = filter_by_score(light_pool, cache, _light_id, LIGHT_THRESHOLD)
    if LIGHT_CAP < len(kept):
        random.shuffle(kept)
        kept = kept[:LIGHT_CAP]
    print(f"light: kept {len(kept)} (score >= {LIGHT_THRESHOLD}, cap {LIGHT_CAP})")
    return kept


# run all the LIGHT steps and return the kept examples
def build():
    if LIGHT_CAP == 0:
        return []
    pool = list(from_light.iter_examples(LIGHT_DIR))
    random.shuffle(pool)
    pool = heuristic_filter(pool)
    kept = judge_light(pool)
    return kept
