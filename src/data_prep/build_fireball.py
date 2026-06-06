# filter and judge the FIREBALL examples with the reactive rubric
import asyncio
import hashlib
import random
import re
from pathlib import Path

from src.data_prep import from_fireball
from src.data_prep.judge import (
    JUDGE_DIR,
    JUDGE_PROMPT,
    extract_player_action,
    filter_by_score,
    judge_all,
    load_judge_cache,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIREBALL_DIR = REPO_ROOT / "data" / "raw" / "fireball_source"
FIREBALL_JUDGE_CACHE = JUDGE_DIR / "fireball_judge.jsonl"

# skip flag is True so a run uses the score cache and makes no api calls
SKIP_FIREBALL = False     # skip the FIREBALL bucket entirely
SKIP_FIREBALL_JUDGE = True    # dont make new FIREBALL judge calls
FIREBALL_CAP = 8000       # FIREBALL examples to take before filtering
FIREBALL_MIN_CHARS = 80   # reject examples with narration shorter than this
FIREBALL_THRESHOLD = 4    # minimum FIREBALL judge score to keep

_FIREBALL_MIN_GOLD_CHARS = 150
_FIREBALL_FIRST_PERSON_WORDS = {
    "i", "i'm", "i'll", "i'd", "i've", "my", "me", "mine", "myself",
}
_FIREBALL_SECOND_PERSON_WORDS = {
    "you", "your", "yours", "you're", "you've", "you'll", "you'd", "yourself",
}
_FIREBALL_INTENT_RESOLUTION_PHRASES = [
    "you successfully", "you manage to", "your attack lands",
    "your attack hits", "you knock him", "you knock her", "you knock them",
    "you kill ", "you defeat", "you slay", "you connect with",
]


# true if the text opens with a code block or raw json
def _starts_with_struct(text):
    text = text.lstrip()
    return text.startswith("```") or text.startswith("{") or text.startswith("[")


# true if the text reads like the player wrote it instead of the GM
def _is_first_person(text):
    stripped = text.strip()
    if stripped.startswith(("I ", "I'm ", "I'll ", "I'd ", "I've ", "My ", "Me ")):
        return True
    words = re.findall(r"\b[\w']+\b", stripped.lower())
    if not words:
        return True
    first = 0
    second = 0
    for w in words:
        if w in _FIREBALL_FIRST_PERSON_WORDS:
            first += 1
        if w in _FIREBALL_SECOND_PERSON_WORDS:
            second += 1
    return first > 2 and second == 0


# true if most of the text is inside quotation marks
def _is_pure_dialogue(text):
    stripped = text.strip()
    if not stripped:
        return True
    inside = False
    quoted = 0
    for c in stripped:
        if c == '"':
            inside = not inside
        elif inside:
            quoted += 1
    return quoted / len(stripped) > 0.7


# true if the narration resolves the player's attack for them
def _resolves_intent(text):
    low = text.lower()
    for p in _FIREBALL_INTENT_RESOLUTION_PHRASES:
        if p in low:
            return True
    return False


# None if the example should be kept, else a short reason string
def _fireball_drop_reason(ex):
    gold = ex["messages"][2]["content"].strip()
    if len(gold) < _FIREBALL_MIN_GOLD_CHARS:
        return "too_short"
    if _starts_with_struct(gold):
        return "codeblock_or_struct"
    if _is_first_person(gold):
        return "player_pov"
    if _is_pure_dialogue(gold):
        return "pure_dialogue"
    if _resolves_intent(gold):
        return "intent_resolution"
    return None


# drop the obviously bad FIREBALL examples before paying for the judge
def heuristic_filter(pool):
    kept = []
    for ex in pool:
        reason = _fireball_drop_reason(ex)
        if reason is None:
            kept.append(ex)
    print(f"fireball heuristic: kept {len(kept)} of {len(pool)}")
    return kept


# stable id from a hash of the gold text since FIREBALL has no episode ids
def _fireball_id(ex):
    gold = ex["messages"][2]["content"]
    return "fireball#" + hashlib.sha1(gold.encode("utf-8")).hexdigest()[:16]


# score every uncached FIREBALL example then keep the good ones
def judge_fireball(fireball_pool):
    cache = load_judge_cache(FIREBALL_JUDGE_CACHE)
    if not SKIP_FIREBALL_JUDGE:
        todo = []
        for ex in fireball_pool:
            eid = _fireball_id(ex)
            if eid in cache and cache[eid][0] is not None:
                continue
            pa = extract_player_action(ex)
            nar = ex["messages"][2]["content"][:2000]
            prompt = JUDGE_PROMPT.format(player_action=pa, narrator=nar)
            todo.append((eid, prompt))
        if todo:
            print(f"fireball: judging {len(todo)} new candidates")
            JUDGE_DIR.mkdir(parents=True, exist_ok=True)
            with FIREBALL_JUDGE_CACHE.open("a") as cf:
                results = asyncio.run(judge_all(todo, cf, FIREBALL_THRESHOLD))
            for eid in results:
                cache[eid] = results[eid]

    kept = filter_by_score(fireball_pool, cache, _fireball_id, FIREBALL_THRESHOLD)
    print(f"fireball: kept {len(kept)} of {len(fireball_pool)} "
          f"(score >= {FIREBALL_THRESHOLD})")
    return kept


# run all the FIREBALL steps and return the kept examples
def build():
    if SKIP_FIREBALL:
        return []
    pool = list(from_fireball.iter_examples(
        FIREBALL_DIR, min_narrator_chars=FIREBALL_MIN_CHARS
    ))
    if len(pool) < 50:
        print("fireball: pool is tiny, the source tarball is probably not "
              "extracted under data/raw/fireball_source")
    random.shuffle(pool)
    if FIREBALL_CAP < len(pool):
        pool = pool[:FIREBALL_CAP]
    pool = heuristic_filter(pool)
    kept = judge_fireball(pool)
    return kept
