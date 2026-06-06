# judge the CRD3 examples and fill the keepers world state slot with episode facts
import asyncio
import json
import os
import random
from pathlib import Path

from src.data_prep import from_crd3
from src.data_prep.judge import (
    JUDGE_DIR,
    JUDGE_PROMPT,
    extract_player_action,
    filter_by_score,
    judge_all,
    load_judge_cache,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CRD3_DIR = REPO_ROOT / "data" / "raw" / "crd3_source" / "data" / "cleaned data"
FACTS_DIR = REPO_ROOT / "data" / "processed" / "crd3_facts"
JUDGE_CACHE = JUDGE_DIR / "crd3_judge.jsonl"
PROACTIVE_JUDGE_CACHE = JUDGE_DIR / "crd3_judge_proactive.jsonl"

# skip flags are True so a run uses the score caches and makes no api calls
SKIP_JUDGE = True         # dont make new CRD3 judge calls
SKIP_PROACTIVE = False    # skip the proactive CRD3 part entirely
SKIP_PROACTIVE_JUDGE = True   # dont make new proactive judge calls
SKIP_FACTS = False        # dont extract or plug per episode facts
JUDGE_THRESHOLD = 4       # minimum judge score to keep an example
PROACTIVE_THRESHOLD = 4   # minimum proactive judge score to keep
MIN_MATT_CHARS = 500      # reject candidates with narration shorter than this

# the JSON braces in here are doubled up so str.format does not choke on them
PROACTIVE_JUDGE_PROMPT = """\
You are scoring tabletop RPG training data for QUALITY. We are building a \
training set for a model that will produce GM narration in PROACTIVE mode - \
turns where the GM advances the scene on their own initiative, without \
waiting for the player to declare an action. The player input slot will be \
empty or just say "advance the scene." We want only high-quality examples \
that clearly demonstrate the target behavior.

The GM response should drive the scene forward by introducing something new. \
ANY of these patterns is valid:

  (a) NEW NPC ARRIVAL - someone enters, approaches, addresses the party.
      Strong example: "The tavern door bangs open and a rain-soaked courier
      stumbles in, shouldering past patrons. He scans the room, eyes landing
      on you. 'You. You're the ones who took the contract?' He drops a
      sealed envelope on your table, water dripping from his sleeve."
  (b) ENVIRONMENTAL / TIME CHANGE - weather, lighting, time of day, season.
      Strong example: "The light through the canopy shifts as the sun drops
      toward the western ridge. A cold breeze rises off the lake, and the
      smell of woodsmoke drifts from the camp behind you."
  (c) CONSEQUENCE / EVENT - something the party set in motion plays out, or
      an external event interrupts.
      Strong example: "From the direction of the temple, a low bell begins
      to toll. Within moments, three more join it from across the city. You
      see merchants on the street stiffen and start packing up their stalls
      in obvious haste."
  (d) SCENE PROGRESSION - the camera moves: a new room is entered, a journey
      transitions to arrival, a performance moves to its next act.
      Strong example: "The road descends in long switchbacks and the trees
      thin. By midafternoon you crest a low rise and the valley opens
      below: terraced fields, a walled town, and the river winding north."

QUALITY SIGNALS (high score):
  - Introduces something NEW the player can react to (an NPC, an event, a \
revelation, an environmental shift) - does not just describe static \
surroundings the party already knew
  - Concrete sensory detail (sight, sound, smell, texture, temperature)
  - Second-person, present-tense narration ("you see", "you hear", \
"before you")
  - In dialogue mode: an NPC clearly voiced with action wrapped around their \
quote (not bare dialogue)
  - Ends at a moment the player can react to - a question asked, a threat \
revealed, a choice presented

DEMERITS (low score):
  - Dice/mechanics talk ("rolls a 17", "DC 15", "5 hit points of damage", \
"saving throw", "AC 18", "initiative", "roll for")
  - Donation reads or out-of-character table chatter ("CritterJody, 10 \
bucks", "thanks for the sub", "you guys", "okay so", "let me know", \
"all right so", "let's see")
  - Pure recap of what already happened with nothing new added
  - Mid-thought continuation that depends on the prior turn for sense \
(e.g. starts with "And then" / "So" / "But also" referring to something \
not in this turn)
  - Vague / flowery without concrete imagery
  - Cuts off mid-thought or is incoherent
  - GM voicing a player character or speaking for the player

NOTE ON PLAYER COHERENCE: do NOT penalize the response for ignoring the \
player's stated action. In proactive mode the player's input is empty or \
trivial; the GM is meant to drive the scene independently.

Score the GM RESPONSE from 1 to 5:
  5 = excellent. Clean proactive turn, introduces something new with vivid \
detail, ends on a clear hook for the player.
  4 = good. Solid proactive turn with minor weaknesses, training-worthy.
  3 = mixed. Some quality but also has noticeable demerits.
  2 = poor. Mostly demerits, or barely proactive.
  1 = unusable. Mechanics/chatter/donations/incoherent or pure static recap.

BE STRICT. When in doubt between two scores, give the LOWER. We want only 4s
and 5s in the final dataset.

Return JSON: {{"score": <integer 1-5>, "reason": "short phrase"}}

PRECEDING CONTEXT (what was happening before this turn):
{history}

GM RESPONSE TO SCORE:
{narrator}
"""


# stable id for a CRD3 example from its episode and turn
def _crd3_id(ex):
    return f"{ex.get('episode', '?')}#{ex.get('turn_idx', 0)}"


# pull the recent history text out of the user message
def _extract_history(ex):
    user_msg = ex["messages"][1]["content"]
    if "## Recent history\n" in user_msg and "## Player action\n" in user_msg:
        hist = user_msg.split("## Recent history\n", 1)[1]
        hist = hist.split("## Player action\n", 1)[0].strip()
        return hist[:2000]
    return "(none)"


# score every CRD3 candidate not already in the cache then keep the good ones
def judge_crd3(crd3_pool):
    cache = load_judge_cache(JUDGE_CACHE)
    if not SKIP_JUDGE:
        todo = []
        for ex in crd3_pool:
            eid = _crd3_id(ex)
            if eid in cache and cache[eid][0] is not None:
                continue
            pa = extract_player_action(ex)
            nar = ex["messages"][2]["content"][:2000]
            prompt = JUDGE_PROMPT.format(player_action=pa, narrator=nar)
            todo.append((eid, prompt))
        if todo:
            print(f"crd3: judging {len(todo)} new candidates")
            JUDGE_DIR.mkdir(parents=True, exist_ok=True)
            with JUDGE_CACHE.open("a") as cf:
                results = asyncio.run(judge_all(todo, cf, JUDGE_THRESHOLD))
            for eid in results:
                cache[eid] = results[eid]

    kept = filter_by_score(crd3_pool, cache, _crd3_id, JUDGE_THRESHOLD)
    print(f"crd3: kept {len(kept)} of {len(crd3_pool)} (score >= {JUDGE_THRESHOLD})")
    return kept


# same as judge_crd3 but with the proactive rubric and its own cache
def judge_crd3_proactive(proactive_pool):
    cache = load_judge_cache(PROACTIVE_JUDGE_CACHE)
    if not SKIP_PROACTIVE_JUDGE:
        todo = []
        for ex in proactive_pool:
            eid = _crd3_id(ex)
            if eid in cache and cache[eid][0] is not None:
                continue
            hist = _extract_history(ex)
            nar = ex["messages"][2]["content"][:2000]
            prompt = PROACTIVE_JUDGE_PROMPT.format(history=hist, narrator=nar)
            todo.append((eid, prompt))
        if todo:
            print(f"crd3 proactive: judging {len(todo)} new candidates")
            JUDGE_DIR.mkdir(parents=True, exist_ok=True)
            with PROACTIVE_JUDGE_CACHE.open("a") as cf:
                results = asyncio.run(judge_all(todo, cf, PROACTIVE_THRESHOLD))
            for eid in results:
                cache[eid] = results[eid]

    kept = filter_by_score(proactive_pool, cache, _crd3_id, PROACTIVE_THRESHOLD)
    print(f"crd3 proactive: kept {len(kept)} of {len(proactive_pool)} "
          f"(score >= {PROACTIVE_THRESHOLD})")
    return kept


# fill the world state slot of kept CRD3 examples with facts from their episode
def plug_facts(crd3_examples):
    facts_by_ep = {}
    if FACTS_DIR.exists():
        for fp in FACTS_DIR.glob("*.json"):
            try:
                text = fp.read_text()
                data = json.loads(text)
                facts_by_ep[data["episode"]] = data["facts"]
            except Exception:
                continue

    needed_eps = set()
    for ex in crd3_examples:
        ep = ex.get("episode")
        if ep:
            needed_eps.add(ep)
    missing = sorted(needed_eps - set(facts_by_ep.keys()))
    if missing and os.environ.get("OPENAI_API_KEY"):
        from src.data_prep.extract_facts import extract_episode_facts
        from openai import OpenAI
        client = OpenAI()
        print(f"crd3 facts: extracting {len(missing)} uncached episodes")
        for ep_id in missing:
            ep_path = CRD3_DIR / f"{ep_id}.json"
            if not ep_path.exists():
                continue
            try:
                facts_by_ep[ep_id] = extract_episode_facts(ep_path, client=client)
            except Exception as e:
                print(f"  {ep_id} failed: {type(e).__name__}: {e}")
                facts_by_ep[ep_id] = []

    plugged = 0
    for ex in crd3_examples:
        ep = ex.get("episode")
        facts = facts_by_ep.get(ep, [])
        if not facts:
            continue
        chosen = random.sample(facts, min(5, len(facts)))
        lines = []
        for fact in chosen:
            lines.append(f"- {fact}")
        ws = "\n".join(lines)
        user_msg = ex["messages"][1]["content"]
        marker = "## World state\n(none)"
        if marker in user_msg:
            ex["messages"][1]["content"] = user_msg.replace(
                marker, f"## World state\n{ws}"
            )
            plugged += 1
    print(f"crd3 facts: plugged into {plugged}/{len(crd3_examples)} examples")
    return crd3_examples


# run all the CRD3 steps and return the kept reactive and proactive examples
def build():
    crd3_pool = list(from_crd3.iter_examples(
        CRD3_DIR, min_matt_chars=MIN_MATT_CHARS
    ))
    crd3_kept = judge_crd3(crd3_pool)

    if SKIP_PROACTIVE:
        proactive_kept = []
    else:
        proactive_pool = list(from_crd3.iter_proactive_candidates(
            CRD3_DIR, min_matt_chars=MIN_MATT_CHARS
        ))
        proactive_kept = judge_crd3_proactive(proactive_pool)

    # facts go in before mixing so unused episodes dont get extracted
    if not SKIP_FACTS:
        crd3_kept = plug_facts(crd3_kept)
        if proactive_kept:
            proactive_kept = plug_facts(proactive_kept)
    return crd3_kept, proactive_kept
