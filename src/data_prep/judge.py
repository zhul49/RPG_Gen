# llm judge that scores examples 1 to 5 and caches every score
import asyncio
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# judge score caches live in their own folder since they are not training rows
JUDGE_DIR = REPO_ROOT / "data" / "processed" / "judge"

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_CONCURRENCY = 20    # concurrent judge api calls

# the reactive rubric, used for both CRD3 and FIREBALL.
# the JSON braces in here are doubled up so str.format does not choke on them
JUDGE_PROMPT = """\
You are scoring tabletop RPG training data for QUALITY. We are building a \
training set for a model that will produce GM narration. We want only \
high-quality examples that clearly demonstrate the target behavior.

The GM response should be in-character narration. EITHER style is valid:

  (a) DIALOGUE MODE - GM voices an NPC with action wrapped around dialogue.
      Strong example: "Garrick wipes his hands on his apron, looking you over.
      'Aye, what'll it be?' He gestures to a row of mugs behind him."
  (b) NARRATION MODE - pure scene/environmental description, no NPC voicing.
      Strong example: "The path narrows as the trees close overhead. The air
      grows damp and somewhere ahead you hear water dripping into stone."

QUALITY SIGNALS (high score):
  - Concrete sensory detail (sight, sound, smell, texture, temperature)
  - Second-person, present-tense narration ("you see", "you hear", "before you")
  - In dialogue mode: an NPC clearly voiced with action wrapped around their quote
  - Coherent response to the player's action that advances the scene
  - Ends at a moment the player can react to

DEMERITS (low score):
  - Dice/mechanics talk ("rolls a 17", "DC 15", "5 hit points of damage", \
"saving throw", "AC 18", "initiative", "roll for")
  - Donation reads ("CritterJody, 10 bucks", "thanks for the sub", \
references to viewers/donors)
  - Out-of-character table chatter ("you guys", "okay so", "let me know", \
"gotcha", "all right so", "let's see")
  - GM quoting a player back instead of voicing an NPC
  - Vague / flowery without concrete imagery
  - Response makes no sense given the player's action
  - Cuts off mid-thought or is incoherent

Score the GM RESPONSE from 1 to 5:
  5 = excellent. Clean target-pattern example, no demerits, vivid detail.
  4 = good. Solid example with minor weaknesses, training-worthy.
  3 = mixed. Some quality but also has noticeable demerits.
  2 = poor. Mostly demerits, occasional good moment.
  1 = unusable. Mechanics/chatter/donations/incoherent.

BE STRICT. When in doubt between two scores, give the LOWER. We want only 4s
and 5s in the final dataset.

Return JSON: {{"score": <integer 1-5>, "reason": "short phrase"}}

PLAYER ACTION:
{player_action}

GM RESPONSE:
{narrator}
"""


# pull the player action text out of the user message
def extract_player_action(ex):
    user_msg = ex["messages"][1]["content"]
    if "## Player action\n" in user_msg:
        return user_msg.split("## Player action\n", 1)[1].strip()[:1000]
    return user_msg[:1000]


# load a judge cache file into a dict of id to score and reason
def load_judge_cache(path):
    cache = {}
    if not path.exists():
        return cache
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                score = rec.get("score")
                # handle cache lines that store a keep flag instead of a score
                if score is None and "keep" in rec:
                    if rec["keep"]:
                        score = 4
                    else:
                        score = 1
                cache[rec["id"]] = (score, rec.get("reason", ""))
            except Exception:
                continue
    return cache


# ask the judge model to score one example, retries with backoff on errors
async def judge_one(client, sem, ex_id, prompt):
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=200,
                )
                text = resp.choices[0].message.content
                if not text:
                    text = "{}"
                verdict = json.loads(text)
                score = verdict.get("score")
                try:
                    score = int(score)
                    if score < 1 or score > 5:
                        score = None
                except (TypeError, ValueError):
                    score = None
                return ex_id, score, verdict.get("reason", "")
            except Exception as e:
                is_rate = "RateLimit" in type(e).__name__ or "429" in str(e)
                if attempt == 5:
                    return ex_id, None, f"error: {type(e).__name__}: {str(e)[:120]}"
                # wait longer each attempt and even longer for rate limits
                if is_rate:
                    delay = 10 * (2 ** attempt)
                else:
                    delay = 2 * (2 ** attempt)
                if delay > 90:
                    delay = 90
                await asyncio.sleep(delay)
    return ex_id, None, "exhausted retries"


# run all the judge calls concurrently and append every result to the cache file
async def judge_all(items, cache_writer, threshold):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    tasks = []
    for ex_id, prompt in items:
        tasks.append(asyncio.create_task(judge_one(client, sem, ex_id, prompt)))
    results = {}
    done = 0
    total = len(tasks)
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, None: 0}
    for coro in asyncio.as_completed(tasks):
        ex_id, score, reason = await coro
        results[ex_id] = (score, reason)
        cache_writer.write(
            json.dumps({"id": ex_id, "score": score, "reason": reason}) + "\n"
        )
        cache_writer.flush()
        counts[score] = counts.get(score, 0) + 1
        done += 1
        if done % 200 == 0 or done == total:
            kept = 0
            for s in counts:
                if s is not None and s >= threshold:
                    kept += counts[s]
            parts = []
            for s in [5, 4, 3, 2, 1]:
                parts.append(f"{s}={counts.get(s, 0)}")
            dist = ", ".join(parts)
            err = counts.get(None, 0)
            print(f"    judged {done}/{total}: {dist}, err={err} (kept >= {threshold}: {kept})")
    return results


# keep the examples whose cached score clears the threshold
def filter_by_score(pool, cache, id_fn, threshold):
    kept = []
    for ex in pool:
        score = cache.get(id_fn(ex), (None, ""))[0]
        if score is not None and score >= threshold:
            kept.append(ex)
    return kept
