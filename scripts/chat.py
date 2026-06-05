import sys
from pathlib import Path

# add the repo root to the path so the src imports below work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.generate import load, generate
from src.data_prep.template import to_messages, SYSTEM_PROMPT

# the model was trained on this prompt, so use the same one or it gets worse
INFERENCE_SYSTEM_PROMPT = SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SETTING = (
    "The Marigold, a merchant ship a few days out at sea, its hold packed with "
    "barrels of sugar bound for the Kingdom of Hyde. Salt spray, creaking "
    "timbers, and a stiff wind in the sails."
)
DEFAULT_CHARACTERS = (
    "- Captain John — weathered, steady, has made this run a dozen times.\n"
    "- Mara — the first mate, sharp-tongued and always watching the horizon.\n"
    "- Tom — the young deckhand, eager but green.\n"
    "- Bess — the cook, runs the galley and hears all the gossip.\n"
)


# how many past turns to send back to the model as history
HISTORY_TURNS = 20

ADAPTER = REPO_ROOT / "runs" / "v11" / "final"
DB_PATH = REPO_ROOT / "data" / "world_state" / "chroma"
TEMPERATURE = 0.6
REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS = 300
RAG_K = 20  # how many facts to pull from memory each turn


def render_history(turns):
    if not turns:
        return "(none)"
    # only keep the last HISTORY_TURNS turns, then format each as "speaker: text"
    return "\n".join(f"{spk}: {txt}" for spk, txt in turns[-HISTORY_TURNS:])


def main():
    model, tok = load(ADAPTER)

    # set up the RAG memory store
    from src.rag.store import WorldStore
    from src.rag.extractor import extract_facts as _extract
    store = WorldStore(DB_PATH)
    extract_facts = _extract

    setting = DEFAULT_SETTING
    characters = DEFAULT_CHARACTERS
    turns = []
    turn_idx = 0

    while True:
        try:
            action = input("Player> ").strip()
        except (EOFError, KeyboardInterrupt):
            # ctrl-c or ctrl-d just exits cleanly
            print()
            break

        if not action or action == "/quit":
            break
        if action == "/reset":
            turns = []
            print("[history cleared]\n")
            continue
        if action == "/forget":
            store.clear()
            print("[RAG memory wiped]\n")
            continue
        if action.startswith("/scene "):
            # everything after "/scene " is the new setting
            setting = action[len("/scene "):].strip()
            turns = []
            store.clear()
            print(f"[scene -> {setting}; history + memory cleared]")
            print("[note: characters unchanged — use /chars <text> to set the "
                  "cast for this scene]\n")
            continue
        if action.startswith("/chars "):
            # everything after "/chars " is the new cast
            characters = action[len("/chars "):].strip()
            print(f"[characters -> {characters}]\n")
            continue
        if action == "/facts":
            facts = store.all()
            print(f"[RAG memory: {len(facts)} facts]")
            for f in facts:
                print(f"  - {f}")
            print()
            continue

        # ask memory for facts related to what's going on right now
        query = f"{setting}\n{characters}\n{action}"
        retrieved = store.query(query, k=RAG_K)

        # turn the retrieved facts into a bullet list, or "(none)" if empty
        world_state = (
            "\n".join(f"- {f}" for f in retrieved)
            if retrieved else "(none)"
        )

        # fill in the prompt template slots
        slots = {
            "setting": setting,
            "characters": characters,
            "world_state": world_state,
            "history": render_history(turns),
            "player_action": action,
        }
        # narrator_response=None means we want the model to generate it
        msgs = to_messages(slots, narrator_response=None,
                           system_prompt=INFERENCE_SYSTEM_PROMPT)

        response, _ = generate(
            model, tok, msgs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
        )
        response = response.strip()

        # save this exchange so it shows up in history next turn
        turns.append(("Player", action))
        turns.append(("GM", response))
        turn_idx += 1

        # pull facts out of the GM's reply and save them to memory
        new_facts = extract_facts(response)
        if new_facts:
            store.add(new_facts, source_turn=turn_idx)


if __name__ == "__main__":
    main()
