import argparse
from pathlib import Path

import gradio as gr

from src.data_prep.template import to_messages
from src.inference.generate import generate, load
from src.rag.extractor import extract_facts
from src.rag.store import WorldStore


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_SETTING = (
    "The Crooked Tankard, a cramped inn at the edge of a port town. Smoke "
    "from cheap tallow candles, sour ale, salt on the wind through the open "
    "shutters."
)
DEFAULT_CHARACTERS = (
    "You are voicing the inn's regulars and staff. Most prominent:\n"
    "- Garrick — the bartender, balding, missing two fingers on his right hand.\n"
    "- A hooded figure in the corner, untouched mug in front of them."
)


# set in main()
MODEL = None
TOK = None
STORE = None
TEMPERATURE = 0.6
MAX_NEW_TOKENS = 300
RAG_K = 5


def facts_md():
    if STORE is None:
        return "*(RAG disabled)*"
    facts = STORE.all()
    if not facts:
        return "*(no facts yet)*"
    return "\n".join(f"- {f}" for f in facts)


def chat(action, history, setting, chars, turns):
    action = (action or "").strip()
    if not action:
        return history, turns, facts_md(), ""

    # pull relevant facts for the world_state slot
    if STORE is not None:
        retrieved = STORE.query(f"{setting}\n{chars}\n{action}", k=RAG_K)
        world_state = "\n".join(f"- {f}" for f in retrieved) or "(none)"
    else:
        world_state = "(none)"

    hist = "\n".join(f"{spk}: {txt}" for spk, txt in (turns or [])[-8:]) or "(none)"

    msgs = to_messages({
        "setting": setting,
        "characters": chars,
        "world_state": world_state,
        "history": hist,
        "player_action": action,
    }, narrator_response=None)

    response, _ = generate(
        MODEL, TOK, msgs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    response = response.strip()

    turns = list(turns or []) + [("Player", action), ("GM", response)]
    history = list(history or []) + [
        {"role": "user", "content": action},
        {"role": "assistant", "content": response},
    ]

    if STORE is not None:
        new_facts = extract_facts(response)
        if new_facts:
            STORE.add(new_facts, source_turn=len(turns))

    return history, turns, facts_md(), ""


def regenerate(history, setting, chars, turns):
    # drop the last GM and the player turn that triggered it  then re-run
    if not turns or turns[-1][0] != "GM":
        return history, turns, facts_md(), ""
    last_player = turns[-2][1]
    turns = turns[:-2]
    history = history[:-2] if len(history) >= 2 else []
    return chat(last_player, history, setting, chars, turns)


def reset():
    return [], [], facts_md()


def forget():
    if STORE is not None:
        STORE.clear()
    return facts_md()


def apply_scene():
    if STORE is not None:
        STORE.clear()
    return [], [], facts_md()


def main():
    global MODEL, TOK, STORE, TEMPERATURE, MAX_NEW_TOKENS, RAG_K

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=REPO_ROOT / "runs" / "v2" / "final")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--db-path", type=Path, default=REPO_ROOT / "data" / "world_state" / "chroma")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--rag-k", type=int, default=5)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print(f"loading {args.adapter}...")
    MODEL, TOK = load(args.adapter)
    print("model ready")

    if not args.no_rag:
        STORE = WorldStore(args.db_path)
        print(f"RAG: {STORE.count()} facts in memory")

    TEMPERATURE = args.temperature
    MAX_NEW_TOKENS = args.max_new_tokens
    RAG_K = args.rag_k

    with gr.Blocks(title="GM RPG") as app:
        gr.Markdown("# Game Master RPG")
        gr.Markdown("_Fine-tuned Llama 3.2 3B + RAG memory._")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Accordion("Scene", open=True):
                    setting_box = gr.Textbox(label="Setting", value=DEFAULT_SETTING, lines=3)
                    chars_box = gr.Textbox(label="Characters present", value=DEFAULT_CHARACTERS, lines=5)
                    apply_btn = gr.Button("Apply scene (clears history + memory)")

                chatbot = gr.Chatbot(label="Conversation", height=480)

                action_box = gr.Textbox(
                    label="Player action",
                    placeholder="e.g. I walk up to the bar and order an ale.",
                    lines=1, max_lines=4,
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    regen_btn = gr.Button("Regenerate last")
                    reset_btn = gr.Button("Reset history")

            with gr.Column(scale=2):
                gr.Markdown("### World bible (RAG memory)")
                facts_box = gr.Markdown(facts_md())
                forget_btn = gr.Button("Forget all memory")

        turns = gr.State([])

        for trigger in (send_btn.click, action_box.submit):
            trigger(chat,
                [action_box, chatbot, setting_box, chars_box, turns],
                [chatbot, turns, facts_box, action_box])

        regen_btn.click(regenerate,
            [chatbot, setting_box, chars_box, turns],
            [chatbot, turns, facts_box, action_box])
        reset_btn.click(reset, outputs=[chatbot, turns, facts_box])
        forget_btn.click(forget, outputs=[facts_box])
        apply_btn.click(apply_scene, outputs=[chatbot, turns, facts_box])

    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
