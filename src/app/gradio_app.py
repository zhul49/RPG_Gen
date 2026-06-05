import gradio as gr

from src.data_prep.template import to_messages, SYSTEM_PROMPT
from src.inference.generate import generate, load
from src.rag.extractor import extract_facts
from src.rag.store import WorldStore
from src.config import (
    DEFAULT_SETTING, DEFAULT_CHARACTERS, ADAPTER, DB_PATH, PORT,
    TEMPERATURE, REPETITION_PENALTY, MAX_NEW_TOKENS, RAG_K, HISTORY_TURNS,
)

# use the same prompt the model was trained on
INFERENCE_SYSTEM_PROMPT = SYSTEM_PROMPT

# model and store get loaded in main
MODEL = None
TOK = None
STORE = None

STORE_WRITE = True  # write extracted facts to memory


def facts_md():
    facts = STORE.all()
    if not facts:
        return "*(no facts yet)*"
    return "\n".join(f"- {f}" for f in facts)


def chat(action, history, setting, chars, turns):
    action = (action or "").strip()
    if not action:
        return history, turns, facts_md(), ""

    # pull relevant facts for the world_state slot
    retrieved = STORE.query(f"{setting}\n{chars}\n{action}", k=RAG_K)
    world_state = "\n".join(f"- {f}" for f in retrieved) or "(none)"

    hist = "\n".join(f"{spk}: {txt}" for spk, txt in (turns or [])[-HISTORY_TURNS:]) or "(none)"

    msgs = to_messages({
        "setting": setting,
        "characters": chars,
        "world_state": world_state,
        "history": hist,
        "player_action": action,
    }, narrator_response=None, system_prompt=INFERENCE_SYSTEM_PROMPT)

    response, _ = generate(
        MODEL, TOK, msgs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
    )
    response = response.strip()

    turns = list(turns or []) + [("Player", action), ("GM", response)]
    history = list(history or []) + [
        {"role": "user", "content": action},
        {"role": "assistant", "content": response},
    ]

    if STORE_WRITE:
        new_facts = extract_facts(response)
        if new_facts:
            STORE.add(new_facts, source_turn=len(turns))

    return history, turns, facts_md(), ""


def regenerate(history, setting, chars, turns):
    # drop the last GM reply and the player turn before it, then run again
    if not turns or turns[-1][0] != "GM":
        return history, turns, facts_md(), ""
    last_player = turns[-2][1]
    turns = turns[:-2]
    history = history[:-2] if len(history) >= 2 else []
    return chat(last_player, history, setting, chars, turns)


def reset():
    return [], [], facts_md()


def forget():
    STORE.clear()
    return facts_md()


def apply_scene():
    STORE.clear()
    return [], [], facts_md()


def main():
    global MODEL, TOK, STORE

    print(f"loading {ADAPTER}...")
    MODEL, TOK = load(ADAPTER)
    print("model ready")

    # set up the RAG memory store
    STORE = WorldStore(DB_PATH)
    print(f"RAG: {STORE.count()} facts in memory")

    with gr.Blocks(title="GM RPG") as app:
        gr.Markdown("# Game Master RPG")
        gr.Markdown("_Fine-tuned Llama 3.1 8B + RAG memory._")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Accordion("Scene", open=True):
                    setting_box = gr.Textbox(label="Setting", value=DEFAULT_SETTING, lines=3)
                    chars_box = gr.Textbox(label="Characters present", value=DEFAULT_CHARACTERS, lines=5)
                    apply_btn = gr.Button("Apply scene (clears history + memory)")

                chatbot = gr.Chatbot(label="Conversation", height=480)

                action_box = gr.Textbox(
                    label="Player action",
                    placeholder="e.g. I head below deck to check on the sugar barrels.",
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

    # share=True always makes a public link
    app.launch(server_name="0.0.0.0", server_port=PORT, share=True)


if __name__ == "__main__":
    main()
