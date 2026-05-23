# Design decisions log

One line per non-obvious decision with the reason. Newest at top.

## 2026-05-23 — LIGHT mirror choice: `dap-exp/light_dialog`
No official Facebook/Meta HF mirror exists for LIGHT. Canonical access is via ParlAI (`-t light_dialog`), but pulling in ParlAI as a dep is heavy. The mirror `dap-exp/light_dialog` (Nov 2024, 20.1k rows, 9.58 MB) has the right schema (`setting`, `characters` with `self_persona`/`partner_name`, `dialogue` sequence) and the row count is consistent with the original. No dataset card / no license statement, but the underlying LIGHT data is openly distributed under ParlAI, so the provenance concern is weaker than Storium's. Swap to direct ParlAI download if mirror access ever breaks. `dap-exp/light_dialog_wild` (42.6k rows) is a larger extension worth keeping in mind.

## 2026-05-23 — Conda over venv
Python 3.11 isn't system-installed (only 3.12 + miniconda-base 3.13). Conda already present, so `conda create -n gm_rpg python=3.11` is one command. Venv would have required installing 3.11 first via deadsnakes/uv/pyenv. If we later want lighter envs, swap to `uv venv --python 3.11` — uv handles Python install too.

## 2026-05-23 — Single `<|narrator|>` output role (not split narrator / dialogue)
Human GM turns are narration + NPC voicing intermixed, with NPC speech marked by quotes + dialogue tags. Single role matches how CRD3 actually looks and avoids forcing unnatural turn structure on the model.

## 2026-05-23 — CRD3 primary, LIGHT supplement, FIREBALL deferred, drop Storium and LIGHT-as-primary
- **Storium dropped:** original distribution site (storium.cs.umass.edu) returns 503 since at least 2024; no official mirror; only HF mirror is reshaped to ShareGPT chat format losing the character/scene cards that were the structural value.
- **CRD3 primary:** Matt Mercer = clean GM narrator voice signal; DM extraction is a one-liner (`names == ["MATT"]`); 159 eps / 399k turns; HF `microsoft/crd3`.
- **LIGHT supplement:** initial instinct was to drop LIGHT because it's character-to-character dialogue, not GM-narrating-to-player. But voicing NPCs IS half of GM craft — when the player says "I talk to bartender," model must voice the bartender. CRD3's intra-DM NPC voicing is muddy/interleaved; LIGHT's is clean and persona-grounded. Reframing turns into "GM narrating an NPC speaking" is a template/regex transformation, NOT GPT-4-per-turn.
- **FIREBALL deferred:** structured per-turn game state is the closest public analog to RAG-populated `<|world_state|>` slot. Hold for phase 2 — pull in if the model isn't attending to state at inference time.

## 2026-05-23 — Llama 3.2 3B Instruct + QLoRA chosen for fast iteration
96 GB total VRAM (2× RTX 6000 Ada) could comfortably handle 7-8B QLoRA. 3B chosen to keep iteration loops short. Budget a 7-8B comparison run once the pipeline is end-to-end working.

## 2026-05-23 — `<|world_state|>` slot must be populated during SFT
Empty-slot SFT trains the model to ignore the slot. Plan: synthesize world-state contents from *later turns of the same episode* — facts that appear in turn 7 become RAG context fed at turn 5. Same dataset, no extra label work, model actually learns to attend.
