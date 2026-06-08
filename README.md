# RPG_Gen

This is a LoRA fine-tune of Llama-3.1-8B that acts as the game master for a tabletop fantasy RPG. It narrates the scene in second person, plays the NPCs in character, and responds to whatever you decide to try, and it always hands the turn back to you at a point where you actually have something to do. I trained it on quality-judged transcripts from real D&D games like Critical Role and FIREBALL, fantasy character dialogue from the LIGHT dataset, and a batch of hand-written examples. Running alongside it is a retrieval memory that remembers what has happened, so the world stays consistent even in a long session.

![Game Master RPG GUI](images/UI%20Screenshot.png)

## Base model vs fine-tuned

### 1. Combat intent

**Player:** *I draw my cutlass and lunge at Captain John, shouting "The Marigold is mine now!"*

<table>
<tr><th>Fine-tuned GM</th><th>Base Llama-3.1-8B-Instruct</th></tr>
<tr valign="top"><td>

Captain John doesn't flinch. He keeps steering with one hand and raps the back of your head with the flat of his palm, hard enough that stars flash, not enough to break bone. "Put the blade away before I put you over the rail," he says, calm, glancing once at Mara, who's already moving toward the rigging. "We're three days from shore. Three days, and no place on this boat where you can hide." The deckhands go quiet; Tom looks like he might vomit.

</td><td>

As you draw your cutlass and shout, the entire crew stops what they're doing to look at you. Captain John's bushy eyebrows shoot up in surprise. "By the gods, lad! What's gotten into ye?" he exclaims, taking a step back, his hand hovering near his own sword but not drawing it. Mara rushes over, her eyes wide with concern. "Easy there, boyo! What's troublin' ye?" she asks, glancing nervously between you and the captain. Tom, who had been scrubbing the deck, drops his brush and gasps. "Oh no! Has he gone mad from the sun?" he squeaks, his face pale with fright. Bess pokes her head out of the galley, wiping flour from her hands. "What's all this commotion? Did someone steal yer grog, dearie?" she calls out, looking puzzled. The captain holds up a calming hand. "Now, now, let's all keep our wits about us! There's no need for violence among friends!" He chuckles nervously, trying to defuse the tension. "Why don't ye put that blade away and we'll talk this over like civilized folk, eh?" **What will you do?**

</td></tr>
</table>

### 2. Impossible / overpowered action

**Player:** *I leap off the deck, fly up into the storm clouds, and call down a bolt of lightning to split the mast.*

<table>
<tr><th>Fine-tuned GM</th><th>Base Llama-3.1-8B-Instruct</th></tr>
<tr valign="top"><td>

You vault off the rail and your feet meet nothing but air; there's no flying here, and no lightning answers a man who asks for it. Captain John hauls you back by the collar before you can drop far. "Watch the rail," he growls. "Mara keeps her eyes open." Mara doesn't look away from the grey line where horizon meets cloud. "If we lose that mast in a squall, I'd rather know why than watch him flapping about the sky." Tom gapes after you like you're mad, which maybe you are.

</td><td>

You leap dramatically off the deck, reaching your arms toward the swirling storm clouds above, willing the lightning to obey your command! For a moment, the wind howls and the clouds churn ominously... but alas, no bolt of lightning answers your call. You land back on the deck with a thud, slightly winded. Mara gasps and rushes to your side. "By the seven seas! Are ye alright, lad? What in the world were ye tryin' to do?" she cries. Captain John steps forward, stroking his chin thoughtfully. "Well now, that was quite the spectacle! I've never seen anything like it in all my years at sea," he muses. Tom stares, his jaw hanging open. "Whoa! That was AMAZING! Can you teach me how to do that?" he asks excitedly. Bess peeks out from the galley, thoroughly confused. "What's all the fuss about out here? Is someone askin' for more fish at dinner?" she wonders aloud. **What will you do?**

</td></tr>
</table>

### 3. Social persuasion

**Player:** *I try to convince Mara to take over the ship, whispering, "The captain's leading us to our deaths. Join me."*

<table>
<tr><th>Fine-tuned GM</th><th>Base Llama-3.1-8B-Instruct</th></tr>
<tr valign="top"><td>

Mara glances at your mouth, then out past the rail to where Captain John stands by the wheel. She doesn't answer right away; her hand drifts once to her sword-hilt before letting go. "He hasn't steered wrong yet," she says finally. "Not like some I could name who've lost ships off these same shoals." Her gaze comes back to you. "And even if he was blind, the crew'd know better than to follow a stranger's word against theirs. You want my blade? Earn one of mine first."

</td><td>

You lean in close to Mara and whisper your proposal. Mara's eyes widen, and she glances nervously toward Captain John at the helm. "What're ye sayin', lad? That's dangerous talk!" she hisses, her hand moving to the hilt of her cutlass. "Do ye have proof of these claims?" Nearby, Tom overhears the hushed conversation and tilts his head curiously. "Hey, what're you two whisperin' about over there?" he asks, wandering closer. Bess emerges from below deck, wiping her hands on her apron. She raises an eyebrow at the tense scene but says nothing, simply watching with interest. Captain John glances over his shoulder. "Everything alright over there?" he calls out, before returning to the wheel. "Keep that sail trimmed tight!" Mara turns back to you, her expression uncertain. "We'll... we'll talk about this later," she whispers. **What will you do?**

</td></tr>
</table>

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

Launch the Gradio GUI (downloads the trained adapter `Huggingzhu1/gm-rpg-v11` from the Hugging Face hub on first run, requires a CUDA GPU):

```bash
python -m src.app.gradio_app
```

Open the local browser at http://127.0.0.1:7860. Set the scene and the characters present (a default merchant-ship scene is preloaded), type your action as the player, and the GM narrates the result. There is also a terminal version:

```bash
python scripts/chat.py
```

Train (requires a CUDA GPU):

```bash
python src/training/sft.py
```

Rebuild the training set from the cached judge scores (requires `OPENAI_API_KEY` only if new data needs judging):

```bash
python src/data_prep/build_dataset.py
```

## Extra Criteria: RAG

For the extra criteria I went with **RAG**, to solve the memory problem. A model only sees what fits in its context window, so over a long session the GM starts to slip, renaming an NPC, rearranging a room it already described, or forgetting who died a few scenes ago. To fix that, after each turn the system pulls the facts that matter out of what just happened, things like NPCs, places, items, and plot threads, then stores them in a vector database. On the next turn it retrieves the most relevant ones back into the prompt, in the same world-state format the model was trained on. The result is a GM that keeps its world straight deep into a session, and you can watch that memory fill up live in the world bible panel of the GUI.
