from pathlib import Path

# repo root, so the model and database paths below are absolute
REPO_ROOT = Path(__file__).resolve().parent.parent

# the scene and cast the chat script and the web app both start on
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

# load the trained adapter from the Hugging Face hub instead of a local folder
ADAPTER = "Huggingzhu1/gm-rpg-v11"
DB_PATH = REPO_ROOT / "data" / "world_state" / "chroma"
PORT = 7860

# generation settings
TEMPERATURE = 0.6
REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS = 300

# how many past turns and memory facts to feed the model each turn
HISTORY_TURNS = 20
RAG_K = 20
