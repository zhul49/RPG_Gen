import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def entity_key(fact):
    """Derive a stable id from a 'Name — description' fact: the normalized
    entity name (left of the dash). Same entity -> same key -> upsert
    overwrites the prior fact about it, instead of appending a duplicate."""
    key = fact
    for sep in (" — ", "—", " - "):
        if sep in fact:
            key = fact.split(sep, 1)[0]
            break
    key = re.sub(r"\s+", " ", key.strip().lower())
    key = re.sub(r"^(the|a|an)\s+", "", key)  # "The ring"/"Ring" -> "ring"
    return key or fact.strip().lower()

_RELATION = re.compile(
    r"\b(sister|brother|daughter|son|father|mother|wife|husband|widow|widower|"
    r"lover|betrothed|fianc[eé]e?|cousin|aunt|uncle|niece|nephew|parent|child|"
    r"sibling|spouse|married|mistress|stepfather|stepmother|grandfather|"
    r"grandmother)\b",
    re.I,
)


def _description(fact):
    """The part right of the dash (the claim), lowercased; '' if no dash."""
    for sep in (" — ", "—", " - "):
        if sep in fact:
            return fact.split(sep, 1)[1].strip().lower()
    return ""


def relation_tokens(fact):
    """Set of kinship/identity tokens asserted in a fact's description.
    'Selia — Tomas's sister, brown hair' -> {'sister'}. Empty if none."""
    return frozenset(m.group(1).lower() for m in _RELATION.finditer(_description(fact)))


class WorldStore:
    def __init__(self, db_path, collection_name="session",
                 embedding_model="all-MiniLM-L6-v2"):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self.client = chromadb.PersistentClient(path=self.db_path)
        ef = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
        )
        # Relationship-contradiction rejections, for /facts inspection + debugging.
        self.conflicts = []

    def add(self, facts, source_turn=None):
        """facts: list[str]. Upserts keyed by entity name. Latest-wins for
        volatile state, but CANON-PINNED for relationships: the first kinship a
        fact asserts for an entity is recorded, and any later fact that asserts a
        DIFFERENT kinship for that same entity is rejected (the canon fact is
        kept) so a confabulated reveal can't overwrite established canon. The
        rejected conflicts are recorded on self.conflicts for inspection.
        No-op on empty list."""
        if not facts:
            return
        st = int(source_turn) if source_turn is not None else -1
        # Dedup within this batch (last mention of an entity wins).
        latest = {}
        for f in facts:
            latest[entity_key(f)] = f

        # Pull existing docs+metas for these entities to apply canon-pinning.
        keys = list(latest.keys())
        existing = self.collection.get(ids=keys)
        cur_doc = dict(zip(existing.get("ids", []), existing.get("documents", [])))
        cur_meta = dict(zip(existing.get("ids", []), existing.get("metadatas", [])))

        ids, docs, metas = [], [], []
        for key, new_doc in latest.items():
            old_meta = cur_meta.get(key) or {}
            # canon_rel: comma-joined kinship tokens pinned at first assertion.
            canon = old_meta.get("canon_rel") or ""
            canon_set = frozenset(t for t in canon.split(",") if t)
            new_set = relation_tokens(new_doc)

            if canon_set and new_set and new_set != canon_set:
                # Contradicts pinned relationship -> reject, keep canon fact.
                self.conflicts.append({
                    "entity": key, "kept": cur_doc.get(key, ""),
                    "rejected": new_doc, "canon_rel": sorted(canon_set),
                    "new_rel": sorted(new_set), "source_turn": st,
                })
                continue

            # Accept the update. Persist (or first-time set) the canon pin.
            pin = canon or (",".join(sorted(new_set)) if new_set else "")
            ids.append(key)
            docs.append(new_doc)
            meta = {"source_turn": st}
            if pin:
                meta["canon_rel"] = pin
            metas.append(meta)

        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def query(self, text, k=5):
        n = self.collection.count()
        if n == 0:
            return []
        result = self.collection.query(
            query_texts=[text],
            n_results=min(k, n),
        )
        return result["documents"][0] if result["documents"] else []

    def all(self):
        """Return every stored fact (for /facts inspection)."""
        return self.collection.get().get("documents", [])

    def clear(self):
        ids = self.collection.get().get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def count(self):
        return self.collection.count()
