import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def entity_key(fact):
    # turn a "Name description" fact into a stable id from the name part
    key = fact
    for sep in (" — ", "—", " - "):
        if sep in fact:
            key = fact.split(sep, 1)[0]
            break
    key = re.sub(r"\s+", " ", key.strip().lower())
    key = re.sub(r"^(the|a|an)\s+", "", key)  # "The ring" to "ring"
    if key:
        return key
    return fact.strip().lower()


_RELATION = re.compile(
    r"\b(sister|brother|daughter|son|father|mother|wife|husband|widow|widower|"
    r"lover|betrothed|fianc[eé]e?|cousin|aunt|uncle|niece|nephew|parent|child|"
    r"sibling|spouse|married|mistress|stepfather|stepmother|grandfather|"
    r"grandmother)\b",
    re.I,
)


def _description(fact):
    # the part right of the dash lowercased
    for sep in (" — ", "—", " - "):
        if sep in fact:
            return fact.split(sep, 1)[1].strip().lower()
    return ""


def relation_tokens(fact):
    # find the kinship words in a fact's description
    desc = _description(fact)
    tokens = set()
    for m in _RELATION.finditer(desc):
        tokens.add(m.group(1).lower())
    return frozenset(tokens)


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
        # rejected relationship contradictions
        self.conflicts = []

    def add(self, facts, source_turn=None):
        # save facts, one per entity name. newest wins for normal facts
        if not facts:
            return
        if source_turn is not None:
            st = int(source_turn)
        else:
            st = -1

        # if an entity shows up twice in this batch, keep the last one
        latest = {}
        for f in facts:
            latest[entity_key(f)] = f

        # look up what we already have stored for these entities
        keys = list(latest.keys())
        existing = self.collection.get(ids=keys)
        ex_ids = existing.get("ids", [])
        ex_docs = existing.get("documents", [])
        ex_metas = existing.get("metadatas", [])
        cur_doc = {}
        cur_meta = {}
        for i in range(len(ex_ids)):
            cur_doc[ex_ids[i]] = ex_docs[i]
            cur_meta[ex_ids[i]] = ex_metas[i]

        ids, docs, metas = [], [], []
        for key, new_doc in latest.items():
            old_meta = cur_meta.get(key) or {}
            # canon_rel is the kinship words we pinned the first time around
            canon = old_meta.get("canon_rel") or ""
            canon_set = set()
            for t in canon.split(","):
                if t:
                    canon_set.add(t)
            new_set = relation_tokens(new_doc)

            if canon_set and new_set and new_set != canon_set:
                # this contradicts the pinned relationship, so reject it
                self.conflicts.append({
                    "entity": key, "kept": cur_doc.get(key, ""),
                    "rejected": new_doc, "canon_rel": sorted(canon_set),
                    "new_rel": sorted(new_set), "source_turn": st,
                })
                continue

            # accept this fact, and work out the canon pin to store with it
            if canon:
                pin = canon
            elif new_set:
                pin = ",".join(sorted(new_set))
            else:
                pin = ""
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
        if result["documents"]:
            return result["documents"][0]
        return []

    def all(self):
        # return every stored fact
        data = self.collection.get()
        return data.get("documents", [])

    def clear(self):
        data = self.collection.get()
        ids = data.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def count(self):
        return self.collection.count()
