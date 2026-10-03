"""Request-time Complete the Look over a versioned serving bundle (no LightGBM, no DuckDB).

The online path for one request (DESIGN §5, D-041, D-042):

1. resolve the anchor (unknown article -> error; known but not live -> fallback ladder);
2. read the anchor's precomputed compatibility pool for the target slot;
3. drop candidates that are unavailable (catalogue snapshot + runtime overrides) *before* top-k;
4. if the customer has a profile, compute the 21 point-in-time personalization features and
   re-order the pool with the personalized ranker (NumPy GBDT runtime);
5. apply the diversity rules as a re-ordering with back-fill (D-035), back-fill short modules
   from slot popularity;
6. attach per-row evidence, provenance and evidence-gated reason chips.

Fallback ladder (``fallback_level`` = where the candidates came from; ``personalized`` = how
they were ordered): ``anchor_pool`` -> ``style_sibling`` (a live colourway of the same style) ->
``visual_neighbor`` (nearest live article of the same slot in FashionCLIP space) ->
``slot_popularity``. Every level is deterministic: ties break on ``article_id``.

The personalization features mirror ``ensemble.completion.features.PERSONAL_SQL`` exactly,
including its NULL semantics; ``ensemble.serving.bundle`` refuses to publish a bundle whose
online scores disagree with the offline SQL + LightGBM path.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble.completion.fusion import apply_diversity
from ensemble.serving.gbdt import Forest

SLOTS = ["upper", "lower", "full", "shoes", "accessories", "socks", "swimwear"]
SLOT_TITLES = {"upper": "Tops", "lower": "Bottoms", "full": "Dresses & jumpsuits", "shoes": "Shoes",
               "accessories": "Accessories", "socks": "Socks & tights", "swimwear": "Swimwear"}
PERSONAL = ["c_has_history", "c_n_tx", "c_n_days", "c_days_since_last", "c_mean_price", "cu_article_n",
            "cu_article_days_ago", "cu_style_n", "cu_type_share", "cu_dept_share", "cu_section_share",
            "cu_ggroup_share", "cu_colour_share", "cu_cmaster_share", "cu_slot_share", "cu_type_n", "cu_dept_n",
            "cu_price_dist", "x_npmi_type", "x_tt_type", "x_co_dept"]
AUX = ["a_npmi_exact", "a_co_exact", "tt_pair_sim_exact"]
# Profile dimensions: (blob key, catalogue column the candidate is matched on).
SHARE_DIMS = {"type": "product_type_no", "dept": "department_no", "section": "section_no",
              "ggroup": "garment_group_no", "colour": "colour_group_code", "cmaster": "perceived_colour_master_id"}
LIFT_MIN_SUPPORT = 5          # a lift multiplier off fewer co-purchases is a small-sample artefact (D-035)
REQUIRED_FILES = ("manifest.json", "keys.npz", "pool_article.npy", "pool_compat.npy", "pool_feat.npy",
                  "pool_aux.npy", "catalog.parquet", "slot_popularity.json", "profiles.sqlite",
                  "model_personalized.json", "feature_schema.json")
BUNDLE_FORMAT = 1


class BundleError(RuntimeError):
    """A serving bundle is missing, corrupt or incompatible; the service must not become ready."""


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def encode_profile(p: dict) -> bytes:
    return zlib.compress(json.dumps(p, separators=(",", ":")).encode(), 6)


def decode_profile(blob: bytes) -> dict:
    p = json.loads(zlib.decompress(blob))
    out = {k: p[k] for k in ("n", "d", "l", "p", "w")}
    for dim in (*SHARE_DIMS, "slot"):
        out[dim] = {e["k"]: (e["n"], e["w"]) for e in p.get(dim) or []}
    out["art"] = {int(e["k"]): (e["n"], e["d"]) for e in p.get("art") or []}
    out["style"] = {int(e["k"]): e["n"] for e in p.get("style") or []}
    return out


def personal_matrix(profile: dict | None, cand: dict[str, np.ndarray], target_slot: str,
                    aux: np.ndarray) -> np.ndarray:
    """The 21 personalization features for each candidate, in ``PERSONAL`` order (float32).

    ``cand`` holds the candidates' catalogue attributes (arrays aligned with ``aux`` rows);
    ``aux`` holds the exact (float64) association NPMI, co-count and two-tower similarity."""
    n = len(aux)
    out = np.full((n, len(PERSONAL)), np.nan, dtype=np.float64)
    col = {f: i for i, f in enumerate(PERSONAL)}
    npmi = np.nan_to_num(aux[:, 0], nan=0.0)
    co = np.nan_to_num(aux[:, 1], nan=0.0)
    tt = np.nan_to_num(aux[:, 2], nan=0.0)
    if profile is None:
        out[:, col["c_has_history"]] = 0.0
        out[:, col["x_npmi_type"]] = npmi * 0.0
        out[:, col["x_tt_type"]] = tt * 0.0
        out[:, col["x_co_dept"]] = co * 0.0
        return out.astype(np.float32)
    wt = float(profile["w"])
    out[:, col["c_has_history"]] = 1.0
    out[:, col["c_n_tx"]] = profile["n"]
    out[:, col["c_n_days"]] = profile["d"]
    out[:, col["c_days_since_last"]] = profile["l"]
    out[:, col["c_mean_price"]] = profile["p"]
    arts = cand["article_id"]
    for r in range(n):
        a = profile["art"].get(int(arts[r]))
        if a is not None:
            out[r, col["cu_article_n"]], out[r, col["cu_article_days_ago"]] = a
        s = profile["style"].get(int(cand["product_code"][r]))
        if s is not None:
            out[r, col["cu_style_n"]] = s
    shares = {}
    for dim, attr in SHARE_DIMS.items():
        vals = cand[attr]
        table = profile[dim]
        sh = np.full(n, np.nan)
        cnt = np.full(n, np.nan)
        for r in range(n):
            v = vals[r]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            e = table.get(int(v))
            if e is not None:
                cnt[r], sh[r] = e[0], e[1] / wt
        shares[dim] = sh
        out[:, col[f"cu_{dim}_share"]] = sh
        if dim in ("type", "dept"):
            out[:, col[f"cu_{dim}_n"]] = cnt
    e = profile["slot"].get(target_slot)
    out[:, col["cu_slot_share"]] = (e[1] / wt) if e is not None else np.nan
    cp = cand["cp_mean_price"].astype(np.float64)
    mp = float(profile["p"])
    out[:, col["cu_price_dist"]] = np.abs(cp - mp) / mp if mp != 0 else np.nan
    t_sh = np.nan_to_num(shares["type"], nan=0.0)
    d_sh = np.nan_to_num(shares["dept"], nan=0.0)
    out[:, col["x_npmi_type"]] = npmi * t_sh
    out[:, col["x_tt_type"]] = tt * t_sh
    out[:, col["x_co_dept"]] = co * d_sh
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def resolve(root: Path) -> Path:
    """A bundle directory, or a bundles root whose ``CURRENT`` file names the active version."""
    root = Path(root)
    if (root / "manifest.json").exists():
        return root
    cur = root / "CURRENT"
    if not cur.exists():
        raise BundleError(f"no serving bundle at {root}: expected manifest.json or a CURRENT pointer. "
                          f"Build one with `make serving-bundle` (needs the ingested H&M data).")
    return root / cur.read_text().strip()


class Bundle:
    """Every artifact the online path needs, loaded once, validated, read-only."""

    def __init__(self, path: Path, verify_hashes: bool = False):
        t0 = time.time()
        self.path = resolve(path)
        missing = [f for f in REQUIRED_FILES if not (self.path / f).exists()]
        if missing:
            raise BundleError(f"bundle {self.path} is missing {missing}")
        try:
            self.manifest = json.loads((self.path / "manifest.json").read_text())
        except json.JSONDecodeError as e:
            raise BundleError(f"bundle manifest is corrupt: {e}") from e
        if self.manifest.get("format") != BUNDLE_FORMAT:
            raise BundleError(f"bundle format {self.manifest.get('format')} != supported {BUNDLE_FORMAT}")
        for name, meta in self.manifest["files"].items():
            p = self.path / name
            if not p.exists() or p.stat().st_size != meta["bytes"]:
                raise BundleError(f"bundle file {name} is missing or has the wrong size")
            if verify_hashes and sha256_file(p) != meta["sha256"]:
                raise BundleError(f"bundle file {name} fails its sha256 check")
        self.schema = json.loads((self.path / "feature_schema.json").read_text())
        schema_sha = hashlib.sha1(json.dumps(self.schema, sort_keys=True).encode()).hexdigest()[:12]
        if schema_sha != self.manifest["feature_schema_sha1"]:
            raise BundleError("feature schema hash does not match the manifest")
        try:
            self.forest = Forest.load(self.path / "model_personalized.json")
        except (OSError, ValueError, KeyError) as e:
            raise BundleError(f"personalized model failed to load: {e}") from e
        if self.forest.feature_names != self.schema["personalized_features"]:
            raise BundleError("personalized model features differ from the bundle schema")
        self.pool_cols = [self.forest.feature_names.index(f) for f in self.schema["pool_features"]]
        self.personal_cols = [self.forest.feature_names.index(f) for f in PERSONAL]
        k = np.load(self.path / "keys.npz")
        self.key_anchor, self.key_slot = k["anchor"], k["slot_id"]
        self.key_offset, self.key_count = k["offset"], k["count"]
        self.pool_article = np.load(self.path / "pool_article.npy", mmap_mode="r")
        self.pool_compat = np.load(self.path / "pool_compat.npy", mmap_mode="r")
        self.pool_feat = np.load(self.path / "pool_feat.npy", mmap_mode="r")
        self.pool_aux = np.load(self.path / "pool_aux.npy", mmap_mode="r")
        if self.pool_feat.shape != (len(self.pool_article), len(self.schema["pool_features"])):
            raise BundleError("pool feature matrix shape does not match the schema")
        cat = pd.read_parquet(self.path / "catalog.parquet")
        self.catalog = cat.set_index("article_id", drop=False)
        self.live = set(cat.article_id[cat.live].astype(int))
        self.slot_pop = {s: [int(a) for a in v] for s, v in
                         json.loads((self.path / "slot_popularity.json").read_text()).items()}
        self.by_code: dict[int, list[int]] = {}
        for a, c in zip(cat.article_id[cat.live].astype(int), cat.product_code[cat.live].astype(int)):
            self.by_code.setdefault(c, []).append(a)
        self.profiles_path = self.path / "profiles.sqlite"
        self._local = threading.local()
        self.visual = self._load_visual()
        self.version = self.manifest["bundle_version"]
        self.model_version = self.manifest["model_version"]
        self.catalog_version = self.manifest["catalog_version"]
        self.profile_version = self.manifest["profile_version"]
        self.load_seconds = round(time.time() - t0, 3)

    def _load_visual(self):
        d = self.path / "visual"
        if not (d / "live_ids.npy").exists():
            return None
        ids = np.load(d / "live_ids.npy")
        emb = np.load(d / "clip.npy").astype(np.float32)
        adapted = (np.load(d / "clip_shop_adapted.npy").astype(np.float32)
                   if (d / "clip_shop_adapted.npy").exists() else None)
        # Every catalogue article with an image (float16, memory-mapped): lets a non-live anchor find
        # its nearest live neighbour; only live articles are ever searched or recommended.
        all_ids = np.load(d / "all_ids.npy") if (d / "all_ids.npy").exists() else ids
        all_emb = np.load(d / "all_clip.npy", mmap_mode="r") if (d / "all_clip.npy").exists() else emb
        return {"ids": ids, "emb": emb, "adapted": adapted, "all_ids": all_ids, "all_emb": all_emb}

    def visual_vector(self, article_id: int) -> np.ndarray | None:
        v = self.visual
        if v is None:
            return None
        i = int(np.searchsorted(v["all_ids"], article_id))
        if i >= len(v["all_ids"]) or v["all_ids"][i] != article_id:
            return None
        return np.asarray(v["all_emb"][i], dtype=np.float32)

    # -- lookups -------------------------------------------------------------
    def profile(self, customer_idx: int) -> dict | None:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.profiles_path}?mode=ro", uri=True, check_same_thread=False)
            self._local.con = con
        row = con.execute("SELECT blob FROM profiles WHERE customer_idx = ?", (int(customer_idx),)).fetchone()
        return decode_profile(row[0]) if row else None

    def pool(self, anchor: int, slot: str) -> slice | None:
        sid = SLOTS.index(slot)
        i = int(np.searchsorted(self.key_anchor, anchor))
        while i < len(self.key_anchor) and self.key_anchor[i] == anchor:
            if self.key_slot[i] == sid:
                o = int(self.key_offset[i])
                return slice(o, o + int(self.key_count[i]))
            i += 1
        return None

    def has_any_pool(self, anchor: int) -> bool:
        i = int(np.searchsorted(self.key_anchor, anchor))
        return i < len(self.key_anchor) and self.key_anchor[i] == anchor


# ---------------------------------------------------------------------------
# Reasons and provenance
# ---------------------------------------------------------------------------

def _num(v):
    return float(v) if v is not None and v == v else None


def evidence_of(ev: dict) -> list[str]:
    out = []
    if (_num(ev.get("a_co")) or 0) >= 1:
        out.append("co_purchase")
    if (_num(ev.get("s_co")) or 0) >= 1:
        out.append("style_co_purchase")
    if ev.get("src_two_tower") == 1 or ev.get("src_two_tower_content") == 1:
        out.append("visual_compatibility")
    if ev.get("src_slot_pop") == 1:
        out.append("popular_in_slot")
    return out


def reasons_for(ev: dict, evidence: list[str], personal: dict | None, type_name: str | None) -> list[dict]:
    """Reason chips, each emitted only when the column it quotes supports it for *this* row."""
    out = []
    co, lift, s_co = _num(ev.get("a_co")), _num(ev.get("a_lift")), _num(ev.get("s_co"))
    if "co_purchase" in evidence:
        text = f"Bought together {int(co)}× in past baskets"
        if lift and co >= LIFT_MIN_SUPPORT:
            text += f", {lift:.1f}× more often than chance"
        out.append({"key": "co_purchase", "evidence_type": "co_purchase", "text": text})
    elif "style_co_purchase" in evidence:
        out.append({"key": "style", "evidence_type": "style_co_purchase",
                    "text": f"Bought with this style {int(s_co)}× in past baskets"})
    elif "visual_compatibility" in evidence:
        out.append({"key": "visual", "evidence_type": "visual_compatibility", "text": "Visually matches this piece"})
    elif "popular_in_slot" in evidence:
        out.append({"key": "popular", "evidence_type": "popular_in_slot", "text": "Popular pick for this category"})
    if personal:
        if (personal.get("cu_style_n") or 0) >= 1:
            out.append({"key": "your_style", "evidence_type": "personal_history",
                        "text": "From a style you bought before"})
        elif (personal.get("cu_type_n") or 0) >= 2 and type_name:
            out.append({"key": "your_type", "evidence_type": "personal_history",
                        "text": f"You often buy {type_name.lower()}"})
    if _num(ev.get("same_colour_master")) == 1:
        out.append({"key": "colour", "evidence_type": "attribute_match", "text": "Same colour family"})
    if _num(ev.get("price_tier_diff")) == 0:
        out.append({"key": "price", "evidence_type": "attribute_match", "text": "Same price range"})
    return out[:3]


EVIDENCE_COLS = ("a_co", "a_lift", "a_npmi", "s_co", "s_lift", "backoff_level", "src_two_tower",
                 "src_two_tower_content", "src_slot_pop", "tt_pair_sim", "clip_sim", "same_colour_master",
                 "price_tier_diff")


# ---------------------------------------------------------------------------
# The recommender
# ---------------------------------------------------------------------------

class LRU:
    def __init__(self, size: int):
        self.size, self.d, self.lock = size, OrderedDict(), threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                self.hits += 1
                return self.d[key]
            self.misses += 1
            return None

    def put(self, key, value):
        if self.size <= 0:
            return
        with self.lock:
            self.d[key] = value
            self.d.move_to_end(key)
            while len(self.d) > self.size:
                self.d.popitem(last=False)

    def clear(self):
        with self.lock:
            self.d.clear()


class RequestError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.status = code, status


class Recommender:
    def __init__(self, bundle: Bundle, cache_size: int = 20000, per_slot: int = 8, max_per_product_type: int = 2,
                 personalization: bool = True):
        self.bundle = bundle
        self.cache = LRU(cache_size)
        self.per_slot, self.max_per_type = per_slot, max_per_product_type
        self.personalization = personalization
        self.unavailable: set[int] = set()
        self._avail_lock = threading.Lock()
        cat = bundle.catalog
        self.ptype = dict(zip(cat.article_id.astype(int), cat.product_type_no.astype(int)))
        self.pcode = dict(zip(cat.article_id.astype(int), cat.product_code.astype(int)))
        self._warm_visual_masks()

    @property
    def availability_version(self) -> str:
        body = ",".join(map(str, sorted(self.unavailable)))
        return f"{self.bundle.catalog_version}+{hashlib.sha1(body.encode()).hexdigest()[:8]}"

    def set_unavailable(self, ids) -> str:
        with self._avail_lock:
            self.unavailable = {int(a) for a in ids}
        return self.availability_version

    def available(self, a: int) -> bool:
        return a in self.bundle.live and a not in self.unavailable

    # -- request validation --------------------------------------------------
    def target_slots(self, anchor: int, slots: list[str] | None) -> list[str]:
        cat = self.bundle.catalog
        if anchor not in cat.index:
            raise RequestError("unknown_article", f"article {anchor} is not in the catalogue", 404)
        a_slot = cat.at[anchor, "slot"]
        if a_slot is None or a_slot != a_slot:
            raise RequestError("anchor_not_wearable", f"article {anchor} has no outfit slot", 422)
        if slots:
            bad = [s for s in slots if s not in SLOTS]
            if bad:
                raise RequestError("invalid_slot", f"unknown slot(s) {bad}; expected one of {SLOTS}")
            same = [s for s in slots if s == a_slot]
            if same:
                raise RequestError("invalid_slot", f"slot {same[0]} is the anchor's own slot")
            return list(dict.fromkeys(slots))
        return [s for s in SLOTS if s != a_slot]

    # -- candidate sources (the fallback ladder) -----------------------------
    def _source(self, anchor: int, slot: str):
        b = self.bundle
        sl = b.pool(anchor, slot)
        if sl is not None:
            return "anchor_pool", anchor, sl
        code = int(b.catalog.at[anchor, "product_code"])
        sibs = [a for a in b.by_code.get(code, []) if a != anchor]
        for s in sorted(sibs, key=lambda a: (-float(b.catalog.at[a, "pop_recent"]), a)):
            sl = b.pool(s, slot)
            if sl is not None:
                return "style_sibling", s, sl
        v = b.visual_vector(anchor)
        if v is not None:
            ok = self._vis_ok.get(b.catalog.at[anchor, "slot"])
            if ok is not None and ok.any():
                sims = b.visual["emb"][ok] @ v                 # exact cosine (rows are L2-normalised)
                cand = b.visual["ids"][ok]
                for j in np.lexsort((cand, -sims))[:20]:
                    if int(cand[j]) == anchor:
                        continue
                    sl = b.pool(int(cand[j]), slot)
                    if sl is not None:
                        return "visual_neighbor", int(cand[j]), sl
        return "slot_popularity", None, None

    def _warm_visual_masks(self) -> None:
        """Per anchor slot, which visual-index articles own a pool (computed once at startup)."""
        b = self.bundle
        self._vis_ok = {}
        if b.visual is None:
            return
        ids = b.visual["ids"]
        slot = b.catalog.slot.reindex(ids).to_numpy()
        has = np.isin(ids, b.key_anchor)
        self._vis_ok = {s: (slot == s) & has for s in SLOTS}

    # -- one module ----------------------------------------------------------
    def module(self, anchor: int, slot: str, customer_idx: int | None, k: int, diversity: bool = True) -> dict:
        b = self.bundle
        profile = b.profile(customer_idx) if (customer_idx is not None and self.personalization) else None
        personalized = profile is not None
        ckey = (b.version, self.availability_version, anchor, slot, customer_idx if personalized else None, k,
                diversity)
        hit = self.cache.get(ckey)
        if hit is not None:
            return {**hit, "cache": "hit"}
        level, source_anchor, sl = self._source(anchor, slot)
        rows: list[dict] = []
        if sl is not None:
            arts = np.asarray(b.pool_article[sl], dtype=np.int64)
            keep = np.array([self.available(int(a)) for a in arts], dtype=bool)
            idx = np.flatnonzero(keep)
            arts = arts[idx]
            feat = np.asarray(b.pool_feat[sl][idx], dtype=np.float32)
            aux = np.asarray(b.pool_aux[sl][idx], dtype=np.float64)
            compat = np.asarray(b.pool_compat[sl][idx], dtype=np.float64)
            pers_rows = None
            if personalized and len(arts):
                cat = b.catalog.loc[arts]
                cand = {c: cat[c].to_numpy() for c in ("product_code", "product_type_no", "department_no",
                                                        "section_no", "garment_group_no", "colour_group_code",
                                                        "perceived_colour_master_id", "cp_mean_price")}
                cand["article_id"] = arts
                P = personal_matrix(profile, cand, slot, aux)
                X = np.empty((len(arts), len(b.forest.feature_names)), dtype=np.float32)
                X[:, b.pool_cols] = feat
                X[:, b.personal_cols] = P
                score = b.forest.predict(X)
                pers_rows = P
            else:
                score = compat
            order = np.lexsort((arts, -score))
            ranked = arts[order]
            chosen = apply_diversity(ranked, k, self.ptype, self.pcode, max_per_product_type=self.max_per_type,
                                     one_per_product_code=True, fill_back=True) if diversity else ranked[:k]
            pos = {int(a): int(i) for i, a in enumerate(arts)}
            fcol = {f: i for i, f in enumerate(b.schema["pool_features"])}
            for a in chosen:
                i = pos[int(a)]
                ev = {c: (float(feat[i, fcol[c]]) if c in fcol else None) for c in EVIDENCE_COLS}
                rows.append({"article_id": int(a), "score": float(score[i]), "compat_score": float(compat[i]),
                             "evidence": ev,
                             "personal": ({f: (None if np.isnan(pers_rows[i, j]) else float(pers_rows[i, j]))
                                           for j, f in enumerate(PERSONAL)} if pers_rows is not None else None)})
        # Back-fill a short module (thin pool, unavailable items) from slot popularity.
        if len(rows) < k:
            have = {r["article_id"] for r in rows}
            for a in b.slot_pop.get(slot, []):
                if len(rows) >= k:
                    break
                if a not in have and a != anchor and self.available(a):
                    rows.append({"article_id": a, "score": None, "compat_score": None,
                                 "evidence": {"src_slot_pop": 1}, "personal": None, "backfill": True})
                    have.add(a)
        items = []
        for rank, r in enumerate(rows, 1):
            ev = evidence_of(r["evidence"])
            tname = b.catalog.at[r["article_id"], "product_type_name"]
            rs = reasons_for(r["evidence"], ev, r["personal"], tname)
            if r["personal"] and any(x["evidence_type"] == "personal_history" for x in rs):
                ev = ev + ["personal_history"]
            items.append({"article_id": r["article_id"], "rank": rank, "target_slot": slot,
                          "score": r["score"], "provenance": ev[0] if ev else "other", "evidence_types": ev,
                          "reasons": rs, "backfill": bool(r.get("backfill")),
                          "evidence": {kk: v for kk, v in r["evidence"].items() if v is not None and v == v}})
        out = {"slot": slot, "title": SLOT_TITLES[slot], "fallback_level": level, "source_anchor": source_anchor, "personalized": bool(personalized and sl is not None),
               "items": items, "n_pool": int(0 if sl is None else sl.stop - sl.start)}
        self.cache.put(ckey, out)
        return {**out, "cache": "miss"}

    def complete_the_look(self, anchor: int, slots: list[str] | None, customer_idx: int | None, k: int,
                          diversity: bool = True) -> dict:
        if not 1 <= k <= 24:
            raise RequestError("invalid_k", "k must be between 1 and 24")
        targets = self.target_slots(anchor, slots)
        b = self.bundle
        has_profile = customer_idx is not None and self.personalization and b.profile(customer_idx) is not None
        modules = [self.module(anchor, s, customer_idx, k, diversity) for s in targets]
        return {"anchor": anchor, "anchor_slot": b.catalog.at[anchor, "slot"], "customer_idx": customer_idx,
                "customer_known": has_profile, "personalized": any(m["personalized"] for m in modules),
                "modules": modules, "model_version": b.model_version, "catalog_version": b.catalog_version,
                "profile_version": b.profile_version, "availability_version": self.availability_version,
                "bundle_version": b.version}
