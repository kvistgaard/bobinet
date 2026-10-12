#!/usr/bin/env python3
"""bobinet - construct an RDF knowledge graph from a text document.

Pipeline (RDF adaptation of Anthropic's knowledge-graph cookbook):
  chunk     split a markdown document into section chunks
  extract   closed-vocabulary entity/relation extraction (claude CLI, Haiku)
  resolve   cross-chunk entity resolution into canonical entities (Sonnet)
  link      entity linking to DBpedia (Lookup API) + Wikidata sameAs
  assemble  build TriG: entity graph, one named graph per claim, PROV graph
  zoom      unfold influence claims into an idea-level layer, written as
            derived <name>-zoom files (fractalino zoom vocabulary; not in `all`)
  validate  SHACL + vocabulary dereference check + evidence re-check + DBpedia reconciliation
  query     run the competency queries in queries/*.rq
  ask       answer a question grounded in the graph (Sonnet)
  all       chunk .. query in sequence

Example:
  python bobinet.py --src note.md --out out all

Requires: rdflib, pyshacl, the `claude` CLI on PATH, network access for
link/validate. See README.md.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

__version__ = "0.2.0"

EXTRACTION_MODEL = "claude-haiku-4-5-20251001"
SYNTHESIS_MODEL = "claude-sonnet-5"

# The tool's own vocabulary and application profile. Fixed rather than derived
# from --base: shapes.ttl and queries/*.rq name kgx: terms, so those terms must
# mean the same thing whatever base a given run writes its data under.
NS_VOCAB = "https://personalknowledgegraphs.com/def/bobinet#"
PROFILE_IRI = "https://personalknowledgegraphs.com/def/bobinet"

DEFAULT_BASE = "https://personalknowledgegraphs.com/data/bobinet/"
VOCAB_CACHE = HERE / "vocab-cache"


class Names:
    """The IRI namespaces one run writes its data under, from --base."""

    def __init__(self, base: str) -> None:
        self.base = base if base.endswith(("/", "#")) else base + "/"
        self.id = self.base + "id/"
        self.claim = self.base + "claim/"
        self.graph = self.base + "graph/"
        self.doc = self.base + "doc/"
        self.activity = self.base + "activity/"
        self.agent = self.base + "agent/"


def names(args) -> Names:
    return Names(getattr(args, "base", "") or DEFAULT_BASE)

TYPE_PREFIX = {
    "PERSON": "person", "ORGANIZATION": "org", "WORK": "work",
    "CONCEPT": "concept", "EVENT": "event", "PLACE": "place",
}

# ---------------------------------------------------------------- helpers

def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or "x"


PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
    "\u2026": "...",
})


def norm_text(s: str) -> str:
    s = s.translate(PUNCT_MAP)
    s = re.sub(r"[*_`\[\]]", "", s)  # markdown emphasis/link markers: models
    s = re.sub(r"\s+", " ", s)       # rightly strip them when quoting
    return s.strip()


def call_claude(prompt: str, model: str, timeout: int = 600, attempts: int = 3) -> str:
    exe = shutil.which("claude")
    if not exe:
        sys.exit("claude CLI not found on PATH")
    cmd = [exe, "-p", "--model", model]
    if exe.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c"] + cmd
    last = None
    for attempt in range(attempts):
        if attempt:
            # transient CLI failures happen (e.g. the binary is briefly
            # replaced during an auto-update); back off and retry
            time.sleep(10 * attempt)
        try:
            r = subprocess.run(cmd, input=prompt.encode("utf-8"),
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = f"timeout after {timeout}s"
            print(f"  claude call timed out (attempt {attempt + 1}/{attempts})")
            continue
        if r.returncode == 0:
            raw = r.stdout or b""
            try:
                return raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                # the CLI occasionally emits console-codepage bytes on Windows
                return raw.decode("cp1252", errors="replace").strip()
        last = f"claude exited {r.returncode}: {(r.stderr or b'').decode('utf-8', 'replace')[:400]}"
        print(f"  claude call failed (attempt {attempt + 1}/{attempts}): {last[:120]}")
    raise RuntimeError(last)


def parse_json_reply(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(t[start:end + 1])


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


def _fetch_bytes(url: str, accept: str, timeout: int = 60) -> bytes:
    """GET with urllib; fall back to curl when the local Python trust store
    rejects a certificate the system's CA bundle accepts."""
    req = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": f"bobinet/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.URLError as exc:
        if "CERTIFICATE" not in str(exc).upper():
            raise
        curl = shutil.which("curl")
        if not curl:
            raise
        r = subprocess.run([curl, "-sL", "--fail", "--max-time", str(timeout),
                            "-H", f"Accept: {accept}", url],
                           capture_output=True, timeout=timeout + 10)
        if r.returncode != 0 or not r.stdout:
            raise
        return r.stdout


def http_get_json(url: str, timeout: int = 30, accept: str = "application/json"):
    return json.loads(_fetch_bytes(url, accept, timeout).decode("utf-8"))


def sparql_json(url: str, timeout: int = 30):
    return http_get_json(url, timeout, accept="application/sparql-results+json")


# ---------------------------------------------------------------- chunk

def cmd_chunk(args) -> None:
    if not args.src:
        sys.exit("--src is required: name the markdown document to build the graph from")
    src = Path(args.src)
    text = src.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(## .*)$", text)
    chunks = []
    if parts[0].strip():
        m = re.search(r"(?m)^# (.+)$", parts[0])
        title = m.group(1).strip() if m else src.stem
        chunks.append({"id": "s00", "heading": title, "text": parts[0].strip()})
    for i in range(1, len(parts), 2):
        heading = parts[i].lstrip("# ").strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunks.append({"id": f"s{len(chunks):02d}", "heading": heading,
                       "text": (parts[i] + "\n" + body).strip()})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "chunks.json", {"source": str(src), "chunks": chunks})
    for c in chunks:
        print(f"  {c['id']}  {len(c['text']):>6} chars  {c['heading'][:70]}")


# ---------------------------------------------------------------- extract

def _validate_extraction(data: dict, chunk_text: str):
    """Returns (entities, relations, errors, dropped). Relations get
    evidence_verified set; type-mismatched-but-invertible relations are flipped."""
    errors, dropped = [], []
    entities, seen = [], set()
    for e in data.get("entities", []):
        name, etype = (e.get("name") or "").strip(), (e.get("type") or "").strip()
        if not name or etype not in prompts.ENTITY_TYPES:
            errors.append(f"entity with missing name or bad type: {e!r}")
            continue
        key = (name.lower(), etype)
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "name": name, "type": etype,
            "description": (e.get("description") or "").strip(),
            **{k: str(e[k]).strip() for k in ("birthYear", "deathYear", "pubYear", "workType")
               if e.get(k)},
        })
    by_name = {}
    for e in entities:
        by_name.setdefault(e["name"].lower(), []).append(e["type"])
    norm_chunk = norm_text(chunk_text)
    relations = []
    for r in data.get("relations", []):
        subj, pred, obj = (r.get("subject") or "").strip(), (r.get("predicate") or "").strip(), (r.get("object") or "").strip()
        if pred not in prompts.PREDICATES:
            dropped.append({"reason": f"unknown predicate {pred!r}", "relation": r})
            errors.append(f"unknown predicate: {pred!r}")
            continue
        stypes = by_name.get(subj.lower(), [])
        otypes = by_name.get(obj.lower(), [])
        if not stypes or not otypes:
            dropped.append({"reason": "dangling endpoint (not in entities list)", "relation": r})
            errors.append(f"relation endpoints not in entities list: {subj!r} / {obj!r}")
            continue
        dom, rng, _ = prompts.PREDICATES[pred]
        doms, rngs = dom.split("|"), rng.split("|")
        ok = any(t in doms for t in stypes) and any(t in rngs for t in otypes)
        flipped = False
        if not ok and any(t in doms for t in otypes) and any(t in rngs for t in stypes):
            subj, obj = obj, subj
            ok, flipped = True, True
        if not ok:
            dropped.append({"reason": f"domain/range mismatch for {pred}", "relation": r})
            errors.append(f"{pred}: types {stypes}->{otypes} do not fit {dom}->{rng}")
            continue
        evidence = (r.get("evidence") or "").strip()
        verified = bool(evidence) and norm_text(evidence) in norm_chunk
        if evidence and not verified:
            errors.append(f"evidence not a verbatim substring: {evidence[:80]!r}")
        try:
            conf = float(r.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8
        rel = {"subject": subj, "predicate": pred, "object": obj,
               "evidence": evidence, "confidence": max(0.0, min(1.0, conf)),
               "evidence_verified": verified}
        if flipped:
            rel["normalized"] = "endpoints swapped to fit domain/range"
        if r.get("influenceNote"):
            rel["influenceNote"] = str(r["influenceNote"]).strip()
        relations.append(rel)
    return entities, relations, errors, dropped


def cmd_extract(args) -> None:
    out = Path(args.out)
    chunks = read_json(out / "chunks.json")["chunks"]
    only = set(args.chunks.split(",")) if args.chunks else None
    prior_path = out / "extraction-raw.json"
    prior = {r["chunk_id"]: r for r in read_json(prior_path)} if prior_path.exists() else {}
    if args.force:
        prior = {}
    results = {cid: r for cid, r in prior.items()}

    def flush():
        ordered = [results[c["id"]] for c in chunks if c["id"] in results]
        prior_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    for c in chunks:
        redo = only and c["id"] in only
        if not redo and c["id"] in prior:
            continue  # resume: keep the stored result
        if only and not redo:
            continue
        print(f"[{c['id']}] extracting: {c['heading'][:60]}")
        prompt = prompts.extraction(c["id"], c["heading"], c["text"])
        reply = call_claude(prompt, EXTRACTION_MODEL)
        try:
            data = parse_json_reply(reply)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  parse failed ({exc}); retrying")
            reply = call_claude(prompt + prompts.RETRY_SUFFIX.format(
                errors=f"- reply was not parseable JSON: {exc}"), EXTRACTION_MODEL)
            data = parse_json_reply(reply)
        ents, rels, errors, dropped = _validate_extraction(data, c["text"])
        if errors and not args.no_retry:
            print(f"  {len(errors)} validation error(s); one corrective retry")
            reply = call_claude(prompt + prompts.RETRY_SUFFIX.format(
                errors="\n".join(f"- {e}" for e in errors)), EXTRACTION_MODEL)
            try:
                data2 = parse_json_reply(reply)
                e2, r2, err2, d2 = _validate_extraction(data2, c["text"])
                if len(err2) < len(errors):
                    ents, rels, errors, dropped = e2, r2, err2, d2
            except (ValueError, json.JSONDecodeError):
                print("  retry unparseable; keeping first attempt")
        unverified = sum(1 for r in rels if not r["evidence_verified"])
        print(f"  {len(ents)} entities, {len(rels)} relations "
              f"({unverified} unverified evidence, {len(dropped)} dropped)")
        results[c["id"]] = {"chunk_id": c["id"], "heading": c["heading"],
                            "entities": ents, "relations": rels,
                            "dropped": dropped, "residual_errors": errors}
        flush()  # incremental save: a crash loses one chunk, not the run
    flush()
    print(f"wrote {prior_path}")


# ---------------------------------------------------------------- resolve

def cmd_resolve(args) -> None:
    out = Path(args.out)
    raw = read_json(out / "extraction-raw.json")
    by_type = {}
    for chunk in raw:
        for e in chunk["entities"]:
            rec = by_type.setdefault(e["type"], {}).setdefault(e["name"], {
                "descriptions": [], "attrs": {}, "mentions": 0})
            rec["mentions"] += 1
            if e.get("description"):
                rec["descriptions"].append(e["description"])
            for k in ("birthYear", "deathYear", "pubYear", "workType"):
                if e.get(k) and k not in rec["attrs"]:
                    rec["attrs"][k] = e[k]
    canonical_entities, alias_map = [], {}

    def add_cluster(etype, canonical, aliases, pool):
        info = {"canonical": canonical, "type": etype, "aliases": sorted(set(aliases)),
                "mentions": 0, "attrs": {}, "description": ""}
        best_desc = ""
        for a in info["aliases"]:
            rec = pool.get(a)
            if not rec:
                continue
            info["mentions"] += rec["mentions"]
            for k, v in rec["attrs"].items():
                info["attrs"].setdefault(k, v)
            for d in rec["descriptions"]:
                if len(d) > len(best_desc):
                    best_desc = d
        info["description"] = best_desc
        canonical_entities.append(info)
        for a in info["aliases"]:
            alias_map.setdefault(a, []).append({"canonical": canonical, "type": etype})

    for etype, pool in sorted(by_type.items()):
        names = sorted(pool)
        if len(names) == 1:
            add_cluster(etype, names[0], names, pool)
            continue
        listing = "\n".join(
            f"- {n}: {pool[n]['descriptions'][0] if pool[n]['descriptions'] else '(no description)'}"
            for n in names)
        print(f"resolving {len(names)} {etype} entities")
        reply = call_claude(prompts.resolve(etype, listing), SYNTHESIS_MODEL)
        clusters = parse_json_reply(reply).get("clusters", [])
        assigned = set()
        for cl in clusters:
            aliases = [a for a in cl.get("aliases", []) if a in pool and a not in assigned]
            if not aliases:
                continue
            canonical = cl.get("canonical") or aliases[0]
            if canonical not in pool and canonical not in aliases:
                canonical = max(aliases, key=len)
            assigned.update(aliases)
            add_cluster(etype, canonical, aliases, pool)
        leftovers = [n for n in names if n not in assigned]
        for n in leftovers:  # resolution fallback: silent-loss guard
            add_cluster(etype, n, [n], pool)
        if leftovers:
            print(f"  fallback singleton clusters for {len(leftovers)} unassigned name(s)")
    write_json(out / "entities.json", canonical_entities)
    write_json(out / "alias-map.json", alias_map)
    print(f"{sum(len(v) for v in by_type.values())} raw names -> {len(canonical_entities)} canonical entities")


# ---------------------------------------------------------------- link

LOOKUP = "https://lookup.dbpedia.org/api/search?query={q}&maxResults=5&format=JSON"
WD_SEARCH = ("https://www.wikidata.org/w/api.php?action=wbsearchentities"
             "&search={q}&language=en&format=json&limit=5&type=item")
TYPE_OK = {
    "PERSON": ("Person",),
    "ORGANIZATION": ("Organis", "Organiz", "Company", "University", "Institution",
                     "School", "Publisher", "Society", "Group", "Newspaper", "Magazine"),
    "WORK": ("Work", "Book", "Film", "Show", "Radio", "Magazine", "Newspaper"),
    "PLACE": ("Place", "City", "Country", "Settlement", "Location"),
    "EVENT": ("Event",),
    "CONCEPT": (),
}


def _strip_tags(s: str) -> str:
    return re.sub(r"</?B>", "", s or "", flags=re.I)


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return norm_text(s).lower()


WD_CLAIMS = ("https://www.wikidata.org/w/api.php?action=wbgetclaims"
             "&entity={qid}&property=P31&format=json")
# instance-of target per entity type; CONCEPT is the negative case below
WD_TYPE_TARGET = {"PERSON": "Q5", "ORGANIZATION": "Q43229", "WORK": "Q386724",
                  "EVENT": "Q1190554", "PLACE": "Q2221906"}


def _wd_p31(qid: str):
    """The item's instance-of classes, from the plain Wikidata API (no WDQS)."""
    err = None
    for attempt in range(3):
        time.sleep(0.3 if attempt == 0 else 5 * attempt)
        try:
            claims = http_get_json(WD_CLAIMS.format(qid=qid)).get("claims", {})
            out = []
            for c in claims.get("P31", []):
                v = (c.get("mainsnak") or {}).get("datavalue")
                if v and isinstance(v.get("value"), dict) and v["value"].get("id"):
                    out.append(v["value"]["id"])
            return out
        except Exception as exc:
            err = exc
    print(f"    wbgetclaims failed for {qid}: {err}")
    return None


def _class_subclass_of(cls: str, target: str, cache: dict) -> bool | None:
    """Cached WDQS check that class `cls` sits under `target` via P279*.
    Anchored at a class, this stays cheap - the itemwise P31/P279* path ASK
    used before timed out on WDQS under load (research log 2026-08-25)."""
    key = f"{cls}|{target}"
    if key in cache:
        return cache[key]
    if cls == target:
        cache[key] = True
        return True
    q = f"ASK {{ wd:{cls} wdt:P279* wd:{target} }}"
    url = ("https://query.wikidata.org/sparql?query="
           + urllib.parse.quote(q) + "&format=json")
    err = None
    for attempt in range(3):
        time.sleep(0.3 if attempt == 0 else 5 * attempt)  # WDQS rate-limits bursts
        try:
            cache[key] = bool(sparql_json(url, timeout=40)["boolean"])
            return cache[key]
        except Exception as exc:
            err = exc
    print(f"    WDQS subclass check failed for {cls}: {err}")
    return None  # unknown - do not cache


def _wd_type_ok(qid: str, etype: str, cache: dict) -> bool:
    classes = _wd_p31(qid)
    if classes is None:
        return False  # cannot type-check -> do not link
    if etype == "PERSON":
        return "Q5" in classes
    if etype == "CONCEPT":
        # a concept item must be neither a human nor any kind of work
        if "Q5" in classes:
            return False
        checks = [_class_subclass_of(c, "Q386724", cache) for c in classes]
        return not any(checks) and None not in checks
    target = WD_TYPE_TARGET[etype]
    return any(_class_subclass_of(c, target, cache) for c in classes)


def _wd_candidates(names, targets, etype, cache, limit=3):
    """Exact-label Wikidata search; returns type-gated candidates with
    label and description. A name match alone never links: candidates go
    on to LLM adjudication against the document's own description."""
    out, seen = [], set()
    for q in dict.fromkeys(names):
        res, err = None, None
        for attempt in range(3):
            time.sleep(0.3 if attempt == 0 else 8 * attempt)  # API rate limits
            try:
                res = http_get_json(WD_SEARCH.format(q=urllib.parse.quote(q)))
                break
            except Exception as exc:
                err = exc
        if res is None:
            print(f"    wbsearchentities failed for {q!r}: {err}")
            continue
        for hit in res.get("search", []):
            qid = hit["id"]
            if qid in seen:
                continue
            cand = {_norm_name(hit.get("label") or "")}
            cand.update(_norm_name(a) for a in hit.get("aliases") or [])
            # the CONCEPT type check is purely negative (not human, not work),
            # which a statement-less stub passes vacuously; without a
            # description such a candidate carries nothing but a name match,
            # so it never reaches adjudication
            if etype == "CONCEPT" and not (hit.get("description") or "").strip():
                continue
            if cand & targets and _wd_type_ok(qid, etype, cache):
                seen.add(qid)
                out.append({"qid": qid, "label": hit.get("label") or "",
                            "description": hit.get("description") or ""})
                if len(out) >= limit:
                    return out
        time.sleep(0.3)
    return out


def cmd_link(args) -> None:
    out = Path(args.out)
    entities = read_json(out / "entities.json")
    links, links_path = {}, out / "links.json"
    only = {s.strip().lower() for s in getattr(args, "entities", "").split(",") if s.strip()}
    if only:  # targeted re-link: everything else keeps its stored record
        links = read_json(links_path) if links_path.exists() else {}
        entities = [e for e in entities if e["canonical"].lower() in only]
        if not entities:
            sys.exit("--entities matched no canonical entity name")
    dbp_pending = []  # (entity, links key, gated candidates)
    for ent in entities:
        names = [ent["canonical"]] + ent["aliases"]
        targets = {_norm_name(n) for n in names}
        cands, reason = [], "no exact label match"
        seen_uris = set()
        for q in dict.fromkeys(names):  # preserve order, dedupe
            docs, err = None, None
            for attempt in range(3):  # Lookup shares the wbsearchentities
                if attempt:           # retry policy: transient failures were
                    time.sleep(4 * attempt)  # costing links (research log)
                try:
                    docs = http_get_json(LOOKUP.format(q=urllib.parse.quote(q))).get("docs", [])
                    break
                except Exception as exc:
                    err = exc
            if docs is None:
                reason = f"lookup error: {err}"
                docs = []
            for d in docs:
                labels = {_norm_name(_strip_tags(x))
                          for x in (d.get("label") or []) + (d.get("redirectlabel") or [])}
                if not labels & targets:
                    continue
                typenames = [t for t in (d.get("typeName") or [])]
                need = TYPE_OK[ent["type"]]
                if need and not any(any(frag in tn for frag in need) for tn in typenames):
                    reason = f"label match but type mismatch ({typenames})"
                    continue
                if ent["type"] == "CONCEPT" and any("Person" in tn or "Place" in tn for tn in typenames):
                    reason = f"concept matched a {typenames} resource"
                    continue
                # the CONCEPT gate is exclusions only, so a candidate without
                # an abstract offers nothing beyond the name match
                if ent["type"] == "CONCEPT" and not (d.get("comment") or [""])[0].strip():
                    reason = "concept candidate has no abstract"
                    continue
                uri = (d.get("resource") or [None])[0]
                if not uri or uri in seen_uris:
                    continue
                seen_uris.add(uri)
                cands.append({"uri": uri,
                              "matchedQuery": q,
                              "matchedLabel": _strip_tags((d.get("label") or [""])[0]),
                              "typeNames": typenames,
                              "comment": _strip_tags((d.get("comment") or [""])[0])})
                if len(cands) >= 3:
                    break
            if len(cands) >= 3:
                break
            time.sleep(0.4)
        key = f"{ent['canonical']}||{ent['type']}"
        links[key] = {"unlinked": True, "reason": reason}
        if cands:
            dbp_pending.append((ent, key, cands))
        status = f"{len(cands)} candidate(s)" if cands else f"unlinked ({reason})"
        print(f"  {ent['type']:<12} {ent['canonical'][:40]:<42} -> {status}")
    # Description adjudication of the DBpedia candidates (decision-log D8):
    # a name-and-type match alone never links - the candidate's abstract must
    # be consistent with the document's own description of the entity.
    # DBpedia's title convention gives an undisambiguated name to its most
    # famous bearer, so the first gated hit is only the most famous namesake.
    # One batched call; when the call fails, entities stay unlinked - the
    # same failure mode as the type gate (cannot check -> do not link).
    if dbp_pending:
        blocks = []
        for i, (ent, _, cands) in enumerate(dbp_pending, 1):
            lines = [f"{i}. document entity: {ent['canonical']} (type {ent['type']})",
                     f"   document description: {ent['description'] or '(none)'}"]
            for c in cands:
                lines.append(f"   candidate {c['uri']}: {c['comment'] or '(no description)'}")
            blocks.append("\n".join(lines))
        print(f"adjudicating {len(dbp_pending)} dbpedia candidate set(s)")
        try:
            reply = call_claude(prompts.adjudicate(
                "DBpedia resources", "http://dbpedia.org/resource/...",
                "\n".join(blocks)), SYNTHESIS_MODEL)
            decisions = {d["id"]: d for d in parse_json_reply(reply).get("decisions", [])}
        except (RuntimeError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"  adjudication failed ({exc}); leaving candidates unlinked")
            decisions = {}
        for i, (ent, key, cands) in enumerate(dbp_pending, 1):
            d = decisions.get(i, {})
            chosen = next((c for c in cands if c["uri"] == d.get("match")), None)
            # the full candidate set stays in the record either way, so every
            # decision - and every rejection - is auditable without re-linking
            weighed = [{"uri": c["uri"], "comment": c["comment"][:200]} for c in cands]
            if d.get("verdict") == "matched" and chosen:
                rec = {k: chosen[k] for k in ("uri", "matchedQuery", "matchedLabel", "typeNames")}
                rec["method"] = "DBpedia Lookup + type gate + LLM adjudication"
                rec["adjudication"] = d.get("why", "")
                rec["dbpediaCandidates"] = weighed
                links[key] = rec
            else:
                # the rationale keeps its route name: the record may go on to
                # the Wikidata route, whose own rationale lands in "adjudication"
                links[key] = {"unlinked": True, "reason": {
                    "rejected": "dbpedia candidates rejected in adjudication",
                    "unsure": "dbpedia adjudication unsure (insufficient context)",
                }.get(d.get("verdict"), "dbpedia adjudication failed"),
                    "dbpediaAdjudication": d.get("why", ""), "dbpediaCandidates": weighed}
                print(f"  dbpedia {links[key]['reason'].split('(')[0].strip()}: "
                      f"{ent['type']:<12} {ent['canonical'][:40]}")
    # Wikidata sameAs for the linked subset, one SPARQL call
    dbr_uris = [v["uri"] for v in links.values() if v.get("uri")]
    if dbr_uris:
        values = " ".join(f"<{u}>" for u in dbr_uris)
        q = ("SELECT ?dbr ?wd WHERE { VALUES ?dbr { " + values + " } "
             "?dbr <http://www.w3.org/2002/07/owl#sameAs> ?wd . "
             "FILTER(STRSTARTS(STR(?wd), \"http://www.wikidata.org/entity/\")) }")
        url = ("https://dbpedia.org/sparql?query=" + urllib.parse.quote(q)
               + "&format=application%2Fsparql-results%2Bjson")
        try:
            res = sparql_json(url, timeout=60)
            wd = {b["dbr"]["value"]: b["wd"]["value"]
                  for b in res["results"]["bindings"]}
            for v in links.values():
                if v.get("uri") and v["uri"] in wd:
                    v["wikidata"] = wd[v["uri"]]
            print(f"wikidata sameAs found for {len(wd)}/{len(dbr_uris)} linked entities")
        except Exception as exc:
            print(f"wikidata sameAs lookup failed: {exc}")
    # Wikidata-native fallback: entities DBpedia could not link may still
    # exist on Wikidata (its coverage is much larger). Deterministic gate
    # (exact label + instance-of check), then LLM adjudication of every
    # candidate against the document's own description of the entity.
    pending = []
    class_cache_path = VOCAB_CACHE / "wd-class.checked.json"
    class_cache = read_json(class_cache_path) if class_cache_path.exists() else {}
    for ent in entities:
        rec = links[f"{ent['canonical']}||{ent['type']}"]
        if rec.get("wikidata") or rec.get("uri"):
            continue
        names = [ent["canonical"]] + ent["aliases"]
        cands = _wd_candidates(names, {_norm_name(n) for n in names},
                               ent["type"], class_cache)
        if cands:
            pending.append((ent, rec, cands))
    VOCAB_CACHE.mkdir(parents=True, exist_ok=True)
    write_json(class_cache_path, class_cache)
    n_wd_direct = 0
    if pending:
        blocks = []
        for i, (ent, _, cands) in enumerate(pending, 1):
            lines = [f"{i}. document entity: {ent['canonical']} (type {ent['type']})",
                     f"   document description: {ent['description'] or '(none)'}"]
            for c in cands:
                lines.append(f"   candidate {c['qid']}: {c['label']} - {c['description'] or '(no description)'}")
            blocks.append("\n".join(lines))
        print(f"adjudicating {len(pending)} wikidata candidate set(s)")
        try:
            reply = call_claude(prompts.adjudicate(
                "Wikidata items", "Q123", "\n".join(blocks)), SYNTHESIS_MODEL)
            decisions = {d["id"]: d for d in parse_json_reply(reply).get("decisions", [])}
        except (RuntimeError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"  adjudication failed ({exc}); leaving candidates unlinked")
            decisions = {}
        for i, (ent, rec, cands) in enumerate(pending, 1):
            d = decisions.get(i, {})
            qid = d.get("match")
            rec["wikidataCandidates"] = [{"qid": c["qid"], "label": c["label"],
                                          "description": c["description"][:200]} for c in cands]
            if d.get("verdict") == "matched" and any(c["qid"] == qid for c in cands):
                rec.pop("unlinked", None)
                rec.pop("reason", None)
                rec["wikidata"] = f"http://www.wikidata.org/entity/{qid}"
                rec["method"] = "wbsearchentities + WDQS type gate + LLM adjudication"
                rec["adjudication"] = d.get("why", "")
                n_wd_direct += 1
                print(f"  wikidata-direct: {ent['type']:<12} {ent['canonical'][:38]:<40} -> {qid}")
            else:
                # keep the earlier reason when the adjudication call failed
                verdict_reason = {
                    "rejected": "wikidata candidates rejected in adjudication",
                    "unsure": "wikidata adjudication unsure (insufficient context)",
                }.get(d.get("verdict"))
                rec["adjudication"] = d.get("why", "")
                rec["reason"] = verdict_reason or rec.get("reason") or "wikidata adjudication failed"
    write_json(out / "links.json", links)
    n_dbp = sum(1 for v in links.values() if v.get("uri"))
    n_wd = sum(1 for v in links.values() if v.get("wikidata"))
    print(f"linked {n_dbp}/{len(links)} entities to DBpedia, "
          f"{n_wd}/{len(links)} to Wikidata ({n_wd_direct} via direct search this run)")


# ---------------------------------------------------------------- assemble

def cmd_assemble(args) -> None:
    from rdflib import Dataset, Graph, Literal, Namespace, URIRef
    from rdflib.namespace import DCTERMS, FOAF, OWL, PROV, RDF, RDFS, SKOS, XSD

    SCHEMA = Namespace("https://schema.org/")
    DBO = Namespace("http://dbpedia.org/ontology/")
    CITO = Namespace("http://purl.org/spar/cito/")
    FABIO = Namespace("http://purl.org/spar/fabio/")
    DOCO = Namespace("http://purl.org/spar/doco/")
    ns = names(args)
    KGX = Namespace(NS_VOCAB)
    ENT = Namespace(ns.id)
    CLAIM = Namespace(ns.claim)
    GRAPHS = Namespace(ns.graph)
    DOC = Namespace(ns.doc)

    CLASS_MAP = {
        "PERSON": [SCHEMA.Person, FOAF.Person],
        "ORGANIZATION": [SCHEMA.Organization, FOAF.Organization],
        "WORK": [SCHEMA.CreativeWork],
        "CONCEPT": [SKOS.Concept],
        "EVENT": [SCHEMA.Event],
        "PLACE": [SCHEMA.Place],
    }
    WORKTYPE_MAP = {
        "BOOK": [SCHEMA.Book, FABIO.Book],
        "ARTICLE": [SCHEMA.ScholarlyArticle, FABIO.JournalArticle],
        "WORKING_PAPER": [FABIO.WorkingPaper],
        "BROADCAST": [SCHEMA.RadioEpisode, FABIO.AudioDocument],
    }
    PRED_MAP = {
        "influencedBy": DBO.influencedBy,
        "academicAdvisor": DBO.academicAdvisor,
        "knownFor": DBO.knownFor,
        "memberOf": SCHEMA.memberOf,
        "affiliation": SCHEMA.affiliation,
        "founder": SCHEMA.founder,
        "author": SCHEMA.author,
        "editor": SCHEMA.editor,
        "publisher": SCHEMA.publisher,
        "about": SCHEMA.about,
        "attendee": SCHEMA.attendee,
        "cites": CITO.cites,
        "citesAsEvidence": CITO.citesAsEvidence,
        "critiques": CITO.critiques,
        "disputes": CITO.disputes,
        "agreesWith": CITO.agreesWith,
        "extends": CITO.extends,
        "related": SKOS.related,
        "broader": SKOS.broader,
    }

    policy = getattr(args, "iri_policy", "wikidata")
    if policy == "dbpedia":
        # DBpedia-native properties too: swap in dbo: equivalents that exist
        # and fit (checked against the live ontology; dbo has no memberOf,
        # attendee, about, citation-stance or concept-relation properties,
        # so those keep their schema.org / CiTO / SKOS homes)
        PRED_MAP.update({
            "author": DBO.author,         # Work -> Person
            "editor": DBO.editor,
            "publisher": DBO.publisher,
            "founder": DBO.founder,       # range Person
            "affiliation": DBO.employer,  # Person -> Organisation
        })
        for etype, extra in {"PERSON": DBO.Person, "ORGANIZATION": DBO.Organisation,
                             "WORK": DBO.Work, "EVENT": DBO.Event,
                             "PLACE": DBO.Place}.items():
            CLASS_MAP[etype].append(extra)

    out = Path(args.out)
    meta = read_json(out / "chunks.json")
    chunks = {c["id"]: c for c in meta["chunks"]}
    raw = read_json(out / "extraction-raw.json")
    entities = read_json(out / "entities.json")
    alias_map = read_json(out / "alias-map.json")
    links = read_json(out / "links.json")

    ds = Dataset()
    bindings = [("schema", SCHEMA), ("dbo", DBO), ("cito", CITO), ("fabio", FABIO),
                ("doco", DOCO), ("kgx", KGX), ("ent", ENT), ("claim", CLAIM),
                ("doc", DOC), ("prov", PROV), ("skos", SKOS), ("dcterms", DCTERMS),
                ("foaf", FOAF), ("owl", OWL),
                ("dbr", Namespace("http://dbpedia.org/resource/")),
                ("wd", Namespace("http://www.wikidata.org/entity/"))]
    for p, nsuri in bindings:  # not `ns`: that name holds this run's namespaces
        ds.bind(p, nsuri)

    g_ent = ds.graph(GRAPHS.entities)
    g_prov = ds.graph(GRAPHS.provenance)

    # document + sections
    doc_slug = slugify(Path(meta["source"]).stem)
    doc_uri = DOC[doc_slug]
    g_ent.add((doc_uri, RDF.type, FABIO.WorkingPaper))
    g_ent.add((doc_uri, RDF.type, SCHEMA.CreativeWork))
    g_ent.add((doc_uri, RDFS.label, Literal(chunks["s00"]["heading"], lang="en")))
    section_uri = {}
    for cid, c in chunks.items():
        s_uri = DOC[f"{doc_slug}/section/{cid}"]
        section_uri[cid] = s_uri
        g_ent.add((s_uri, RDF.type, DOCO.Section))
        g_ent.add((s_uri, RDFS.label, Literal(c["heading"], lang="en")))
        g_ent.add((s_uri, SCHEMA.isPartOf, doc_uri))

    # extraction activity
    run_uri = URIRef(ns.activity + "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    agent_uri = URIRef(ns.agent + "bobinet-" + __version__.replace(".", "-"))
    g_prov.add((run_uri, RDF.type, PROV.Activity))
    g_prov.add((run_uri, RDFS.label, Literal("bobinet extraction run", lang="en")))
    g_prov.add((run_uri, PROV.used, doc_uri))
    g_prov.add((run_uri, PROV.used, URIRef(PROFILE_IRI)))
    g_prov.add((run_uri, PROV.wasAssociatedWith, agent_uri))
    g_prov.add((run_uri, PROV.startedAtTime,
                Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime)))
    g_prov.add((agent_uri, RDF.type, PROV.SoftwareAgent))
    g_prov.add((agent_uri, RDFS.label, Literal(
        f"bobinet {__version__} (extraction: {EXTRACTION_MODEL}; "
        f"synthesis: {SYNTHESIS_MODEL})", lang="en")))

    # canonical entities. Subject IRIs follow --iri-policy: reuse the
    # entity's Wikidata or DBpedia IRI directly when the linker found one
    # (the registries' own identifier becomes the node), or always mint
    # locally under policy "local". Unlinked entities mint locally in every
    # policy; owl:sameAs carries whichever registry IRIs are not the subject.
    uri_of = {}  # (canonical, type) -> URIRef
    used_slugs, used_ext = set(), {}  # ext IRI -> canonical name that claimed it
    collisions = []  # two document entities on one registry IRI: probably a
    for ent in entities:  # missed merge in resolve; reported by validate
        lk = links.get(f"{ent['canonical']}||{ent['type']}", links.get(ent["canonical"], {}))
        ext = {"dbpedia": lk.get("uri"), "wikidata": lk.get("wikidata")}.get(policy)
        if ext and ext in used_ext:
            print(f"  note: {ext} already used as a subject; minting locally for "
                  f"{ent['canonical']} ({ent['type']})")
            collisions.append({"iri": ext, "first": used_ext[ext],
                               "minted": f"{ent['canonical']} ({ent['type']})"})
            ext = None
        if ext:
            used_ext[ext] = f"{ent['canonical']} ({ent['type']})"
            u = URIRef(ext)
        else:
            slug = f"{TYPE_PREFIX[ent['type']]}/{slugify(ent['canonical'])}"
            while slug in used_slugs:
                slug += "-x"
            used_slugs.add(slug)
            u = ENT[slug]
        uri_of[(ent["canonical"], ent["type"])] = u
        for cls in CLASS_MAP[ent["type"]]:
            g_ent.add((u, RDF.type, cls))
        for cls in WORKTYPE_MAP.get(ent["attrs"].get("workType", ""), []):
            g_ent.add((u, RDF.type, cls))
        g_ent.add((u, RDFS.label, Literal(ent["canonical"], lang="en")))
        for a in ent["aliases"]:
            if a != ent["canonical"]:
                g_ent.add((u, SKOS.altLabel, Literal(a, lang="en")))
        if ent["description"]:
            g_ent.add((u, SCHEMA.description, Literal(ent["description"], lang="en")))
        for key, prop in [("birthYear", SCHEMA.birthDate), ("deathYear", SCHEMA.deathDate),
                          ("pubYear", SCHEMA.datePublished)]:
            v = ent["attrs"].get(key, "")
            if re.fullmatch(r"\d{4}", v):
                g_ent.add((u, prop, Literal(v, datatype=XSD.gYear)))
        for target in (lk.get("uri"), lk.get("wikidata")):
            if target and target != str(u):
                g_ent.add((u, OWL.sameAs, URIRef(target)))

    def endpoint_uri(name: str, allowed_types):
        cands = [c for c in alias_map.get(name, []) if c["type"] in allowed_types]
        if len({(c["canonical"], c["type"]) for c in cands}) == 1:
            c = cands[0]
            return uri_of.get((c["canonical"], c["type"]))
        return None

    # claims
    n_claim, skipped = 0, []
    for chunk in raw:
        for rel in chunk["relations"]:
            dom, rng, _ = prompts.PREDICATES[rel["predicate"]]
            s_uri = endpoint_uri(rel["subject"], dom.split("|"))
            o_uri = endpoint_uri(rel["object"], rng.split("|"))
            if s_uri is None or o_uri is None:
                skipped.append({"chunk": chunk["chunk_id"], "relation": rel,
                                "reason": "endpoint did not resolve to a unique canonical entity"})
                continue
            n_claim += 1
            c_uri = CLAIM[f"c{n_claim:03d}"]
            g_c = ds.graph(c_uri)
            pred = PRED_MAP[rel["predicate"]]
            g_c.add((s_uri, pred, o_uri))
            if rel["predicate"] == "influencedBy":
                g_c.add((s_uri, PROV.wasInfluencedBy, o_uri))
                if rel.get("influenceNote"):
                    inf = URIRef(str(c_uri) + "/influence")
                    g_c.add((s_uri, PROV.qualifiedInfluence, inf))
                    g_c.add((inf, RDF.type, PROV.Influence))
                    g_c.add((inf, PROV.influencer, o_uri))
                    g_c.add((inf, DCTERMS.description, Literal(rel["influenceNote"], lang="en")))
            g_prov.add((c_uri, RDF.type, KGX.Claim))
            g_prov.add((c_uri, PROV.value, Literal(rel["evidence"])))
            g_prov.add((c_uri, PROV.wasQuotedFrom, section_uri[chunk["chunk_id"]]))
            g_prov.add((c_uri, KGX.confidence,
                        Literal(f"{rel['confidence']:.2f}", datatype=XSD.decimal)))
            g_prov.add((c_uri, KGX.evidenceVerified,
                        Literal(bool(rel["evidence_verified"]), datatype=XSD.boolean)))
            g_prov.add((c_uri, PROV.wasGeneratedBy, run_uri))

    trig_path = out / (args.name + ".trig")
    ds.serialize(destination=str(trig_path), format="trig")
    print(f"wrote {trig_path}")
    merged = Graph()
    for p, nsuri in bindings:
        merged.bind(p, nsuri)
    for s, p, o, _ in ds.quads((None, None, None, None)):
        merged.add((s, p, o))
    ttl_path = out / (args.name + ".ttl")
    merged.serialize(destination=str(ttl_path), format="turtle")
    print(f"wrote {ttl_path} (merged, {len(merged)} triples)")
    if skipped:
        write_json(out / "assemble-skipped.json", skipped)
    if collisions:
        write_json(out / "assemble-collisions.json", collisions)
    else:
        (out / "assemble-collisions.json").unlink(missing_ok=True)
    print(f"{len(entities)} entities, {n_claim} claims, {len(skipped)} relations skipped")


# ---------------------------------------------------------------- zoom

# The fractalino zoom vocabulary (fractalino/vocab/fr.ttl). Fixed like kgx:,
# not derived from --base: consumers query these terms by IRI.
FR_NS = "http://velitchkov.eu/def/fr#"

# English labels for the fr: terms this stage writes into the ideas layer,
# so a visualiser can label edges from the data alone. Mirrors vocab/fr.ttl.
FR_LABELS = {"took": "took", "rejected": "rejected",
             "reformulated": "reformulated", "extended": "extended",
             "disputed": "disputed", "ideaOf": "idea of"}


def _validate_zoom(data: dict, pairs: list, chunk_text: str, person_names: set):
    """pairs: [{"labels": [subj label, obj label], "names": {norm accepted names}}].
    Returns (statements, errors). Statements keep their pair index (1-based)."""
    errors, stmts = [], []
    norm_chunk = norm_text(chunk_text)
    for st in data.get("statements", []):
        try:
            pair_no = int(st.get("pair", 0))
        except (TypeError, ValueError):
            pair_no = 0
        if not 1 <= pair_no <= len(pairs):
            errors.append(f"statement with bad pair number: {st.get('pair')!r}")
            continue
        stance = (st.get("stance") or "").strip()
        if stance not in prompts.ZOOM_STANCES:
            errors.append(f"unknown stance: {stance!r}")
            continue
        person = (st.get("person") or "").strip()
        if _norm_name(person) not in pairs[pair_no - 1]["names"]:
            errors.append(f"person {person!r} is not an endpoint of pair {pair_no}")
            continue
        idea = (st.get("idea") or "").strip()
        if not idea:
            errors.append("statement with empty idea")
            continue
        originator = (st.get("ideaOriginator") or "").strip()
        if originator and _norm_name(originator) not in person_names:
            errors.append(f"ideaOriginator {originator!r} is not a known person")
            originator = ""
        evidence = (st.get("evidence") or "").strip()
        verified = bool(evidence) and norm_text(evidence) in norm_chunk
        # An attribution has to be evidenced like anything else: the quote must
        # name the person it credits. Without this the slot fills with whoever
        # the pair's influencer is, which restates the edge instead of
        # explaining it - every attribution in the first run did exactly that,
        # and it produced "socialism is an idea of Mises" (fractalino D8). The
        # stance statement survives; only the attribution is dropped.
        if originator:
            surname = originator.split()[-1]
            if _norm_name(surname) not in _norm_name(evidence):
                errors.append(f"evidence does not name the originator {originator!r}: "
                              f"attribution dropped for idea {idea!r}")
                originator = ""
        if evidence and not verified:
            errors.append(f"evidence not a verbatim substring: {evidence[:80]!r}")
        try:
            conf = float(st.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8
        stmts.append({"pair": pair_no, "person": person, "stance": stance,
                      "idea": idea,
                      "ideaDescription": (st.get("ideaDescription") or "").strip(),
                      "ideaOriginator": originator, "evidence": evidence,
                      "confidence": max(0.0, min(1.0, conf)),
                      "evidence_verified": verified})
    return stmts, errors


def cmd_zoom(args) -> None:
    """Unfold coarse influence claims into an idea-level layer (fractalino T1):
    re-extract the named sections at claim granularity, assert each idea-level
    statement in a unit graph of its own, materialize the graph/ideas layer,
    and link every coarse influence claim to its units by fr:unfoldsTo in
    graph/zoom. Writes <name>-zoom.trig/.ttl as derived artifacts; the
    assembled <name>.trig is never touched."""
    from rdflib import Dataset, Graph, Literal, Namespace, URIRef
    from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, SKOS, XSD

    ns = names(args)
    out = Path(args.out)
    chunks_meta = read_json(out / "chunks.json")
    chunks = {c["id"]: c for c in chunks_meta["chunks"]}
    doc_slug = slugify(Path(chunks_meta["source"]).stem)
    entities = read_json(out / "entities.json")

    ds = Dataset()
    ds.parse(str(out / (args.name + ".trig")), format="trig")
    DBO = Namespace("http://dbpedia.org/ontology/")
    FR = Namespace(FR_NS)
    KGX = Namespace(NS_VOCAB)
    SCHEMA = Namespace("https://schema.org/")
    VOID = Namespace("http://rdfs.org/ns/void#")
    prov_g = ds.graph(URIRef(ns.graph + "provenance"))
    ent_g = ds.graph(URIRef(ns.graph + "entities"))

    # entity-name -> IRI maps from the assembled entity graph (labels and
    # altLabels), split by kind so namesakes across kinds cannot collide
    def name_map(cls):
        m = {}
        for s in ent_g.subjects(RDF.type, cls):
            for _, _, lbl in ent_g.triples((s, RDFS.label, None)):
                m.setdefault(_norm_name(str(lbl)), s)
            for _, _, alt in ent_g.triples((s, SKOS.altLabel, None)):
                m.setdefault(_norm_name(str(alt)), s)
        return m
    person_iri = name_map(SCHEMA.Person)
    concept_iri = name_map(SKOS.Concept)

    def label_of(u):
        v = ent_g.value(u, RDFS.label)
        return str(v) if v else str(u).rsplit("/", 1)[-1]

    # coarse influence claims: by section (which pairs a section evidences)
    # and by pair (every claim asserting that edge, whatever its section -
    # the unfolding details the edge, not the quote)
    claims_by_pair, pairs_by_section = {}, {}
    for g in ds.graphs():
        gname = str(g.identifier)
        if not gname.startswith(ns.claim):
            continue
        for s, _, o in g.triples((None, DBO.influencedBy, None)):
            claims_by_pair.setdefault((s, o), []).append(g.identifier)
            sect = str(prov_g.value(g.identifier, PROV.wasQuotedFrom) or "")
            sid = sect.rsplit("/", 1)[-1]
            if sid in chunks:
                lst = pairs_by_section.setdefault(sid, [])
                if (s, o) not in lst:
                    lst.append((s, o))

    targets = [c.strip() for c in args.chunks.split(",") if c.strip()] \
        if args.chunks else sorted(pairs_by_section)
    person_names = set(person_iri)

    # aliases per canonical name, for endpoint matching
    aliases_of = {e["canonical"]: [e["canonical"]] + e["aliases"] for e in entities}

    # ---- extraction (resumable: one section per call, saved as it lands)
    raw_path = out / "zoom-raw.json"
    raw = read_json(raw_path) if raw_path.exists() and not args.force else {}
    concept_names = sorted({e["canonical"] for e in entities if e["type"] == "CONCEPT"})
    for sid in targets:
        if sid in raw:
            continue
        pair_uris = pairs_by_section.get(sid) or []
        if not pair_uris:
            print(f"[{sid}] no influence claims quoted from this section; skipping")
            continue
        pairs = []
        for s, o in pair_uris:
            ls, lo = label_of(s), label_of(o)
            accepted = set()
            for l in (ls, lo):
                for n in aliases_of.get(l, [l]):
                    accepted.add(_norm_name(n))
            pairs.append({"labels": [ls, lo], "names": accepted})
        pair_lines = "\n".join(f"{i}. {p['labels'][0]} influencedBy {p['labels'][1]}"
                               for i, p in enumerate(pairs, 1))
        print(f"[{sid}] unfolding {len(pairs)} pair(s): {chunks[sid]['heading'][:60]}")
        prompt = prompts.zoom(sid, chunks[sid]["heading"], chunks[sid]["text"],
                              pair_lines, "\n".join(f"- {c}" for c in concept_names))
        reply = call_claude(prompt, SYNTHESIS_MODEL)
        try:
            data = parse_json_reply(reply)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  parse failed ({exc}); retrying")
            reply = call_claude(prompt + prompts.RETRY_SUFFIX.format(
                errors=f"- reply was not parseable JSON: {exc}"), SYNTHESIS_MODEL)
            data = parse_json_reply(reply)
        stmts, errors = _validate_zoom(data, pairs, chunks[sid]["text"], person_names)
        if errors and not args.no_retry:
            print(f"  {len(errors)} validation error(s); one corrective retry")
            reply = call_claude(prompt + prompts.RETRY_SUFFIX.format(
                errors="\n".join(f"- {e}" for e in errors)), SYNTHESIS_MODEL)
            try:
                data2 = parse_json_reply(reply)
                s2, e2 = _validate_zoom(data2, pairs, chunks[sid]["text"], person_names)
                if len(e2) < len(errors):
                    stmts, errors = s2, e2
            except (ValueError, json.JSONDecodeError):
                print("  retry unparseable; keeping first attempt")
        unverified = sum(1 for s in stmts if not s["evidence_verified"])
        print(f"  {len(stmts)} idea-level statement(s) "
              f"({unverified} unverified evidence, {len(errors)} residual errors)")
        raw[sid] = {"pairs": [p["labels"] for p in pairs], "statements": stmts,
                    "residual_errors": errors}
        write_json(raw_path, raw)

    # ---- assembly of the derived dataset
    run_uri = URIRef(ns.activity + "zoom-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    agent_uri = URIRef(ns.agent + "bobinet-" + __version__.replace(".", "-"))
    prov_g.add((run_uri, RDF.type, PROV.Activity))
    prov_g.add((run_uri, RDFS.label, Literal("bobinet zoom stage run", lang="en")))
    prov_g.add((run_uri, PROV.wasAssociatedWith, agent_uri))
    prov_g.add((run_uri, PROV.startedAtTime,
                Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime)))
    prov_g.add((agent_uri, RDF.type, PROV.SoftwareAgent))

    zoom_g = ds.graph(URIRef(ns.graph + "zoom"))
    ideas_g = ds.graph(URIRef(ns.graph + "ideas"))

    used_slugs = {str(s)[len(ns.id):] for s in ent_g.subjects()
                  if str(s).startswith(ns.id)}
    idea_by_name, n_new_ideas = {}, 0

    def borrowed_term(st, sid):
        """True when the statement pins itself to a concept the graph already
        knows while the section never uses that concept's name. The extractor
        reaches for vocabulary from elsewhere in the document and attaches it
        to whoever is at hand: the first run had Mises rejecting
        "constructivist rationality", a term that appears only in the Vernon
        Smith section, where the paper credits it to Hayek via Descartes
        (fractalino D8). Coined descriptions are unaffected - only reuse of an
        existing named concept has to be attested."""
        key = _norm_name(st["idea"])
        if key not in concept_iri:
            return False
        # Content words, stemmed to five characters, so a section saying
        # "laboratory experiment as an economic method" still attests
        # "Laboratory experiment in economics", while one that never says
        # "constructivist" cannot attest "constructivist rationality".
        stop = {"the", "a", "an", "of", "in", "on", "as", "and", "or", "to", "for"}
        words = [w for w in re.findall(r"[a-z]+", st["idea"].lower()) if w not in stop]
        text = chunks[sid]["text"].lower()
        return not all(w[:5] in text for w in words)

    def idea_uri(st):
        nonlocal n_new_ideas
        key = _norm_name(st["idea"])
        if key in idea_by_name:
            return idea_by_name[key]
        u = concept_iri.get(key)
        if u is None:
            slug = "concept/" + slugify(st["idea"])
            while slug in used_slugs:
                slug += "-x"
            used_slugs.add(slug)
            u = URIRef(ns.id + slug)
            n_new_ideas += 1
            ent_g.add((u, RDF.type, SKOS.Concept))
            ent_g.add((u, RDFS.label, Literal(st["idea"], lang="en")))
            if st["ideaDescription"]:
                ent_g.add((u, SCHEMA.description, Literal(st["ideaDescription"], lang="en")))
        idea_by_name[key] = u
        ideas_g.add((u, RDFS.label, Literal(st["idea"], lang="en")))
        return u

    def resolve_endpoint(name, pair_labels):
        # a person name matches within the pair's endpoints first, then globally
        key = _norm_name(name)
        for l in pair_labels:
            hit = None
            for n in aliases_of.get(l, [l]):
                if _norm_name(n) == key:
                    hit = person_iri.get(_norm_name(l))
                    break
            if hit is not None:
                return hit
        return person_iri.get(key)

    n_units, units_by_pair = 0, {}
    attr_by_pair, attr_unit = {}, {}
    borrowed = []   # statements dropped for reusing a concept the section never names
    for sid in targets:
        rec = raw.get(sid)
        if not rec:
            continue
        for st in rec["statements"]:
            if not 1 <= st["pair"] <= len(rec["pairs"]):
                continue
            # the pair comes from the labels RECORDED at extraction time:
            # graph iteration order is not stable across runs, so an index
            # into a recomputed pair list would scramble multi-pair sections
            pair_labels = rec["pairs"][st["pair"] - 1]
            pair = (person_iri.get(_norm_name(pair_labels[0])),
                    person_iri.get(_norm_name(pair_labels[1])))
            if pair[0] is None or pair[1] is None:
                continue
            if borrowed_term(st, sid):
                borrowed.append({"section": sid, "idea": st["idea"],
                                 "person": st["person"], "stance": st["stance"],
                                 "reason": "names an existing concept the section never uses"})
                continue
            person = resolve_endpoint(st["person"], pair_labels)
            if person is None:
                continue
            idea = idea_uri(st)
            stance = FR[st["stance"]]
            n_units += 1
            unit_uri = URIRef(ns.base + f"unit/u{n_units:03d}")
            units_by_pair.setdefault(pair, []).append(unit_uri)
            # C2: the unit graph holds exactly the idea-level statement
            ds.graph(unit_uri).add((person, stance, idea))
            # C3: the same statement, materialized into the level layer
            ideas_g.add((person, stance, idea))
            if st["ideaOriginator"]:
                orig = person_iri.get(_norm_name(st["ideaOriginator"]))
                if orig is not None:
                    ideas_g.add((idea, FR.ideaOf, orig))
                    ent_g.add((idea, FR.ideaOf, orig))
                    attr_by_pair.setdefault(pair, []).append((idea, orig))
            # the anchor is the unit graph itself: the unit IRI names the
            # graph holding exactly this statement, so membership carries the
            # unit-to-statement link. Never classic reification (fractalino
            # D6); a flat projection for CONSTRUCT consumers is the view
            # builder's job (singleton properties, fractalino t1-queries).
            zoom_g.add((unit_uri, FR.level, Literal(2)))
            # provenance, same contract as claim graphs
            prov_g.add((unit_uri, RDF.type, KGX.Claim))
            prov_g.add((unit_uri, PROV.value, Literal(st["evidence"])))
            prov_g.add((unit_uri, PROV.wasQuotedFrom,
                        URIRef(ns.doc + doc_slug + f"/section/{sid}")))
            prov_g.add((unit_uri, KGX.confidence,
                        Literal(f"{st['confidence']:.2f}", datatype=XSD.decimal)))
            prov_g.add((unit_uri, KGX.evidenceVerified,
                        Literal(bool(st["evidence_verified"]), datatype=XSD.boolean)))
            prov_g.add((unit_uri, PROV.wasGeneratedBy, run_uri))

    # attribution units: each idea's fr:ideaOf statement is itself a unit,
    # unfolded by the same anchors as the stance statements that mention the
    # idea. Without this the attribution edge would count as base-level data
    # and pin every idea node onto the coarse map. Shared ideas yield one
    # unit referenced from every pair that mentions them.
    for pair, attrs in attr_by_pair.items():
        for idea, orig in dict.fromkeys(attrs):
            u = attr_unit.get((idea, orig))
            if u is None:
                n_units += 1
                u = URIRef(ns.base + f"unit/u{n_units:03d}")
                attr_unit[(idea, orig)] = u
                ds.graph(u).add((idea, FR.ideaOf, orig))
                zoom_g.add((u, FR.level, Literal(2)))
            if u not in units_by_pair[pair]:
                units_by_pair[pair].append(u)

    # coarse anchors: every claim asserting an unfolded edge links to the
    # edge's units, wherever the claim was quoted from
    n_anchors = 0
    for pair, unit_uris in units_by_pair.items():
        for claim in claims_by_pair.get(pair, []):
            n_anchors += 1
            zoom_g.add((claim, FR.level, Literal(1)))
            for u in unit_uris:
                zoom_g.add((claim, FR.unfoldsTo, u))

    # the layer itself as a resource (C3), and edge labels for the fr: terms
    # so a visualiser can label stance edges from the data alone
    zoom_g.add((ideas_g.identifier, RDF.type, FR.Layer))
    zoom_g.add((ideas_g.identifier, FR.level, Literal(2)))
    for term, lbl in FR_LABELS.items():
        ideas_g.add((FR[term], RDFS.label, Literal(lbl, lang="en")))
    ideas_g.add((DBO.influencedBy, RDFS.label, Literal("influenced by", lang="en")))

    # VoID self-description of the graphs this stage adds (default graph)
    for g in (ideas_g, zoom_g):
        ds.default_context.add((g.identifier, RDF.type, VOID.Dataset))
        ds.default_context.add((g.identifier, VOID.triples, Literal(len(g))))

    ds.bind("fr", FR)
    ds.bind("void", VOID)
    ds.bind("unit", Namespace(ns.base + "unit/"))

    trig_path = out / (args.name + "-zoom.trig")
    ds.serialize(destination=str(trig_path), format="trig")
    print(f"wrote {trig_path}")
    merged = Graph()
    for p, nsuri in ds.namespaces():
        merged.bind(p, nsuri)
    for s, p, o, _ in ds.quads((None, None, None, None)):
        merged.add((s, p, o))
    ttl_path = out / (args.name + "-zoom.ttl")
    merged.serialize(destination=str(ttl_path), format="turtle")
    print(f"wrote {ttl_path} (merged, {len(merged)} triples)")
    if borrowed:
        write_json(out / "zoom-borrowed-terms.json", borrowed)
        for b in borrowed:
            print(f"  dropped [{b['section']}] {b['person']} {b['stance']} {b['idea']!r}: {b['reason']}")
    else:
        (out / "zoom-borrowed-terms.json").unlink(missing_ok=True)
    print(f"{n_units} idea-level units over {len(units_by_pair)} edges, "
          f"{n_anchors} coarse claims anchored, {n_new_ideas} new idea concepts, "
          f"{len(borrowed)} statements dropped as borrowed terms")


# ---------------------------------------------------------------- validate

VOCAB_SOURCES = {
    "http://www.w3.org/ns/prov#": ("prov", "https://www.w3.org/ns/prov.ttl", "turtle"),
    "http://purl.org/spar/cito/": ("cito", "https://sparontologies.github.io/cito/current/cito.ttl", "turtle"),
    "http://purl.org/spar/fabio/": ("fabio", "https://sparontologies.github.io/fabio/current/fabio.ttl", "turtle"),
    "http://purl.org/spar/doco/": ("doco", "https://sparontologies.github.io/doco/current/doco.ttl", "turtle"),
    "https://schema.org/": ("schema", "https://schema.org/version/latest/schemaorg-current-https.ttl", "turtle"),
    "http://www.w3.org/2004/02/skos/core#": ("skos", "https://www.w3.org/2009/08/skos-reference/skos.rdf", "xml"),
    "http://xmlns.com/foaf/0.1/": ("foaf", "http://xmlns.com/foaf/spec/index.rdf", "xml"),
    "http://purl.org/dc/terms/": ("dcterms", "https://www.dublincore.org/specifications/dublin-core/dcmi-terms/dublin_core_terms.ttl", "turtle"),
}
BUILTIN_NS = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
)


def _vocab_subjects(nskey: str):
    """Fetch (once) and cache the set of subject IRIs defined by a vocabulary."""
    import rdflib
    name, url, fmt = VOCAB_SOURCES[nskey]
    cache = VOCAB_CACHE / f"{name}.subjects.json"
    if cache.exists():
        return set(read_json(cache))
    try:
        body = _fetch_bytes(url, "text/turtle, application/rdf+xml;q=0.9", timeout=120)
        g = rdflib.Graph()
        g.parse(data=body, format=fmt)
        subjects = {str(s) for s in g.subjects() if isinstance(s, rdflib.URIRef)}
        VOCAB_CACHE.mkdir(parents=True, exist_ok=True)
        write_json(cache, sorted(subjects))
        return subjects
    except Exception as exc:
        print(f"  vocab fetch failed for {name}: {exc}")
        return None


def _dbo_term_exists(term: str, cache: dict):
    if term in cache:
        return cache[term]
    q = f"ASK {{ <{term}> ?p ?o }}"
    url = ("https://dbpedia.org/sparql?query=" + urllib.parse.quote(q)
           + "&format=application%2Fsparql-results%2Bjson")
    try:
        cache[term] = bool(sparql_json(url, timeout=30)["boolean"])
        return cache[term]
    except Exception:
        return None


def cmd_validate(args) -> None:
    import rdflib
    from rdflib import Dataset, Graph
    from pyshacl import validate as shacl_validate

    out = Path(args.out)
    ns = names(args)
    report = ["# Validation report", "",
              f"Graph: `{args.name}.trig` - generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')}", ""]

    ds = Dataset()
    ds.parse(str(out / (args.name + ".trig")), format="trig")
    merged = Graph()
    for s, p, o, _ in ds.quads((None, None, None, None)):
        merged.add((s, p, o))
    n_graphs = len(set(g for _, _, _, g in ds.quads((None, None, None, None))))
    report.append(f"## Size\n\n{len(merged)} distinct triples in {n_graphs} named graphs.\n")

    # 1. SHACL
    conforms, _, results_text = shacl_validate(
        merged, shacl_graph=str(HERE / "shapes.ttl"), inference="none")
    report.append("## SHACL\n")
    report.append(f"Conforms: **{conforms}**\n")
    if not conforms:
        report.append("```\n" + results_text.strip()[:4000] + "\n```\n")
    print(f"SHACL conforms: {conforms}")

    # 2. vocabulary dereference check
    report.append("## Vocabulary conformance\n")
    report.append("Every class and property in the graph that is not run-local, checked "
                  "against its ontology's current dereferenced form; the tool's own kgx: "
                  "terms against `profile.ttl`.\n")
    terms = set()
    for _, p, _ in merged:
        terms.add(str(p))
    for _, _, o in merged.triples((None, rdflib.RDF.type, None)):
        terms.add(str(o))
    VOCAB_CACHE.mkdir(parents=True, exist_ok=True)
    dbo_cache_path = VOCAB_CACHE / "dbo.checked.json"
    dbo_cache = read_json(dbo_cache_path) if dbo_cache_path.exists() else {}
    profile = rdflib.Graph()
    profile.parse(str(HERE / "profile.ttl"), format="turtle")
    profile_terms = {str(s) for s in profile.subjects()}
    rows, n_ok, n_fail = [], 0, 0
    for t in sorted(terms):
        if t.startswith(ns.base) or t.startswith(BUILTIN_NS):
            continue
        if t.startswith(NS_VOCAB):
            ok, src = t in profile_terms, "profile.ttl"
        elif t.startswith("http://dbpedia.org/ontology/"):
            ok = _dbo_term_exists(t, dbo_cache)
            src = "DBpedia SPARQL ASK"
        else:
            ok, src = None, "?"
            for nskey in VOCAB_SOURCES:
                if t.startswith(nskey):
                    subs = _vocab_subjects(nskey)
                    ok = (t in subs) if subs is not None else None
                    src = VOCAB_SOURCES[nskey][0]
                    break
        mark = {True: "OK", False: "**MISSING**", None: "unverified (fetch failed)"}[ok]
        if ok is True:
            n_ok += 1
        elif ok is False:
            n_fail += 1
        rows.append(f"| `{t}` | {src} | {mark} |")
    write_json(dbo_cache_path, dbo_cache)
    report.append("| term | vocabulary | status |")
    report.append("|---|---|---|")
    report.extend(rows)
    report.append(f"\n{n_ok} verified, {n_fail} missing.\n")
    print(f"vocabulary: {n_ok} terms verified, {n_fail} missing")

    # 3. registry IRI collisions (two document entities on one registry IRI)
    coll_path = out / "assemble-collisions.json"
    if coll_path.exists():
        coll = read_json(coll_path)
        report.append("## Warnings\n")
        report.append(f"{len(coll)} registry IRI(s) claimed by two document entities - "
                      "probably one real-world entity that `resolve` failed to merge. "
                      "The second claimant minted a local IRI:\n")
        for c in coll:
            report.append(f"- `{c['iri']}` claimed by {c['first']}; also linked from {c['minted']}")
        report.append("")
        print(f"warnings: {len(coll)} registry IRI collision(s)")

    # 4. evidence re-check
    chunks = {c["id"]: c for c in read_json(out / "chunks.json")["chunks"]}
    KGX = rdflib.Namespace(NS_VOCAB)
    PROV = rdflib.Namespace("http://www.w3.org/ns/prov#")
    n_claims = n_verified = n_flagged = 0
    contested = []
    for claim in merged.subjects(rdflib.RDF.type, KGX.Claim):
        n_claims += 1
        quote = str(merged.value(claim, PROV.value) or "")
        sect = str(merged.value(claim, PROV.wasQuotedFrom) or "")
        cid = sect.rsplit("/", 1)[-1]
        ok = cid in chunks and norm_text(quote) in norm_text(chunks[cid]["text"])
        if ok:
            n_verified += 1
        else:
            n_flagged += 1
        conf = merged.value(claim, KGX.confidence)
        if conf is not None and float(conf) < 0.7:
            contested.append((str(claim), float(conf), quote[:100]))
    report.append("## Evidence\n")
    report.append(f"{n_claims} claims; {n_verified} evidence quotes re-verified as verbatim "
                  f"substrings of their source section; {n_flagged} not verbatim (flagged "
                  f"`kgx:evidenceVerified false` at assembly or failing now).\n")
    if contested:
        report.append(f"{len(contested)} claims carry confidence < 0.7 (the text marks them contested or secondhand):\n")
        for c, conf, q in sorted(contested, key=lambda x: x[1]):
            report.append(f"- `{c.rsplit('/', 1)[-1]}` ({conf:.2f}) \"{q}...\"")
        report.append("")
    print(f"evidence: {n_verified}/{n_claims} verified, {len(contested)} low-confidence")

    # 5. reconciliation against the DBpedia influence lists
    gold_path = Path(args.gold) if args.gold else None
    if gold_path and gold_path.exists():
        gold = read_json(gold_path)
        up_gold, down_gold = set(), set()
        for b in gold["results"]["bindings"]:
            if "InfluencerLabel" in b:
                up_gold.add(_norm_name(b["InfluencerLabel"]["value"]))
            if "InfluencedLabel" in b:
                down_gold.add(_norm_name(b["InfluencedLabel"]["value"]))
        DBO = rdflib.Namespace("http://dbpedia.org/ontology/")
        focus_key = _norm_name(args.gold_entity or "")
        focus = None
        for s in merged.subjects(rdflib.RDFS.label, None):
            lbl = merged.value(s, rdflib.RDFS.label)
            if focus_key and lbl and focus_key in _norm_name(str(lbl)):
                if (s, rdflib.RDF.type, rdflib.URIRef("https://schema.org/Person")) in merged:
                    focus = s
                    break

        def entity_labels(n):
            labels = set()
            lbl = merged.value(n, rdflib.RDFS.label)
            if lbl:
                labels.add(_norm_name(str(lbl)))
            for alt in merged.objects(n, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#altLabel")):
                labels.add(_norm_name(str(alt)))
            return labels

        report.append("## Reconciliation with the DBpedia influence lists\n")
        if focus is None:
            report.append(f"No person node matches `--gold-entity {args.gold_entity}`; "
                          "reconciliation skipped.\n")
        else:
            who = str(merged.value(focus, rdflib.RDFS.label) or args.gold_entity)
            up_ours = set(merged.objects(focus, DBO.influencedBy))
            down_ours = set(merged.subjects(DBO.influencedBy, focus))
            for direction, ours, gold_set in [(f"upstream (influenced {who})", up_ours, up_gold),
                                              (f"downstream ({who} influenced)", down_ours, down_gold)]:
                inter, only = [], []
                for n in ours:
                    labels = entity_labels(n)
                    display = str(merged.value(n, rdflib.RDFS.label) or n)
                    (inter if labels & gold_set else only).append(display)
                report.append(f"**{direction}**: document graph {len(ours)} entities, DBpedia {len(gold_set)} names, overlap {len(inter)}.")
                report.append(f"- in both: {', '.join(sorted(inter)) or '(none)'}")
                report.append(f"- document only: {', '.join(sorted(only)) or '(none)'}")
                report.append("")
            report.append("The DBpedia lists are a candidate pool, not ground truth; "
                          "overlap is a sanity check, not a recall score.\n")

    # 6. summary stats
    SCHEMA_NS = rdflib.Namespace("https://schema.org/")
    counts = {}
    for cls, label in [(SCHEMA_NS.Person, "persons"), (SCHEMA_NS.Organization, "organizations"),
                       (SCHEMA_NS.CreativeWork, "works"),
                       (rdflib.URIRef("http://www.w3.org/2004/02/skos/core#Concept"), "concepts"),
                       (SCHEMA_NS.Event, "events"), (SCHEMA_NS.Place, "places")]:
        counts[label] = len(set(merged.subjects(rdflib.RDF.type, cls)))
    OWL_NS = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
    n_sameas = len(set(merged.subjects(OWL_NS.sameAs, None)))
    report.append("## Statistics\n")
    report.append("| | count |")
    report.append("|---|---|")
    for k, v in counts.items():
        report.append(f"| {k} | {v} |")
    report.append(f"| claims | {n_claims} |")
    report.append(f"| entities with owl:sameAs | {n_sameas} |")
    report.append("")

    rpt = out / "validation-report.md"
    rpt.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {rpt}")


# ---------------------------------------------------------------- query

def cmd_query(args) -> None:
    from rdflib import Dataset
    out = Path(args.out)
    ns = names(args)
    ds = Dataset(default_union=True)
    ds.parse(str(out / (args.name + ".trig")), format="trig")
    lines = ["# Competency query results", ""]
    for rq in sorted((HERE / "queries").glob("*.rq")):
        q = rq.read_text(encoding="utf-8")
        title = next((l[2:].strip() for l in q.splitlines() if l.startswith("# ")), rq.stem)
        lines.append(f"## {title}\n")
        lines.append(f"`queries/{rq.name}`\n")
        try:
            res = ds.query(q)
        except Exception as exc:
            lines.append(f"Query failed: {exc}\n")
            continue
        vars_ = [str(v) for v in res.vars] if res.vars else []
        lines.append("| " + " | ".join(vars_) + " |")
        lines.append("|" + "---|" * len(vars_))
        n = 0
        for row in res:
            cells = []
            for v in vars_:
                val = row[v] if v in row.labels else None
                s = str(val) if val is not None else ""
                for pre, short in [(ns.id, "ent:"), (ns.claim, "claim:"), (ns.doc, "doc:"),
                                   (NS_VOCAB, "kgx:"),
                                   ("http://www.wikidata.org/entity/", "wd:"),
                                   ("http://dbpedia.org/resource/", "dbr:"),
                                   ("http://dbpedia.org/ontology/", "dbo:"),
                                   ("http://purl.org/spar/cito/", "cito:"),
                                   ("https://schema.org/", "schema:"),
                                   ("http://www.w3.org/2004/02/skos/core#", "skos:")]:
                    s = s.replace(pre, short)
                cells.append(s.replace("|", "\\|").replace("\n", " ")[:120])
            lines.append("| " + " | ".join(cells) + " |")
            n += 1
        lines.append(f"\n{n} rows.\n")
        print(f"{rq.name}: {n} rows")
    path = out / "query-results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


# ---------------------------------------------------------------- ask

def cmd_ask(args) -> None:
    from rdflib import Dataset, Namespace
    from rdflib.namespace import RDF as RDF_, RDFS
    out = Path(args.out)
    ds = Dataset(default_union=True)
    ds.parse(str(out / (args.name + ".trig")), format="trig")
    KGX = Namespace(NS_VOCAB)
    PROV = Namespace("http://www.w3.org/ns/prov#")

    def label(n):
        v = ds.value(n, RDFS.label)
        return str(v) if v else str(n).rsplit("/", 1)[-1]

    facts = []
    for c in ds.subjects(RDF_.type, KGX.Claim):
        gid = str(c)
        quote = str(ds.value(c, PROV.value) or "")[:150]
        conf = str(ds.value(c, KGX.confidence) or "")
        for s, p, o in ds.graph(c):
            if str(p).startswith(str(PROV)) or str(p) == str(RDF_.type):
                continue
            pname = str(p).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            facts.append(f"({label(s)}) --[{pname}]--> ({label(o)})  "
                         f"[claim {gid.rsplit('/', 1)[-1]}, confidence {conf}, evidence: \"{quote}\"]")
    context = "\n".join(sorted(set(facts)))
    print(call_claude(prompts.ask(args.question, context), SYNTHESIS_MODEL))


# ---------------------------------------------------------------- main

def _default_name(args) -> str:
    """Name the outputs after the source document. Stages after `chunk` need
    not repeat --src: chunks.json remembers the source path."""
    src = args.src
    if not src:
        chunks = Path(args.out) / "chunks.json"
        if chunks.exists():
            src = read_json(chunks).get("source", "")
    return slugify(Path(src).stem) if src else "graph"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="source markdown document; required by `chunk`")
    ap.add_argument("--out", default="out", help="output directory (default: ./out)")
    ap.add_argument("--name", help="basename for the .trig/.ttl output "
                                   "(default: the source document's slug)")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="base IRI for the data this run produces "
                         f"(default: {DEFAULT_BASE})")
    ap.add_argument("--gold", default="",
                    help="DBpedia SPARQL result JSON to reconcile influence edges against")
    ap.add_argument("--gold-entity", default="", dest="gold_entity",
                    help="label of the person whose influence lists --gold holds")
    ap.add_argument("--iri-policy", choices=("wikidata", "dbpedia", "local"),
                    default="wikidata", dest="iri_policy",
                    help="subject IRIs for linked entities: reuse the Wikidata IRI "
                         "(default), reuse the DBpedia IRI, or always mint local IRIs; "
                         "unlinked entities mint local IRIs in every policy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chunk")
    p_ex = sub.add_parser("extract")
    p_ex.add_argument("--chunks", default="", help="comma-separated chunk ids to (re)extract")
    p_ex.add_argument("--no-retry", action="store_true")
    p_ex.add_argument("--force", action="store_true", help="re-extract chunks already done")
    sub.add_parser("resolve")
    p_ln = sub.add_parser("link")
    p_ln.add_argument("--entities", default="",
                      help="comma-separated canonical names to (re)link; "
                           "all other entities keep their stored links")
    sub.add_parser("assemble")
    p_zm = sub.add_parser("zoom")
    p_zm.add_argument("--chunks", default="",
                      help="comma-separated section ids to unfold at idea level "
                           "(default: every section that evidences an influence claim)")
    p_zm.add_argument("--no-retry", action="store_true")
    p_zm.add_argument("--force", action="store_true",
                      help="re-extract sections already in zoom-raw.json")
    sub.add_parser("validate")
    sub.add_parser("query")
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    sub.add_parser("all")
    args = ap.parse_args()
    if not args.name:
        args.name = _default_name(args)
    if args.cmd == "all":
        for step in ("chunk", "extract", "resolve", "link", "assemble", "validate", "query"):
            print(f"\n=== {step} ===")
            args.cmd = step
            if step == "extract":
                args.chunks, args.no_retry, args.force = "", False, False
            if step == "link":
                args.entities = ""
            globals()[f"cmd_{step}"](args)
    else:
        globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
