"""Prompt templates for the bobinet pipeline.

The extraction prompt follows the Anthropic knowledge-graph cookbook
(central-entities-only, grounded descriptions, no dangling edges) with two
RDF-specific changes: the predicate vocabulary is closed (Text2KGBench-style
ontology conformance) and every relation must carry a verbatim evidence
quote, which the pipeline verifies mechanically against the source text.
"""

ENTITY_TYPES = ["PERSON", "ORGANIZATION", "WORK", "CONCEPT", "EVENT", "PLACE"]

WORK_TYPES = ["BOOK", "ARTICLE", "WORKING_PAPER", "BROADCAST", "OTHER"]

# predicate -> (domain, range, gloss)
PREDICATES = {
    "influencedBy": ("PERSON", "PERSON",
        "The subject's thinking was genuinely shaped by the object: learned from, "
        "adopted, built on, extended. ONLY when the text asserts real intellectual "
        "influence. Mere opposition, criticism, or being an adversary is NOT "
        "influence - do not file antagonists as influences."),
    "academicAdvisor": ("PERSON", "PERSON",
        "The object formally taught or supervised the subject at a university."),
    "knownFor": ("PERSON", "CONCEPT|WORK",
        "The subject is known for originating or developing the object concept or work."),
    "memberOf": ("PERSON", "ORGANIZATION",
        "The subject was a member of the object organization or society."),
    "affiliation": ("PERSON", "ORGANIZATION",
        "The subject held a position at or was employed by the object institution."),
    "founder": ("ORGANIZATION", "PERSON",
        "The subject organization was founded by the object person."),
    "author": ("WORK", "PERSON",
        "The subject work was written by the object person."),
    "editor": ("WORK", "PERSON",
        "The subject work was edited (or introduced as editor) by the object person."),
    "publisher": ("WORK", "ORGANIZATION",
        "The subject work was published by the object organization."),
    "about": ("WORK", "CONCEPT|PERSON",
        "The subject work is about the object concept or person."),
    "attendee": ("EVENT", "PERSON",
        "The object person attended the subject event."),
    "cites": ("WORK", "WORK",
        "The subject work cites the object work (use a typed variant below when the text supports it)."),
    "citesAsEvidence": ("WORK", "WORK",
        "The subject work cites the object work as evidence for a claim."),
    "critiques": ("WORK", "WORK",
        "The subject work critiques the object work."),
    "disputes": ("WORK", "WORK",
        "The subject work disputes claims made in the object work."),
    "agreesWith": ("WORK", "WORK",
        "The subject work expresses agreement with the object work."),
    "extends": ("WORK", "WORK",
        "The subject work extends or builds on the object work."),
    "related": ("CONCEPT", "CONCEPT",
        "The two concepts are substantively related in the text."),
    "broader": ("CONCEPT", "CONCEPT",
        "The object concept is broader than / generalises the subject concept."),
}


def predicate_table() -> str:
    lines = []
    for p, (dom, rng, gloss) in PREDICATES.items():
        lines.append(f"- {p}  ({dom} -> {rng}): {gloss}")
    return "\n".join(lines)


EXTRACTION_PROMPT = """You are an information-extraction engine populating an RDF knowledge graph. Extract entities and relations from the document section below.

<document_section id="{chunk_id}" heading="{heading}">
{text}
</document_section>

ENTITY TYPES - use exactly these strings:
{entity_types}

PREDICATES - a closed vocabulary. Use exactly these strings; never invent a predicate. A fact that fits no predicate is simply not extracted as a relation (it may still justify an entity and its description).
{predicates}

Respond with STRICT JSON only - no markdown fences, no commentary - matching:
{{"entities": [{{"name": "...", "type": "PERSON", "description": "...",
                "birthYear": "1899", "deathYear": "1992",
                "pubYear": "1944", "workType": "BOOK"}}],
  "relations": [{{"subject": "...", "predicate": "influencedBy", "object": "...",
                 "evidence": "...", "confidence": 0.9,
                 "influenceNote": "..."}}]}}
birthYear/deathYear only for PERSON when stated; pubYear/workType only for WORK; influenceNote only for influencedBy relations. Omit optional keys rather than sending null.

Guidelines:
- Extract only entities that are central to what this section is about - skip incidental mentions.
- For each entity, write a one-sentence description grounded in this section (it is used later to disambiguate entities with similar names).
- Use the fullest form of a name that appears in the section.
- Every relation must connect two entities from your entities list, with types matching the predicate's domain and range.
- evidence: a quote of at most 250 characters copied VERBATIM from the section text - character for character, no paraphrase, no ellipsis you added - that supports the relation.
- confidence: 0.0-1.0 that the section itself asserts the relation as fact. Lower it when the section reports the claim as contested, secondhand, unverifiable, or merely someone's opinion, and say why in influenceNote (for influence relations) or by choosing a lower value.
- influenceNote: for influencedBy, one sentence on the nature or channel of the influence in the text's own terms.
- The section may discuss data-quality problems in some influence list (antagonists misfiled as influences, contaminated entries). Statements ABOUT a defective list are not assertions of influence: extract from them only what the section itself endorses as real."""


def extraction(chunk_id: str, heading: str, text: str) -> str:
    return EXTRACTION_PROMPT.format(
        chunk_id=chunk_id,
        heading=heading,
        text=text,
        entity_types=", ".join(ENTITY_TYPES),
        predicates=predicate_table(),
    )


RETRY_SUFFIX = """

Your previous attempt failed validation:
{errors}

Produce the corrected STRICT JSON now. Evidence quotes must be exact verbatim substrings of the section text above."""


# stance -> gloss, for the zoom stage (idea-level unfolding of influence edges)
ZOOM_STANCES = {
    "took": "The person genuinely took over the idea, method or problem from its "
            "originator: adopted it, learned it, built with it.",
    "rejected": "The person explicitly rejected the idea. Opposition, not influence.",
    "reformulated": "The person took the problem or idea and rebuilt it in different "
                    "terms (e.g. a calculation problem recast as a knowledge problem).",
    "extended": "The person carried the idea further than its originator did (ran the "
                "test the originator declined, took the agenda to new ground).",
    "disputed": "The person contested the idea or its attribution without wholesale "
                "rejection.",
}


ZOOM_PROMPT = """You are unfolding the coarse influence edges of an RDF knowledge graph into idea-level statements. The document section below is the evidence base. For each influence pair listed, extract WHAT was transmitted, rejected, reformulated, extended or disputed - one statement per idea and stance.

<document_section id="{chunk_id}" heading="{heading}">
{text}
</document_section>

INFLUENCE PAIRS - the coarse edges to unfold (person influencedBy person):
{pairs}

STANCES - a closed vocabulary. Use exactly these strings:
{stances}

KNOWN CONCEPTS - concepts already in the graph. When the idea is the same, reuse the exact name from this list; otherwise coin a short noun-phrase name:
{concepts}

Respond with STRICT JSON only - no markdown fences, no commentary - matching:
{{"statements": [{{"pair": 1, "person": "...", "stance": "took",
                 "idea": "...", "ideaDescription": "...",
                 "ideaOriginator": "...", "evidence": "...", "confidence": 0.9}}]}}

Guidelines:
- pair: the number of the influence pair the statement unfolds.
- person: the pair member whose stance it is - normally the influenced side; write the name exactly as it appears in the pair listing.
- stance: one of the closed vocabulary above; never invent a stance.
- idea: the specific idea, method, problem, distinction or theory that moved. It must be statable as a claim, a method, a problem or a distinction. These are NOT ideas: a book or essay title, including a translated one; a person; a whole field, discipline or school (monetary theory, capital theory, political philosophy, socialism, Austrian economics); an organisation or the building of one; an act, decision or event ("the reinterpretation of X", "the rejection of Y") - name the underlying claim or question instead. Where a sentence is about a book, name the idea the book carried, or extract nothing from it.
- Name the idea in words this section uses. Do not import a term from elsewhere in the document or from your own knowledge: if the section does not use the term, it is not evidence for it.
- ideaDescription: one sentence describing the idea, grounded in this section. Where the section hedges - calls the claim contested, reported, secondhand, or says it could not verify a quotation - carry that hedge into this sentence rather than leaving it to the confidence number alone.
- ideaOriginator: OMIT THIS unless the section itself says whose idea it is and names them in the quote you are citing. Most statements should omit it. Never fill it in with the pair's influencer merely because they are the influencer - the pairing is what is being explained, so using it as the answer explains nothing. An idea the section treats as already current, or as belonging to a tradition rather than a person, has no originator to record.
- evidence: a quote of at most 250 characters copied VERBATIM from the section text - character for character, no paraphrase, no added ellipsis - that supports the statement. Quote the sentence that states the stance, the one carrying the verb or its close synonym, not a neighbouring sentence that merely sounds related. A quote that only restates the idea is not evidence that anyone took, rejected or disputed it.
- confidence: 0.0-1.0 that the section itself asserts the statement as fact; lower it when the section reports the claim as contested, secondhand or reconstructed after the fact.
- Check the direction before choosing a stance. An idea the section says was dislodged, refuted, abandoned or argued away was not "took". A book that changed someone's mind transmits the argument it makes, not the position it attacks.
- Check the direction of the attribution too. For reformulated, extended and disputed, the person named in "person" is the one acting; the idea is the raw material they acted on, so it belongs to whoever produced that material, never to the actor. An idea someone reformulated is not their own idea.
- Do not emit both "took" and "extended" for one idea unless the section evidences the reception and the later development separately, with different quotes.
- Extract only what this section states about the listed pairs. A rich relationship section typically yields 3-8 statements per pair; a pair the section says little about may get none."""


def zoom(chunk_id: str, heading: str, text: str, pairs: str, concepts: str) -> str:
    stances = "\n".join(f"- {s}: {g}" for s, g in ZOOM_STANCES.items())
    return ZOOM_PROMPT.format(chunk_id=chunk_id, heading=heading, text=text,
                              pairs=pairs, stances=stances, concepts=concepts)


RESOLVE_PROMPT = """Below are {entity_type} entities extracted from several sections of one document. Some are different surface forms of the same real-world entity.

<entities>
{entity_list}
</entities>

Cluster them. Each input name must appear in exactly one cluster's aliases list. Entities that are genuinely distinct get their own single-element cluster. Use the descriptions to avoid merging entities that merely share a name. The canonical name should be the most complete, unambiguous form.

Respond with STRICT JSON only - no markdown fences - matching:
{{"clusters": [{{"canonical": "...", "aliases": ["...", "..."]}}]}}"""


def resolve(entity_type: str, entity_list: str) -> str:
    return RESOLVE_PROMPT.format(entity_type=entity_type, entity_list=entity_list)


LINK_ADJUDICATE_PROMPT = """You are validating candidate links from a document's entities to {registry}. For each numbered item, decide whether one of the candidates denotes the same real-world entity as the document entity.

<items>
{items}
</items>

Rules:
- A name match alone is NEVER enough. Confirm a candidate only when its description is consistent with the document entity's type and description.
- People are the dangerous case: many humans share a name. If the candidate's description points to a different occupation, era, or field than the document's description, it denotes someone else.
- Verdicts: "matched" - exactly one candidate fits the document's description; "rejected" - the descriptions show that the candidates denote other entities; "unsure" - the identity cannot be decided. Prefer rejected or unsure over a guess.
- A candidate with a missing or thin description may still be matched when its name is distinctive (a multi-word name or title unlikely to have namesakes), it passed the registry type check, and no rival candidate fits. Reserve unsure for common names without descriptions and for conflicting candidates.
- Candidates that are the same work at different bibliographic levels (the work itself vs an edition, reprint or translation of it) are not rivals: prefer the work-level candidate.
- "match" carries the matched candidate's identifier exactly as written in the item (e.g. {example}); null unless the verdict is "matched".

Respond with STRICT JSON only - no markdown fences - matching:
{{"decisions": [{{"id": 1, "verdict": "matched", "match": "{example}", "why": "..."}},
                {{"id": 2, "verdict": "unsure", "match": null, "why": "..."}}]}}"""


def adjudicate(registry: str, example: str, items: str) -> str:
    return LINK_ADJUDICATE_PROMPT.format(registry=registry, example=example, items=items)


ASK_PROMPT = """Answer using only the knowledge graph facts below. Cite the specific facts that support your answer; if the graph does not contain the answer, say so explicitly.

<graph>
{graph_context}
</graph>

Question: {question}"""


def ask(question: str, graph_context: str) -> str:
    return ASK_PROMPT.format(question=question, graph_context=graph_context)
