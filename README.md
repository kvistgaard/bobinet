# bobinet

RDF knowledge graphs from text, with the source sentence behind every statement.

Give Bobinet a Markdown document and it writes a TriG graph. It uses Claude to extract the entities and relations, published ontologies supply the terms, and
entities link to Wikidata and DBpedia.

It's initial use case was in support of the Thick Edges demo (forthcoming publication) vocabulary covers intellectual history: people, works, ideas, and who
influenced whom.

Bobinet follows the four stages of Anthropic's [knowledge-graph
cookbook](https://platform.claude.com/cookbook/capabilities-knowledge-graph-guide):
extract, resolve, assemble and query.

The name comes from [bobbinet](https://en.wikipedia.org/wiki/Bobbinet), a machine-made net.

## Requirements

- Python 3.10 or later: `pip install -r requirements.txt`
- the `claude` CLI on your PATH, logged in. Bobinet calls Claude through it.
- network access to DBpedia, Wikidata and the ontologies' servers

## Run

```bash
python bobinet.py --src note.md --out out all                     # build the graph
python bobinet.py --out out ask "Who influenced whom, and how?"   # answer from the graph
```

`all` runs seven stages, and each also runs alone:

| stage | what it does |
|---|---|
| `chunk` | split the document at `##` headings |
| `extract` | entities and relations per section (Haiku) |
| `resolve` | merge the names that refer to one entity (Sonnet) |
| `link` | link entities to DBpedia, then Wikidata (Sonnet) |
| `assemble` | write `<name>.trig` and `<name>.ttl` |
| `validate` | check shapes, terms and quotes into `validation-report.md` |
| `query` | run `queries/*.rq` into `query-results.md` |

An interrupted `extract` resumes where it stopped. To redo part of a run:

```bash
python bobinet.py --out out extract --chunks s05,s06        # redo two sections
python bobinet.py --out out extract --force                 # redo all sections
python bobinet.py --out out link --entities "Carl Menger"   # re-link one entity
```

## Evidence

For every relation, Claude copies the sentence that supports it. Bobinet
looks for that sentence in the section, ignoring Markdown markup, spacing
and typographic punctuation. If the sentence is not there, Claude gets one
more try. A claim still without its sentence is kept and marked
`kgx:evidenceVerified false`.

Each relation sits in its own named graph, so the provenance graph can
describe it:

```trig
claim:c064 {
    dbr:Milton_Friedman  dbo:influencedBy  dbr:Friedrich_Hayek .
}

# in the provenance graph
claim:c064  a kgx:Claim ;
    kgx:confidence       0.90 ;
    kgx:evidenceVerified true ;
    prov:value           "What Friedman took from Hayek here was not a theorem. It was a demonstration of method" ;
    prov:wasQuotedFrom   <.../doc/hayek-lines-of-influence/section/s08> ;
    prov:wasGeneratedBy  <.../activity/run-20260825T095909Z> .
```

With a name for each statement, you can:

- show the evidence for any edge
- drop one statement and keep the rest
- merge graphs from many documents and still tell which document said what

For tools without named graphs, `<name>.ttl` holds every statement in one
graph. It keeps the quotes but not which statement each belongs to.

## Cost

Every run spends Claude tokens. Bobinet needs Claude to extract relations
in a closed vocabulary, each with its quote. Linkers such as DBpedia
Spotlight and TextRazor spend no tokens. They find and link names, and take
their relations from a vendor's fixed list or from grammar.

A run makes one Claude call per section, a retry where a quote fails, and up
to eight calls for merging and linking.

## Entity linking

A candidate links only when its registry description agrees with the
document's description of the entity. So a document about the jazz
pianist John Hicks does not link to the economist of the same name.

Bobinet tries DBpedia first, then Wikidata. Each verdict is matched,
rejected, or unsure. `links.json` records the verdict, the reason and the
candidates weighed.

## Vocabulary

Claude may use only the 6 entity types and 19 relations in `profile.ttl`,
reused from schema.org, FOAF, SKOS, DCMI Terms, PROV-O, CiTO, FaBiO, DoCO
and the DBpedia ontology. Bobinet drops any relation outside them. It mints
only the three `kgx:` terms.

To extract for another domain, edit the predicates in `prompts.py` and their
IRIs in `bobinet.py`, then update `profile.ttl` and `shapes.ttl` to match.

## Options

| flag | meaning |
|---|---|
| `--src` | the Markdown document; required by `chunk` |
| `--out` | output directory, default `./out` |
| `--name` | basename of the `.trig` and `.ttl`, default the document's slug |
| `--base` | base IRI of the run's data, default `https://personalknowledgegraphs.com/data/bobinet/` |
| `--iri-policy` | `wikidata` (default), `dbpedia` or `local`: whose IRI names a linked entity |
| `--gold`, `--gold-entity` | a DBpedia SPARQL result with one person's influence lists, and that person's label; `validate` reports the overlap |

Unlinked entities get IRIs under `<base>id/` whatever the policy. The
`dbpedia` policy also uses DBpedia-ontology properties and classes where an
equivalent exists. To switch policy, re-run `assemble`, `validate` and
`query`.

## Zoom

`zoom` unfolds a claim into the finer statements behind it, each with its
own quote and named graph. `all` does not run it.

Today `zoom` unfolds influence claims only. It finds the ideas one person
took from another, rejected, reformulated, extended or disputed, and writes
them with the `fr:` vocabulary at `http://velitchkov.eu/def/fr#`.

```bash
python bobinet.py --out out zoom --chunks s05,s08
```

Without `--chunks`, `zoom` takes every section that an influence claim
quotes. It writes `<name>-zoom.trig` and `<name>-zoom.ttl` and leaves the
other files alone. [Nodica](https://github.com/kvistgaard/nodica) displays them with click and wheel zoom.

## Roadmap

- Vocabulary profiles for any domain, with a general profile as the default
- Zoom for any relation a profile names
- The position of each quote in the source
- A quote for dates and other values
- Plain text and HTML input
- Entity linking scored against DBpedia Spotlight and TextRazor
- Candidates from a local linker, and a choice of model, to lower the cost

## License

MIT. See [`LICENSE`](LICENSE).
