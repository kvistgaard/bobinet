# Changelog

## 0.2.0 – unreleased

The first public version.

### Added

- Extraction of entities and relations, each relation with its source sentence
- One named graph per relation, described with PROV-O
- Entity linking to DBpedia and Wikidata that compares descriptions
- `--iri-policy` to name linked entities by Wikidata, DBpedia or local IRIs
- `extract --chunks` and `--force` to redo sections
- `link --entities` to re-link named entities
- `validate`, which checks shapes, terms and quotes
- `query` and `ask` to question the graph
- `zoom` to unfold influence claims into the ideas behind them
