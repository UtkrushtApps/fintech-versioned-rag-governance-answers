# Solution

The starter answers from whatever disclosure text reads most like the question. The bank's
records — the product catalogue, the override rules, the compliance register and the
knowledge graph — already say which documents apply to an application, whether the product
was ever offered in that state, and where the records disagree. None of that reaches the
answer path. The solution makes those records the authority and the text the evidence.

## 1. Resolve what applies, once per request — `app/knowledge.py` (new)

`KnowledgeResolver.resolve(context)` returns an `ApplicableKnowledge` describing the
application:

- **Offered or not.** `product_offerings` rows for (product, jurisdiction) whose window
  covers `signed_at`, cross-checked against `governance_state`. A product the bank never
  offered in that state — or withdrew — resolves to `offered=False` with the reason.
- **Which documents must be read together.** National documents (`jurisdiction = 'ALL'`)
  and the borrower's state addendum, each at the revision in force at signing. Where an
  `override_rules` row says a state addendum overrides a national document, *both* roles
  become required: an answer that omits either is incomplete.
- **What the records disagree about.** Collected as `Conflict` values rather than resolved:
  more than one product node behind the same product key, an offering status the compliance
  register contradicts, a graph edge that is not `asserted`, an override still linked to a
  revision no longer in force, an addendum named by a rule but absent at signing.
- **A fingerprint** over all of the above — the identity of the knowledge the answer rests
  on.

## 2. Retrieve within what applies, and cover every required role — `app/hybrid_retrieval.py`

`retrieve(question, context, knowledge)` merges dense and sparse candidates over a wider
pool, drops anything whose `document_id` is not in the applicable set (this alone removes
another state's addendum and any document not in force at signing), then fills the context
**per required role** before filling by score. `coverage()` reports required roles with no
passage, so an incomplete answer is detectable rather than silent.

## 3. Decline a close match — `app/main.py`

When `knowledge.offered` is false the request never reaches the model: the response says no
disclosure applies, gives the reason and what was checked, carries no citations, and is
audited like any other answer. A near miss is not an answer.

## 4. Carry the disagreements through — `app/main.py`, `app/generation.py`, `app/graph_expansion.py`

Conflicts become response `warnings`, graph `statements`, and a section of the prompt that
instructs the model to state the disagreement and not choose between its sides. Attribution
is built from the revisions actually used plus every product node the key maps to, so a
merged-away product shows up instead of silently replacing the surviving one.
`ContextAssembler` keeps every graph reference in provenance instead of the first two.

## 5. Make answers follow the records — `app/cache.py`

The cache key is `index_id : application_id : knowledge_fingerprint : question`. Correcting
a crosswalk, an offering or an override changes the fingerprint, so the next answer is
recomputed; two applications no longer share one entry.

## 6. A gate that separates failures and repeats — `app/evaluation.py`

Five dimensions scored per case, offline, with no model call:

| dimension | fails when |
|---|---|
| `evidence_completeness` | a required role has no passage in the context |
| `scope_isolation` | a passage comes from a document that does not apply |
| `abstention_correctness` | evidence is assembled for a combination never offered |
| `conflict_visibility` | a disagreement is resolved silently instead of reported |
| `citation_grounding` | a passage used cannot be cited to a source and revision |

The run label is a digest of the results, so unchanged code and unchanged data produce the
same verdict, and a regression in one dimension is visible on its own.

## Verified

On a live stack (Qdrant + PostgreSQL + Redis, 230 chunks, 30 evaluation cases):

```
starter     evidence_completeness 17/30   scope_isolation  2/30   abstention 23/30   conflict_visibility 18/30
solution    evidence_completeness 30/30   scope_isolation 30/30   abstention 30/30   conflict_visibility 30/30
            two consecutive runs -> identical verdict
```

End to end against a real provider: the New York application is answered at $75 citing the
addendum *and* the national schedule; the HomeLine/New York application is declined with the
catalogue's reason; the California HomeLine answer carries the product-identity disagreement
as a warning.
