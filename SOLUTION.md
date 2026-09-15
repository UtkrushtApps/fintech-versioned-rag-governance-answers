# Solution Steps

1. Update version-scoped evidence selection by changing retrieval and graph expansion to use the application’s signed_at snapshot (not the “latest” revision).

2. Implement authorized revision scoping in `GraphRepository` with `disclosure_revisions_for_context(...)` and `disclosure_revision_for_context(...)` SQL queries that enforce effective/superseded ranges and match product_id + jurisdiction.

3. Harden retrieval in `HybridRetriever.retrieve(...)`: merge dense/sparse hits as before, then filter strictly by chunk `revision_id` (fallback to `document_id`), rejecting evidence that cannot be mapped to the authorized signing-time revision set.

4. Harden graph attribution in `GraphExpander.expand(...)`: select the disclosure revision authorized at signing and include the associated disclosure document node so governance edges (e.g., jurisdiction governance) are attributable.

5. Prevent cache leakage by scoping cache keys to the signing snapshot: update `RetrievalCache` to include a context fingerprint (product_id, jurisdiction, signed_at) and the configured index_id.

6. Improve provenance/audit traceability by enriching `ContextAssembler.assemble(...)`: store selected `revision_ids`, resolve document provenance rows from `document_provenance`, and attach index manifest details (`index_manifests`).

7. Strengthen grounding signals in the API: after generation, extract citation identifiers (e.g., `C1`, `C2`) from the answer and add warnings if none of the expected citations appear.

8. Refactor and upgrade evaluation to keep quality dimensions separable: compute `retrieval_scope`, `graph_attribution`, `citation_grounding`, and `answer_faithfulness` using deterministic checks. Attempt answer generation only when provider keys are configured; otherwise mark faithfulness as failed with a clear reason.

