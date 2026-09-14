# Prospective v2 development protocol

**STATUS: NOT RUN.** Candidate prepared after inspecting five v1 development reports. It is development, not independent validation. Freeze a reviewed execution recipe before a future model call. Never rewrite v1 outputs or acceptance metrics.

## Unit, scope and comparison

The independent unit is a study. Retain the same five development studies, same order, both models and two repetitions per model. Twelve conditions give 60 correlated cells per run, not 60 independent patients. Repeats measure determinism and are not additional independent samples. The original 40-development/18-validation assignment is unchanged; validation reports/labels and the remaining 4,349 studies are outside this stage. No randomization or new split is introduced.

Compare v2 against v1 descriptively on the same five cases after future execution. Because prompt, acceptance and secondary-generation policies change together, a difference cannot isolate a single cause or medical specialization. Historical runs also differ in environment/time; measure future run order, model loading and hardware. Equal acceptance gates apply to MedGemma and Qwen.

## Four states and clinical scope

`positive`: explicit relevant current evidence satisfies the target, including required anatomy, severity, extent and timing. `negative`: explicit absence of that target or an explicitly below-threshold finding. `uncertain`: relevant but hedged, contradictory or insufficiently specified finding. `not_mentioned`: no relevant finding; empty evidence and confidence zero. Never convert not mentioned or a technical failure to negative.

For positive, negative and uncertain, evidence is required. Confidence is finite numeric 0–1, uncalibrated, and not an acceptance threshold. A low-confidence binary label is still a binary extraction for accounting; confidence does not create validation. Target definitions are [versioned](prompts/target_definitions_v2.txt), and both models use identical [core instructions](prompts/medgemma_prompt_v2.txt).

General prompt changes address wrong compartments, normal-subregion overgeneralization, wrong adjacent structures, insufficient severity/extent, polarity errors and unresolved current/history contradictions. Relevant unspecified severity normally remains uncertain. Unrelated anatomy alone is not mentioned for this target. No study IDs, organizer answers, copied validation failures, case-specific demonstrations or prevalence assumptions are included.

## Deterministic outer-fence handling

Record `raw_output`, `normalized_output`, `normalization_applied`, `generation_status` and `parser_status`. If generation did not complete, reject all condition cells without trying to rescue a complete object inside it.

For a completed response, ignore only surrounding space, tab, CR and LF for recognizing one whole wrapper. The wrapper must start with three backticks followed by either lowercase `json` or no language tag, then LF or CRLF; it must end with LF or CRLF followed by three backticks. Remove only that single opening and closing framing. Preserve the captured contents byte-for-byte in UTF-8, including all internal whitespace. Untyped and `json` wrappers are accepted equally for both models. Other language tags, nested wrappers, prose outside a wrapper, multiple objects and incomplete JSON fail strict parsing. Bare JSON goes directly to strict parsing, retaining its raw string.

Reject duplicate keys at any depth, nonfinite JSON numbers, wrong condition sets, extra fields and invalid value types. Each condition requires exactly `label`, `evidence_text`, `confidence`. A top-level parse/key-set failure rejects the whole response; a malformed condition value or evidence failure rejects that cell. Technical failures have a null accepted label and separate reason. Never extract arbitrary JSON substrings, remove prose, edit a diagnosis, or regenerate a primary response. No secondary generation is planned in this version.

## Evidence matching

For normalized fallback only, trim outer space/tab/CR/LF from the evidence for searching. Collapse each maximal run of those **four ASCII whitespace characters** into one ordinary space in both source and query. Preserve a mapping from every normalized source character to its original start/end. Everything else—words, case, punctuation, accents, non-ASCII whitespace and Unicode composition—remains unchanged. No spelling correction, fuzzy search, translation or semantic similarity.

First search the unchanged evidence literally, including overlapping occurrences. Exactly one exact occurrence is accepted immediately; multiple exact occurrences produce `ambiguous_evidence_error`. Only when there is no exact occurrence, search normalized whitespace. Require exactly one normalized occurrence; zero produces `evidence_error` and multiple produce `ambiguous_evidence_error`. A unique exact quotation takes precedence over other passages that coincide only after normalization. A longer unique quotation is the preferred model behavior. Return the unique contiguous original `source_span`, start inclusive and end exclusive. Offsets are Python Unicode code-point indices in the original report, not UTF-8 byte offsets. Verify `report[start:end] == source_span`.

Keep `model_evidence` unchanged. Record whether source and model strings differ. The mapped source span preserves its original whitespace and punctuation. A query that is literally a substring may omit a surrounding punctuation mark, but punctuation **inside** the chosen query cannot be changed by normalization. Normalization establishes lexical identity only, not clinically correct anatomy, severity or entailment. Every accepted output requires semantic review.

## Software tests versus semantic review

Automated tests cover parsing, normalization identity/determinism, unique spans, offsets, duplicate/schema errors, no silent state conversion, failed-generation rejection and repeatability gates. The synthetic anatomy/severity/temporality cases are reviewer-authored expected decisions. Their tests check serialization and preservation of those expectations; they do not claim a model generated the correct answer. An intentionally wrong positive label attached to a normal statement is used to demonstrate this limitation explicitly.

No ad hoc keyword classifier is added as a substitute for the LLM. For future generation, review actual outputs for wrong structure, missing qualifiers, incorrect polarity, contradictions and unsupported abstentions as well as binary mismatches. Relevant evidence outside the quoted passage may change its interpretation.

## Adjudication and advancement

A private queue includes all 12 previously identified development review items: nine binary organizer discordances, two evidence failures and one questionable uncertain output. These are not 12 proven diagnostic errors. References remain immutable. A qualified human may later document an adjudication, but it stays separate from the original organizer label; absent such a reviewer, all statuses remain unresolved. See [ADJUDICATION.md](ADJUDICATION.md).

Before proposing full development: all 60 first-pass cells must be technically usable under this exact policy; no context overflow, OOM or truncation; all 60 label, model-evidence, matched-span/offset and status tuples repeat exactly; no unexplained evidence failure; explicit manual semantic review finds no systematic anatomy, severity or polarity issue; and remaining discordances are separately classified. Any exception requires a new prospective protocol, not a post hoc waiver to favor a teacher candidate. Passing the engineering gate never authorizes compute or establishes clinical validity.

Report unique-study and cell denominators, bare-versus-fence-normalized counts, exact-versus-whitespace evidence counts, technical failures, four-state distribution, binary decisions/reference matches, unresolved discordances, generation timing/tokens and peak allocated VRAM. Keep valid output, reference agreement and semantic correctness distinct. No full 40-study run is automatic. The previously inspected 18-study holdout cannot become fresh validation; adjudicated independent evaluation is needed before weak-label scale-up.

The primary research endpoint remains correct binary-label yield over all study-condition checks; always report binary coverage, conditional error and both semantic abstentions and technical failures. The current stage reports no such v2 scores. Ambiguous evidence uses the separate `ambiguous_evidence_error` status. Normalization records both an applied boolean and a method/type string.

## Human semantic approval artifact

For each model, require a `review_source: human` artifact with exactly 60 unique first-run case/condition entries. Its `candidate_code_sha256` mapping must equal the current complete fingerprints of protocol, prompts, target definitions, parser/evidence code, model configs and tests/templates. Unchanged parsed rows cannot carry a human approval across a changed experiment. Each includes `case_id`, `condition`, `technical_status`, `semantic_review_status`, `review_category`, `reviewer`, timezone-aware ISO `reviewed_at`, substantive `notes`, and `response_sha256` bound to the exact parsed row. Only `acceptable` can satisfy semantic approval. `not_reviewed`, `model_semantic_error`, `reference_discordance_unresolved` and `needs_image_adjudication` do not. Missing or stale provenance blocks the gate. Matching the organizer never supplies automatic approval. Repeated outputs must independently match the reviewed first outputs. If label, evidence, offsets or technical status differs between repeats, the gate fails; the first-run review cannot authorize using only the preferred repeat.

These fields document a human review claim; software cannot authenticate a clinician's identity or competence. Require an actual qualified review workflow, not an LLM or script filling acceptable entries. Synthetic tests use clearly labeled fictitious reviewer records only. No real review is claimed here.
