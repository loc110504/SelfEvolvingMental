Status: DRAFT FOR CHECKPOINT C
Source commit: `0e2fc8ff27552845ae743e3373351120a96e216b`
Depends on: Docs 01-05; supplied PsyVEC paper
Blocks: Phases 4-8; final-test authorization
Open decisions: authorized corpora/splits; profile adjudication; simulator families; statistical endpoints; safety rubric

# Simulator, evaluation, and reproducibility

## Simulator integrity specification

```text
authorized Transcript
 -> ProfileBuilder(versioned, audited)
 -> FrozenParticipantProfile(facts + explicit unknowns + hidden label refs)
 -> ParticipantSimulator(versioned)
 -> ReplayEngine(same state/profile/config; deterministic or common seeds)
```

ProfileBuilder maps every fact to source spans, separates unknown fields, validates contradictions, and freezes a content hash. Demographics are unknown unless supported; no plausible inference is permitted. Item labels are separate hidden references, unavailable to policy and ordinarily unnecessary to simulator.

Simulator answers only from visible frozen facts and permitted paraphrases. Unsupported questions yield an approved uncertain/refusal answer. Each response records profile/model/prompt/decoding/seed hashes. Primary matched comparisons use deterministic mode where the backend supports it. Stochastic sensitivity uses common seeds across alternatives/methods. “Clear memory” prompts do not establish isolation; each replay uses a fresh or explicitly reset session verified by tests.

Branch invalidation occurs for profile contradiction, cross-branch inconsistency attributable to simulator, unsupported clinical fact, malformed response, safety failure, or invalid state transition. Invalid branches are reported, not repeatedly regenerated until favorable.

## Split and leakage protocol

| Partition | Permitted use | Forbidden use |
|---|---|---|
| Evolve | collection, hard states, branches, pairs, lesson candidates | final claims |
| Internal development | thresholds/hyperparameters, lesson transfer validation, regression/model selection | pair/candidate source unless separately partitioned |
| Untouched final test | one frozen-protocol evaluation | evolution, retrieval index, tuning, promotion |
| External corpus | approved direct-transfer/frozen evaluation; optional separately labeled recalibration | hidden target selection |

Participants are disjoint. Candidate-lesson validation excludes its source participant and uses only permitted matched states. Retrieval/profile/response caches and indices are rebuilt per split/version; test utterance, profile, embedding, lesson, or cached response cannot enter evolve/development artifacts. Split manifest records participant IDs/hashes and authorization, but released reports de-identify them.

## Evaluation levels

### Unit

Schema/state invariants; projection/access denial; role/alias/adapter mapping; memory lifecycle; profile unknown handling; deterministic seed plumbing; utility components and safety gate; local/transfer gate boundary; split membership; manifest hashing.

### Integration

One topic; complete assessment; hard state and matched branch set; accepted/rejected contrast; source-excluded lesson validation; retrieval poisoning rejection; one role-DPO smoke update; candidate regression rejection/promotion; Memory ON/OFF pairing.

### Regression

Characterized upstream control flow; question/stopping/scoring/report outputs under frozen fixture; updated and non-updated role matrix; safety cues; question cost/redundancy; adapter bleed; backend parity; rollback.

### Research evaluation

| Family | Metrics |
|---|---|
| Assessment | PHQ total/item MAE, QWK, Macro-F1; sensitivity/specificity/AUROC where preregistered |
| Interview | evidence coverage, valid questions, slots/question, tokens, redundancy, premature stop |
| Evolution | accepted-pair rate, pair yield/call, lesson promotion/precision, negative transfer, retrieval hit/action-change |
| Internalization | Memory ON gain, Memory OFF parametric gain, IAR |
| Simulator/safety | profile consistency, contradiction/refusal/invalid rate, unsafe question, missed escalation cue |
| Compute | assessor/simulator tokens, branch calls, latency/wall time, GPU-hours, peak memory, trainable params, storage |

The paper identifies participant-level total-score MAE under a fixed maximum question budget as its planned primary endpoint, but **[DECISION/TBD]** the project must preregister its actual endpoints and statistics before final test. Placeholder tables/figures/numbers in paper pages 11-17 are not results and must never populate reports.

## Comparison and statistical rules

Compare interactive systems at matched budgets and common participant/profile/seed schedules. Passive full-transcript methods are contextual only, never question-efficiency competitors. Report paired uncertainty, failures, invalids, and negative results. Paper-proposed bootstrap/permutation/McNemar/Holm methods are candidate protocol details pending data/statistical review; do not infer significance without measured participant-level outputs.

## Reproducibility manifest

Every run records code/upstream SHA; dataset authorization and split IDs/hash; prompt/evidence/profile-builder/simulator/safety versions; base/tokenizer; accepted/candidate adapter hashes; lesson-memory/index versions; seeds; generation/training configs; backend and software versions; hardware/precision/quantization; branch caches; accepted/rejected contrasts/lessons/checkpoints; tokens, time, failed runs, and checkpoint decision. Artifacts are content-addressed; a rerun checks manifest compatibility before cache reuse.

## Leakage and adversarial tests

- Canary labels/hidden facts cannot appear in agent prompts, retrieval keys, lesson text, or training inputs.
- Test participant hashes fail evolution/training ingestion.
- Raw ExperienceBuffer cannot be passed to inference interfaces.
- Rejected/quarantined lessons cannot retrieve even if semantically perfect.
- Simulator cannot answer seeded unsupported fact except unknown/refusal.
- Cross-case raw phrase/identifier scan and retrieval-index lineage audit.
- Memory poisoning with plausible rejected lesson; adapter mismatch/concurrency test.

## Acceptance criteria

Before Phase 4: approved profile/unknown/access contracts and deterministic replay proof. Before Phase 6: accepted-pair provenance and split tests. Before final evaluation: protocol, endpoints, thresholds, splits, model/checkpoints, prompts, indices, and seeds frozen; final-test access logged; no reruns chosen by outcome. A research claim must link to measured artifacts and state its scope.

## Source-backed facts

- **[PAPER]** The planned protocol requires participant-disjoint splits, source-excluded validation, common randomness, simulator audit, matched budgets, and reporting of invalid branches.
- **[UPSTREAM]** Current full-transcript simulator, co-located labels, missing split manifest, and ad hoc evaluation are documented in Doc 00.

## Engineering decisions

- **[DECISION]** Profile-first simulation replaces full-transcript prompting; unsupported facts fail validity.
- **[DECISION]** Final-test isolation is enforced by data lineage and access, not convention alone.

## TBD / unresolved

- **[TBD]** Dataset authorization and exact participant manifests; profile construction/adjudication; simulator model(s); deterministic guarantees.
- **[TBD]** Safety/escalation rubric, primary/secondary endpoints, confidence/statistical plan, budgets, and human-study review/ethics requirements.
