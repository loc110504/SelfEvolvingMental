Status: DRAFT FOR CHECKPOINT A
Source commit: `0e2fc8ff27552845ae743e3373351120a96e216b`
Depends on: approved `plan.md`; audited local clone
Blocks: Docs 01-07 and all implementation
Open decisions: runnable baseline environment; licensed dataset access; whether AutoGen remains after baseline phases

# Upstream AgentMental audit

## Audit identity and method

- **[UPSTREAM]** URL: `https://github.com/MindIntLab-HFUT/AgentMental`.
- **[UPSTREAM]** Commit: `0e2fc8ff27552845ae743e3373351120a96e216b`, dated 2026-05-16, subject `Update README.md`.
- **[UPSTREAM]** The clone retains `origin` as metadata only. It was not fetched after cloning, modified, executed against patient data, or pushed.
- **[UPSTREAM]** Audit covered every tracked text/source/config file, the README, dependency manifest, scale JSON, data scripts, entrypoint, evaluation script, and absence of tests.

## Relevant tree

```text
AgentMental/
├── README.md
├── requirements.txt
├── data/{data_download.py,extract_data.py,data_process.py}
├── evaluation/README.md
├── result.py
├── scales/{PHQ-8.json,scoring_standards.json}
└── src/
    ├── OAI_CONFIG_LIST
    ├── agents.py
    ├── assessment.py
    ├── config.py
    ├── data_load.py
    ├── generate_response.py
    ├── logging_setup.py
    ├── main.py
    ├── memory.py
    └── utils.py
```

No `tests/`, package metadata, lockfile, CI, experiment manifest, prompt-version registry, or structured configuration hierarchy exists.

## Dependencies and operability

**[UPSTREAM]** `requirements.txt` pins `openai 1.70.0`, `pandas 2.2.3`, `autogen 0.7.3`, `networkx 3.4.2`, `uuid 1.30`, `python-dotenv 1.0.1`, `pyyaml 6.0.2`, and `tqdm 4.67.1`. Source additionally imports undeclared `requests`, `bs4`, `numpy`, `scikit-learn`, `pingouin`, `scipy`, `matplotlib`, `seaborn`, and pandas Markdown's optional table dependency. The stdlib already supplies `uuid`, making that PyPI pin questionable.

**[UPSTREAM]** README says `python main.py`, but the entrypoint is `src/main.py`; its `data_dir` is empty and relative paths assume execution from `src/`. Data extraction/download paths and evaluation CSV paths are placeholders. Therefore the clone is not runnable without local configuration, authorized DAIC-WOZ data, API endpoints, and dependency repair.

## Current control flow

```text
src/main.py
 -> load scale prompts and scoring standards
 -> choose manual/automated mode
 -> process_single_file for each processed participant JSON
 -> setup four AutoGen ConversableAgents + UserProxy
 -> perform_assessment
    -> collect/infer demographics
    -> for each PHQ-8 topic, ask up to 3 questions
    -> model-assisted graph-memory extraction after each answer
    -> necessity score decides continue/stop
    -> scoring agent emits item score + summary
    -> MemoryGraph completes topic and rechecks prior summaries
 -> SummaryAgent sees full history, scores, and memory
 -> SummaryAgent creates report and may adjust item scores
 -> CSV prediction output; result.py computes offline metrics
```

The loop is procedural and monolithic: mutable lists, graph state, prompts, simulator calls, scoring, output persistence, and error fallbacks live in one function. Decisions do not emit typed pre/post-state events.

## Roles and model/API flow

| Upstream role | Verified behavior | Target interpretation |
|---|---|---|
| `QuestionAgent` | Generates empathetic initial/follow-up questions from identification, last turn, graph memory, and other topics | Interviewer seed logic |
| `NecessityAgent` | Returns 0/1/2 for information sufficiency and severity; procedural rule stops after limits | Evaluator seed logic |
| `ScoringAgent` | Scores one topic from its Q/A history and rubric; emits JSON score/summary | Scorer seed logic |
| `SummaryAgent` | Reads full history and initial scores; creates final summary/recommendations and adjusts scores | ReportingAgent, not the PsyVEC Updater |
| `MemoryGraph._trigger_holistic_reassessment` | Rewrites prior topic summary bases after later evidence; no typed action/event and does not revise full evidence state | Partial updater-like source material only |

**[UPSTREAM]** All AutoGen agents and the group manager call `get_llm_config()` and share the filtered `qwen2.5-72b` config, temperature 0, cache seed 42. `MemoryGraph` and simulator bypass AutoGen and separately instantiate OpenAI clients using the same `API_BASE_URL`, `API_KEY`, and `API_MODEL` variable names but different defaults. There is no role-aware backend, adapter registry, LoRA, or training boundary.

## Memory audit

**[UPSTREAM]** `MemoryGraph` is an in-process `networkx.DiGraph` created per assessment: one User node, Topic nodes, and UUID Statement nodes. It is case-local and graph-based, but statement content is unstructured text produced by another LLM. Completed topic scores/summaries and current statements are serialized into prompts. There is no persistence contract, cross-case retrieval, deletion API, access control, schema validation, or deterministic extractor. It is model-assisted and not a LessonMemory or ExperienceBuffer.

Failure behavior is permissive: API/JSON errors fall back to truncated raw answer summaries; holistic reassessment logs and continues. Provenance does not include prompt/model/version/hash.

## Simulator and data leakage audit

**[UPSTREAM]** `generate_mock_response` places the complete source interview in every simulator system prompt. It has no profile-building step, frozen typed facts, explicit unknown set, contradiction detector, branch identity, replay API, or seed manifest. When demographics are missing, it explicitly asks the model to infer plausible age/gender/occupation. Later prompts advise against fabrication but enforce no fact-level validity gate. Temperature is 0, yet deterministic replay is not guaranteed or recorded.

**[UPSTREAM]** Processed participant JSON co-locates `real_interview` with total/binary/item PHQ labels. `load_real_data` returns labels and `assessment.py` passes `scale_scores` into the simulator, although the audited simulator body never uses that parameter. This is a dangerous interface-level leakage path even without observed use. Policy prompts receive scoring rubrics and case-memory predicted scores, not source ground-truth labels. No split manifest prevents test participants from entering automated assessment.

Required rewrite boundary: `Transcript -> ProfileBuilder -> FrozenParticipantProfile -> ParticipantSimulator -> ReplayEngine`, with labels visible only to the offline utility evaluator and hidden profile facts visible only to simulator/validator.

## Data, scales, evaluation, tests, and secrets

- **[UPSTREAM]** `data_process.py` converts authorized DAIC-WOZ tabular transcripts and PHQ labels into per-participant JSON; no split provenance, schema validation, de-identification check, or leakage guard is recorded.
- **[UPSTREAM]** `PHQ-8.json` contains three example questions for each of eight topics. `scoring_standards.json` defines 0-3 frequency rubrics over two weeks. The question examples are computed but never included in `question_payload`.
- **[UPSTREAM]** `result.py` computes total MAE/Pearson/ICC, binary accuracy/F1/Kappa, and item MAE/accuracy/F1 from placeholder file paths. It has undeclared dependencies and no confidence intervals, paired tests, cost/safety/evidence metrics, or experiment manifest.
- **[UPSTREAM]** There are no automated tests. Errors often default to zero score, empty output, or generic response, which can silently bias evaluation.
- **[UPSTREAM]** Secrets are placeholders in tracked config and otherwise expected in `API_KEY`, `API_BASE_URL`, `API_MODEL`; no `.env` schema or redaction policy exists. Logs contain prompts, answers, memory, model responses, and demographic information.

## Concrete reuse matrix

| Upstream module | Current responsibility | Decision | Reuse details | Required changes | Risk |
|---|---|---|---|---|---|
| `README.md` | Setup overview | KEEP_WITH_REFACTOR | Preserve attribution/setup context | Correct paths, commands, architecture | Medium |
| `LICENSE` | Upstream license | KEEP | Preserve with reused source | Human license review | Low |
| `requirements.txt` | Partial runtime pins | REWRITE | Use as dependency evidence | Add packaging/locks; separate runtime/training/eval | High |
| `src/OAI_CONFIG_LIST` | AutoGen endpoint list | DEPRECATE | Baseline-only fixture | Replace secrets/placeholders with layered config | High |
| `src/config.py` | One global AutoGen LLM config | WRAP | Preserve baseline adapter initially | `ModelBackend`, role router, config validation | High |
| `src/agents.py` | Four prompts and AutoGen objects | KEEP_WITH_REFACTOR | Reuse question/evaluator/scorer prompt intent | Typed I/O; role aliases; split Reporter/Updater | High |
| `src/assessment.py` | End-to-end mutable loop | KEEP_WITH_REFACTOR | Preserve topic/order/stop behavior as characterization baseline | Extract engine/state/events and boundaries | Very high |
| `src/memory.py` | Case-local graph + LLM extraction/reassessment | KEEP_WITH_REFACTOR | Adapt only into CaseMemory migration path | Typed slots, provenance, access/deletion; isolate model call | High |
| `src/generate_response.py` | Full-transcript participant imitation | REWRITE | Retain only behavioral prompt evidence | Frozen-profile simulator/replay/validity | Critical |
| `src/data_load.py` | JSON loading | KEEP_WITH_REFACTOR | Reuse basic parsing | Typed validation, split IDs, separate labels/profile | High |
| `src/utils.py` | Parsing, agent calls, stopping, persistence/report helpers | KEEP_WITH_REFACTOR | Keep pure validated utilities; characterize stopping | Split modules; typed errors; no zero-default evaluation | High |
| `src/main.py` | Interactive/batch CLI | REWRITE | Preserve user workflow intent | Real CLI/config, explicit modes, safe defaults | High |
| `src/logging_setup.py` | File/dialog logs | WRAP | Retain logging integration | Structured traces, redaction, run IDs | Critical |
| `scales/PHQ-8.json` | Topic question examples | KEEP | Protocol seed asset | Validate version/license; wire explicitly | Medium |
| `scales/scoring_standards.json` | PHQ-8 rubric | KEEP | Scale-grounded scoring source | Version/hash; clinical review | Medium |
| `data/data_process.py` | Transcript/label conversion | KEEP_WITH_REFACTOR | Reuse authorized parsing | Schema, manifests, de-identification, split separation | Critical |
| `data/extract_data.py` | Licensed archive extraction | KEEP_WITH_REFACTOR | Optional local preparation | Safe paths, authorization, checksums | High |
| `data/data_download.py` | Placeholder scraper/downloader | DEPRECATE | None by default | Document authorized manual acquisition | High |
| `result.py` | Ad hoc metric script | REWRITE | Reuse metric names as baseline evidence | EvaluationRunner, paired metrics, manifests | High |
| `evaluation/README.md` | One-line output note | REWRITE | None | Evaluation protocol and artifact layout | Medium |
| Tests/CI | Absent | NEW_COMPONENT_REQUIRED | N/A | Unit/integration/regression/leakage tests | Critical |
| PsyVEC evolution/training/lesson modules | Absent | NEW_COMPONENT_REQUIRED | N/A | Add per Docs 02-07 | Critical |

## Major gaps and technical debt

The largest changes are: extract a testable event/state engine from `assessment.py`; eliminate shared/global and direct API coupling; create a real structured Updater while retaining reporting; split four knowledge stores; rewrite simulation for frozen-profile replay; enforce split/field-level leakage; and build the entire verified evolution, LoRA/DPO, regression, and reproducibility surface as new offline components.

## Source-backed facts

- **[UPSTREAM]** Facts above are based on the complete tracked tree at the pinned SHA.
- **[PAPER]** The paper requires four functional roles and distinguishes case memory, lesson memory, policy adapters, and an inference-inaccessible experience buffer.

## Engineering decisions

- **[DECISION]** Preserve AutoGen compatibility through early baseline refactoring; reassess only after stable interfaces exist.
- **[DECISION]** Treat SummaryAgent as ReportingAgent and create a separate typed UpdaterAgent.
- **[DECISION]** Reuse current graph memory only as CaseMemory migration material; rewrite simulator and evaluation paths.

## TBD / unresolved

- **[TBD]** Confirm upstream license and DAIC-WOZ authorized-use obligations with project owners.
- **[TBD]** Pin a reproducible runnable baseline deployment/model and decide whether baseline quirks become characterization fixtures.
- **[TBD]** Decide when, if ever, AutoGen replacement is justified after Phase 3.
