Status: DRAFT FOR CHECKPOINT C
Source commit: `0e2fc8ff27552845ae743e3373351120a96e216b`
Depends on: Docs 01-03; supplied PsyVEC paper
Blocks: evolution implementation; Docs 06-07 final approval
Open decisions: development-selected thresholds/weights; attribution policy; lesson governance reviewers

# PsyVEC evolution algorithm specification

## Governing invariants

**[PAPER]** No raw trajectory is consolidated. Collection uses a fixed accepted policy and lesson-memory snapshot. Branch alternatives share starting state and frozen profile. Safety is a hard gate. Local preference validity and cross-state lesson transfer are separate. Only the responsible role is trained. Evolution is offline.

**[DECISION]** Config keys `hard_state_threshold`, role utility weights, `pair_margin`, `attribution_margin`, `minimum_transfer_support`, `transfer_win_rate`, `promotion_margin`, `minimum_role_pairs`, and regression tolerances are `TBD/development-selected`; their values and selection run are frozen before final test.

## One assessment decision

```text
decide(state, role, policy_snapshot, lesson_memory):
  assert state is policy-visible and label-free
  lessons <- retrieve promoted lessons filtered by role/scope
  binding <- router.resolve(role, policy_snapshot)
  output <- binding.generate(project(state, role), lessons)
  action <- parse_and_validate(role, output)
  patch <- Updater.propose_patch(state, role, action, output)
  post <- validate_and_apply(state, patch)       # atomic or unchanged on failure
  emit DecisionEvent(hash(state), role, action, output, hash(post), provenance)
  return post
```

## Hard-state mining

For a role decision, compute normalized missing-critical-evidence `m`, role uncertainty `u_r`, and coordination risk `c`; `h_r = alpha*m + beta*u_r + gamma*c`, with weights summing to one. Role uncertainty must be operationalized audibly (for example scorer distribution entropy or evaluator boundary proximity). Coordination includes disagreement, duplication, or stale/inconsistent state. Select only if score meets the development-selected threshold and the state is eligible under split/privacy/budget rules. Compare against random-state branching at identical branch-call budget.

## One branch set and validity

```text
branch_set(hard_state, K, accepted_policy, frozen_profile, simulator_cfg):
  assert K and generation strategies were frozen in run config
  candidates <- diverse_plausible_actions(hard_state, accepted_policy, K)
  for candidate in candidates:
    assert same state_hash, profile_hash, policy_id, simulator_version
    result <- replay(candidate, common_seed_policy)
    result.valid <- safe(candidate) AND profile_consistent(result)
                    AND no_unsupported_fact(result) AND state_transition_valid(result)
  return all results, including invalid ones and reasons
```

The paper's default four strategy composition is evidence, not a mandatory constant: policy/low-temperature, exploratory, diversity-prompted, and compatible lesson-guided. **[DECISION]** `K` and exact strategies remain configured experiments.

## Role-conditioned utility

Use a common decomposed skeleton `U_r = lambda_task*G_task_r + lambda_evidence*G_evidence_r - lambda_cost*C_r`, evaluated only when `Safe=1`.

| Role | Credited gain | Principal failure/cost |
|---|---|---|
| Interviewer | critical-slot acquisition, score-error reduction, relevance | duplicate/unnecessary/unsafe question |
| Evaluator | continue with missing critical evidence; stop when sufficient | premature stop/unnecessary continuation |
| Scorer | normalized item-error reduction with sufficient evidence | error despite sufficient evidence |
| Updater | correction of stale/inconsistent state | unsupported or omitted revision |

Labels are hidden environment supervision for offline utility only and are tagged in `UtilityBreakdown`; evidence-only ablation is required.

## Local contrast verification and attribution

```text
verify_local(results, role, config):
  admissible <- safe AND valid results
  chosen <- argmax utility(admissible)
  rejected <- plausible lower-utility, non-trivial alternative
  checks <- delta >= pair_margin; consistent; nonduplicate; attributed(role)
  return VerifiedContrast if all checks else rejection with reasons
```

Explicit role branching is directly attributed. For ambiguous terminal failures, apply diagnostic mapping (missing evidence -> Interviewer; premature stop -> Evaluator; scoring error with sufficient evidence -> Scorer; stale state -> Updater). If still ambiguous, replay one candidate role decision while holding all other role policies fixed. Exclude parameter supervision unless one role clears the configured attribution margin.

## Fast path: lesson lifecycle

Distill from both matched arms, not a single failure: `(role, trigger, do, avoid, criterion, scope/exceptions, confidence)` plus provenance. Candidate text must not contain labels, participant identity, or raw quotations unnecessary to the rule.

```text
validate_transfer(candidate, accepted_policy, current_memory):
  states <- compatible evolve states excluding source participant/case
  for state in states:
    control <- replay(state, memory=current_memory, common conditions)
    treatment <- replay(state, memory=current_memory + candidate, same conditions)
    record paired utility delta and safety
  promote only if support, win rate, mean gain meet development config
                  AND zero safety regression
  otherwise reject/quarantine; never retrieve candidate
```

Promotion creates an immutable version. Near duplicates merge evidence; unresolved conflict quarantines both recommendations from retrieval; promoted lessons retire when approved monitoring shows non-positive paired benefit or internalization makes retrieval unnecessary. All prior versions/evidence remain auditable.

## Slow path and role DPO

Every locally verified contrast can form `(state, role, chosen, rejected)` even if its lesson fails transfer. Partition by role and accepted-policy snapshot; remove ambiguous, duplicate, invalid, leaked, or test-participant pairs.

```text
update_role(role, pairs, accepted_policy, train_cfg):
  if count(pairs) < minimum_role_pairs: return NO_UPDATE
  assert every pair.role == role and pair.policy_id == accepted_policy.id
  candidate <- train_role_adapter(role, accepted_policy, pairs, train_cfg)
  assert backbone and all non-target adapters are byte/parameter unchanged
  return candidate with manifest
```

The DPO reference for evolution round `e` is the currently accepted role policy at that round. Only after the assembled candidate passes regression does it become next round's accepted reference.

## Complete evolution round

```text
evolve(round_e):
  freeze accepted policy P_e, lesson memory M_e, configs, splits
  B_e <- collect role-tagged evolve interviews with P_e, M_e (no updates)
  H_e <- mine eligible hard decisions
  C_e <- branch and locally verify under matched conditions
  independently:
    M_candidate <- distill + source-excluded transfer-validate lessons
    A_candidate <- role-local DPO for roles with enough accepted pairs
  P_candidate <- assemble accepted unchanged roles + candidate role adapters
  regression <- evaluate P_candidate on participant-disjoint development data
  if pass: atomically accept P_candidate else retain P_e
  independently publish only transfer-approved lesson-memory transaction
  run Memory ON/OFF + IAR audit; close immutable round manifest
```

## Regression accept/reject

```text
gate(candidate, accepted, dev_manifest):
  paired <- evaluate both on identical participants/profiles/seeds/budgets
  pass <- quality/cost/redundancy objective meets configured criterion
          AND no safety regression
          AND every non-updated role meets non-inferiority tolerance
          AND manifests/leakage/parameter-diff checks pass
  write RegressionResult and CheckpointDecision
  promote atomically iff pass; rejected artifacts remain non-routable
```

## Memory ON/OFF and Internalized Action Rate

Evaluate initial/evolved policies with memory ON and OFF on identical held-out states and seeds. Combined ON gain is retrieval plus policy; evolved-vs-initial OFF gain is parametric. IAR is the fraction of held-out triggered states where evolved memory-OFF action belongs to the validated positive-action set. Claim internalization only if both memory-OFF quality and IAR improve under the approved analysis.

## Failure and audit rules

Insufficient valid branches, ties below margin, unsafe output, unsupported simulator fact, inconsistent profile, missing attribution, insufficient transfer support, or split/provenance violation yields no consolidation. Missing roles may skip an evolution round. Failed runs, invalid branches, rejected lessons/pairs/checkpoints, tokens, latency, and resource use remain in the audit manifest; they are not regenerated away.

## Source-backed facts

- **[PAPER]** Sections 3.1-3.5 support the lifecycle, two gates, two independent paths, role routing, accepted-policy reference, regression, ON/OFF, and IAR semantics.
- **[UPSTREAM]** None of these offline evolution modules exists at the audited SHA.

## Engineering decisions

- **[DECISION]** Atomic state/memory/checkpoint transitions and fail-closed consolidation operationalize paper semantics.
- **[DECISION]** Exact action categories and validators are versioned configuration/contracts, not model free text.

## TBD / unresolved

- **[TBD]** All numerical weights, thresholds, budgets, margins, support counts, and non-inferiority criteria.
- **[TBD]** Approved uncertainty estimators, safety validator, attribution-map details, and lesson governance authority.
