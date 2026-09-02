# Post-Event Zero Samples and Zero-Day Termination Plan

## 人话版：我们要做什么

我们接下来准备先做两件最小改动，来解决模型在事件附近和事件之后行为不稳定的问题。

第一件事，是在训练数据里补一小段真实的 post-event 样本。也就是说，某个 stage 已经发生之后，如果后面几天仍然有真实天气特征，我们就把这些天也纳入训练，并把它们的 `days_to_event` 统一标成 `0`。这样模型不只是学会“越来越接近事件”，还会被明确教会“事件发生以后应该停在 0”。

第二件事，是在按天连续预测的场景里，加一个非常保守的 zero-day termination 安全网。它不是用来替代模型，而是在模型已经连续两天给出非常接近 0 的预测时，帮助把后续结果锁定，避免事件边界附近继续来回漂。

## 人话版：为什么这样做合理

当前问题不是模型完全不会接近事件，而是它在事件发生后和事件附近的行为还不够稳定。现在的模型主要学到了“接近事件”，但没有被系统地教会“事件发生后应该继续停在 0”。

这套方案的合理性在于，它不伪造天气数据，也不改变业务输出格式，更不是一上来就重做整个模型架构。我们只是把当前训练监督里缺失的一段“事件后行为”补回来，再加一个很保守的后处理安全网。

换句话说，这是一条最小改动路径：

- 不需要先换成新模型结构
- 不需要先重定义目标值
- 不需要先重做回测框架

但它已经足够直接地针对当前最明显的问题，也就是 post-event positive 和事件附近的 bounce-back。

## Human Summary

The current model is trained mostly on pre-event samples. It learns how to approach a stage date, but it is not explicitly taught how to behave after the event has already happened. This is a major reason bounce-back still appears after stage dates: the model can reduce remaining days as the event approaches, but then drift upward again because post-event zero behavior is weak or missing in the training signal.

The LightGBM monotonic constraint on `cumu_gdd_air` helped, but it was not enough by itself. It reduced bounce-back and lowered post-event positive predictions, but it did not teach the model the semantic rule that once the event has happened, remaining days should stay at zero.

The proposed fix has two parts:

- add short-horizon post-event training samples with targets clipped at zero
- add zero-day termination as a post-processing safety net

The expected effect is:

- fewer post-event positive predictions
- fewer bounce-backs near and after the event
- better alignment between model behavior and the intended business meaning of `days_to_event`

## Final Design Decisions

- Do **not** use negative targets after the event
- Extend training data from `03/01 -> stage_date` to `03/01 -> stage_date + K`
- Use `target = max(days_to_event, 0)`
- First implementation uses a short post-event horizon, not extension to `10/01`
- Recommended first value: `K = 5`
- Zero-day termination is a safety net, not the primary fix
- First zero-day rule:
  - threshold `predicted_days <= 1.0`
  - trigger after two consecutive prediction days satisfy the threshold
  - lock the event to the **first** day in that first qualifying pair
  - once locked, all later predictions are forced to `predicted_days = 0`
  - all later `predicted_stage_date` values equal the locked event date

## 最简单执行步骤

如果只走最小可行路径，这件事可以按 3 步做完：

1. 先只改训练数据构造  
   把每个事件后最多 `5` 天、且有真实天气特征的样本补回训练，并统一设为 `days_to_event = 0`。

2. 再加一个最小 zero-day termination 规则  
   当同一条连续预测轨迹里，连续两天 `predicted_days <= 1.0`，就从第一天开始锁成 `0`。

3. 用现有 `monotonic_business_check.py` 复跑对比  
   主要看：
   - `bounce_back_count`
   - `bounce_back_rate`
   - `post_event_positive_count`
   - `post_event_positive_rate`

这条路径之所以是当前最小、最实际的执行方式，是因为它避免了：

- 新模型结构
- 新目标定义
- 新回测框架

## Codex Execution Plan

### A. Training-data augmentation

Primary file:

- `app/services/data_preparation_service.py`

Implementation intent:

- update training-data preparation so that each `(location_id, pest_year, stage_id)` sequence includes a short post-event extension
- after the real `stage_date`, append daily rows through `stage_date + 5 days`
- label these extra rows with `days_to_event = 0`
- preserve existing pre-event label logic for dates before the event

Required implementation details:

- do not change the public meaning of `days_to_event`
- do not emit negative targets
- do not extend to a global fixed season end such as `10/01`
- keep the extension local to each observed event
- ensure the augmented rows still align with weather feature rows for the same location/date
- if weather rows are unavailable for a post-event day, skip that augmented day rather than fabricating features
- keep downstream feature filtering and null-dropping behavior unchanged after augmentation

Preferred implementation shape:

- augment `model_data` after the weather/weevil merge and before target filtering, or add a helper dedicated to post-event row expansion and zero clipping
- keep the logic isolated enough that horizon length can be made configurable later

Recommended concrete implementation steps:

1. In `prepare_training_data()`, keep the current merge and pre-event label logic intact.
2. Add a helper that identifies the event anchor for each merged `(location_id, pest_year, stage_id)` trajectory.
3. For each trajectory, keep existing rows up to the event as-is.
4. For dates after the event and up to `event_date + 5 days`, retain or append rows only when weather/features exist for that date.
5. Set `days_to_event = 0` for those post-event rows.
6. Ensure the final training frame still drops rows with missing model features.

### B. Zero-day termination post-processing

Primary file:

- whichever model or pipeline path assembles ordered prediction sequences for LightGBM inference, most likely in the prediction/model-manager layer rather than feature engineering

Implementation intent:

- add a reusable post-processing function that operates on ordered per-window prediction sequences
- apply it after raw `predicted_days` are computed, before final result output is assembled where sequence context is available

Rule definition:

- input sequence is ordered by date within one `(location_id, stage_id)` prediction trajectory
- define `near_zero = predicted_days <= 1.0`
- when the first pair of consecutive dates both satisfy `near_zero`, set:
  - locked event date = first date in that pair
- for the locked date and all later dates:
  - `predicted_days = 0`
  - `predicted_stage_date = locked event date`
- if no consecutive qualifying pair exists, leave the sequence unchanged

Important edge-case rules:

- do not require strict `<= 0`
- do not choose the last qualifying zero date
- do not re-open the sequence after lock
- do not apply sequence locking to unordered single-row predictions unless a sequence context exists

Recommended concrete implementation steps:

1. Add a helper that accepts a single ordered trajectory DataFrame.
2. Detect the first pair of consecutive dates with `predicted_days <= 1.0`.
3. Lock at the first date in that pair.
4. Overwrite all later rows in the same trajectory with zero remaining days and the locked stage date.
5. Keep this helper separate from core model inference so it can be switched on only where sequence context exists.

### C. Configuration support

Primary config target:

- `app/config/weeviltrak_v2.2.yml`

Add explicit config entries for future clarity:

- `post_event_zero_days: 5`
- `zero_day_termination_threshold: 1.0`
- `zero_day_termination_consecutive_days: 2`

Implementation note:

- if config plumbing is too large for the first pass, these values may be hardcoded initially
- preferred final state is config-driven behavior

### D. Lightweight business check update

Primary file:

- `app/pipeline/monotonic_business_check.py`

Update the business check so it can validate the new behavior clearly:

- exclude rows with missing `model_version` or missing predictions from summary metrics
- continue reporting:
  - `bounce_back_count`
  - `bounce_back_rate`
  - `post_event_positive_count`
  - `post_event_positive_rate`
- keep detail output unchanged enough to support manual trajectory inspection

Success baseline for comparison:

- `v2.1`: bounce-back `579`, post-event positive `562`
- `v2.2` before this change: bounce-back `449`, post-event positive `516`

## Test Plan

### Unit tests

Add or update tests covering:

- training-target augmentation:
  - pre-event rows keep positive labels
  - event day label is `0`
  - post-event augmented rows through `+5` days are `0`
  - no negative targets are produced
- missing weather on some post-event dates:
  - unavailable days are skipped cleanly
- zero-day termination:
  - no trigger when isolated single low day appears
  - trigger when two consecutive days are `<= 1.0`
  - lock uses the first day in the pair
  - later rows remain forced to zero
  - no later bounce-back after lock
- business-check summary:
  - rows with null `model_version` or null predictions are excluded from metrics

### Lightweight integration check

Re-run the monotonic business check and compare against the existing baseline.

Desired directional outcomes:

- `v2.2` bounce-back lower than current `449`
- `v2.2` post-event positive lower than current `516`
- `v2.2` post-event positive rate lower than current `0.9181`

## Acceptance Criteria

The implementation is complete when:

- training data includes short post-event zero-labeled rows
- no negative `days_to_event` values are used
- zero-day termination locks to the first day of the first qualifying consecutive pair
- locked sequences never bounce back afterward
- business-check summary excludes null or empty prediction rows
- updated `v2.2` outperforms the prior lightweight business-check baseline on at least one of:
  - bounce-back count
  - post-event positive count
- no regression is introduced in existing monotonic-constraint tests

## Assumptions

- The current target semantics remain “days remaining until event,” not signed offset from event
- The first post-event augmentation horizon is fixed to `5` days
- Zero-day termination is first implemented where ordered date sequences are available
- This phase does not yet include GP residual modeling or full backtesting rollout
