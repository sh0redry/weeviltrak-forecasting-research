# WeevilTrak Agent 中文上下文

本文件与 `AGENTS.md`、`CLAUDE.md` 保持同一当前事实；发生冲突时，以
`app/config/weeviltrak_v2.4.yml` 和当前 `app/` 代码为准。

## 当前模型

- 当前正式配置：`app/config/weeviltrak_v2.4.yml`；不要为新产物使用 v2.5。
- 架构：LightGBM Base Layer + OOF-trained learned Ridge termination。
- 目标：`signed_days_to_event`；正式输出为 Stage 1、Phase 1、Stage 2、
  Stage 3 / Phase 2。
- canonical IDs：`1, 4, 2, 3`。Phase 1 是直接预测目标，历史训练标签来自
  成对 Stage 1/2 的 midpoint；不得由 observed Phase 1 伪造 Stage 1/2 标签。
- 训练窗口固定为最近 3 年，direct-label coverage gate 失败时不得临时放宽；
  应使用最后批准的 artifact 做 prediction-only。

## 关键数据规则

- Redshift 原始 Stage/Phase 含义在 2026 后不稳定，必须使用
  `app/services/canonical_events.py`。
- 11 条 curated manual events 在运行时注入，不写回 Redshift。
- `12/31` 是占位日期，不是 observed event；Excel–Redshift 已知存在
  2020/2023/2024 月日交换批次。详见
  `docs/data/historical_data_contract_and_risks.md`。

## 新人入口

1. `README.md` 与 `AGENTS.md`
2. `docs/project/2026_project_handoff.md`
3. `docs/architecture/weeviltrak_model_process_overview.md`
4. `docs/operations/v24_training_release_runbook.md`
5. `manual/README.md`（客户交付与本地 pickle 推理）

## 工程边界

- `outputs/` 是本地生成证据，不是 source of truth。
- 网络、S3、Redshift、artifact 发布均可能影响外部状态；先读 runbook，
  不要为 notebook 结果直接修改数据或覆盖 artifact。
- 使用 `create_model_manager` 和 manager 的 `predict()`；不要直接调用底层
  sklearn/LightGBM estimator。
- Python 3.11 + Poetry：`poetry install`；测试：`poetry run pytest tests/ -q`。

完整文件定位、环境变量、命令和高价值测试见 `AGENTS.md`；本文件不重复维护
逐行实现细节，以避免中英文文档再次漂移。
