# CLAUDE.md

Guidance for Claude working in this repository. The specs live in `docs/`:

- `docs/DESIGN.md`: system design (Track A / Track B, evaluation protocol, milestones M0–M6)
- `docs/DATA.md`: data sources, derived concepts, splits, sampling parameters, integrity checks
- `docs/DECISIONS.md`: Architecture Decision Records (ADRs) D-001 … D-012
- `docs/GLOSSARY.md`: industry vocabulary. **Every term used in code, reports and replies must come from here** (see D-008)

Read all three before starting any milestone. When the docs and this file disagree
on *what* to build, the docs win. This file governs *how* to work.

---

## ⚠️ 工作模式：数据到位后，自主跑通 MVP，再交给用户检验

**这是本项目最重要的一条规则。**

1. **触发条件。** `data/raw/` 中存在 `transactions_train.csv`、`customers.csv`、
   `articles.csv`（`sample_submission.csv` 可选）即视为"数据已到位"。
2. **一次性跑通。** 数据到位后，从 M0 一直做到 MVP 完成（范围见下文"MVP 定义"），
   **中途不要停下来向用户汇报进度或请求确认**。只在 MVP 全部完成、结果可复现后，才告诉用户。
3. **自行 debug。** MVP 之前的所有步骤中遇到的报错、内存不足、性能瓶颈、数据异常、
   指标异常，都由 Claude 自行定位根因并修复。不要把报错丢给用户。不要用 try/except
   吞错误、硬编码结果、跳过测试或降低断言来"让它通过"。
4. **选择更好的方法。** 如果发现文档中的某个做法在实际数据上效果差、跑不动或有更优方案，
   Claude 应当自行评估并采用更好的方法，以达到更好的效果。条件是：
   - 用验证周（validation week）的数字证明它更好，而不是凭感觉；
   - 在 `docs/DECISIONS.md` 追加一条新 ADR（D-013 起），写明日期、原做法、新做法、
     依据（含对比数字）和 revisit 条件；
   - 不得违背以下不可妥协的原则：时间切分（D-003）、流行度基线（D-004）、
     测试周只碰一次、不做 customer 下采样（D-006）。
5. **唯一允许中途打断用户的情况。** 只有 Claude 无法自行解决的外部阻塞：缺少凭证/数据文件、
   磁盘空间不足、需要付费资源，或某项改动必须推翻 D-001–D-007 中的核心前提。
   其余一律自行处理，并在最终报告中说明。
6. **交付时告诉用户什么。** MVP 完成后，给出一份简洁汇报（同时写入 `reports/MVP_REPORT.md`）：
   - 如何一条命令复现（`make mvp` 或等价命令）及总耗时、峰值内存；
   - Track A：各基线与 ranker 的验证周 / 测试周 MAP@12，分段（老客/新客/冷启动商品）结果，
     各 retrieval channel 及合并后的 Recall@K；
   - Track B：**相对 popularity baseline 的 relative lift 放在第一位**，然后是 Recall@K、NDCG@K、
     tail-item recall、各过滤阶段（support / NPMI）的存活对数、catalog coverage / novelty；
   - 过程中遇到的主要问题与修复方式、新增的 ADR（D-013…）、与文档的偏差；
   - 已知局限和建议用户重点检验的地方。

### MVP 定义

MVP = 每个里程碑的最小可用版本，全部串联、可一条命令复现：

| 里程碑 | MVP 最低要求 |
| --- | --- |
| M0 | CSV → Parquet → DuckDB；`DATA.md` 中的完整性检查全部通过并写成测试；训练/验证/测试周切分，并有测试证明无泄漏 |
| M1 | 流行度 + 复购基线，验证周 MAP@12，含冷启动分段 |
| M2 | 至少：复购、流行度（全局 + 分群）、item-to-item 共购、同款异色；逐 retrieval channel 和合并后的 Recall@K 以及候选集大小 |
| M3 | LightGBM LambdaRank，特征覆盖 DESIGN §4 的四组；验证周选模型，最后在测试周只评估一次 |
| M4 | Outfit completion（跨 slot 的全品类，jewellery 作为单独报告的 segment，D-002）。Market basket analysis：support + NPMI 过滤并报告各阶段存活数；two-tower 模型（MVP 可先只用 metadata，不用 CLIP 图片）；in-batch negatives + logQ correction + popularity-based negative sampling + hard negatives；报告 relative lift、Recall@K、NDCG@K |
| M5 | LightGBM 的 SHAP 解释 → 可读理由；至少一项 ablation study（推荐：Track B negative sampling 方式，或 lift/PMI 过滤开/关） |
| M6 | FastAPI + SQLite 服务，提供 Track A 与 Track B 推荐接口（含理由）；Web UI 按 DESIGN §7.2：首页（For you / Buy it again / Trending）、商品页（Complete the Look / Other colours）、reason chips、demo customer picker、DS view；不含 LLM |
| 额外 | 用全部数据重新训练后生成 Kaggle 格式的 `submission.csv`，供用户提交 late submission 获取真实 private LB 分数 |

图片已下载到 `data/raw/images/`，但 CLIP、visual search（M7a）和 LLM 助手（M7b）都在 MVP 之后，不在 MVP 范围内（D-009–D-012）。M7a、M7b 都是必做；M8（Polyvore）可选。M7b 需要用户提供 Anthropic API key。

---

## 像业界团队一样工作

用户希望通过这个项目贴近业界、为求职积累经验。因此：

- **术语与方法对齐业界（D-008）。** 只使用 `docs/GLOSSARY.md` 中的术语；需要新术语时先加进
  GLOSSARY 并注明受众（DS / PM）。优先选择业界团队会首先采用的方法（公开发表、被广泛部署的），
  不自创方法；如确需自创，标注 *(project-specific)* 并写 ADR 说明理由。
- **向用户解释时**：首次出现的术语给出英文原词 + 一句话解释 + 业界谁在用（DS 还是 PM），
  并在合适处说明面试或工作中会怎么被问到。
- **Git 流程。** 每个里程碑一个分支（`m0-ingestion`、`m1-baselines`…），小步提交，
  commit message 用 Conventional Commits（`feat:`、`fix:`、`test:`、`docs:`）。
  里程碑完成后以 `--no-ff` 合并回 `main`，合并提交的正文写成 PR description
  （Summary / Changes / Results / How to test）。
- **Experiment tracking。** 每次训练/评估用 MLflow（本地 `mlruns/`，已 gitignore）记录参数、指标和产物；
  报告中的数字必须能追溯到某个 run。
- **配置。** 超参与采样参数放在 `configs/*.yaml`，不写死在代码里。
- **交付物对齐业界形式。** MVP 报告按 offline evaluation readout 的结构写（结论先行 → 指标表 →
  segment analysis → ablation → 风险与下一步）；两个模型各写一份简短 model card（`reports/`）。
  明确说明所有结果都是 offline，真正的上线决策需要 online A/B test。

## 合理性护栏（每一步自检）

- **MAP@12 量级。** 第 1 名 ≈ 0.038，银牌 ≈ 0.030。流行度基线通常在 0.02 左右。
  如果看到 > 0.06，**默认是泄漏**，先查泄漏再做别的。
- **Recall@K 是上限。** Ranker 的 MAP@12 不可能超过候选集能覆盖的内容；先调候选召回，再调 ranker。
- **Track B relative lift ≈ 0% 意味着模型只学到了 popularity**，这是一个要如实报告的结果，不要绕过去。
- **lift/NPMI 过滤后如果几乎没剩下什么**，这也是结论（D-005），如实报告。
- 所有特征构造函数都**必须**把 cutoff 日期作为必填参数（无默认值），并有测试断言不读取 cutoff 之后的数据。

## 环境与工程约定

- 机器：Apple Silicon，16 GB 内存。系统 Python 是 3.9，**请使用 Python ≥ 3.11 的 venv**
  （例如 `brew install python@3.11` 或 `uv venv -p 3.11`）。
- 16 GB 内存意味着：大表聚合一律走 DuckDB（设置 `memory_limit` 和 `temp_directory`），
  不要把 3,100 万行整表读进 pandas；原始 CSV 先转成 Parquet（`article_id` 保留为带前导 0 的 10 位字符串，
  或统一存为整数并在输出时补齐；`customer_id` 可哈希成 int64 节省内存）。
- 包采用 `src/` 布局：需要添加 `pyproject.toml` 并 `pip install -e .`，否则 `python -m ensemble...`
  无法导入（当前 Makefile 的 `ingest` 目标依赖这一点）。
- 采样参数（`WINDOW_WEEKS` 等）放在配置里，不要写成常量。
- 所有中间产物写到 `data/interim/` 或 `data/processed/`（已 gitignore），报告写到 `reports/`。
- 常用命令：`make setup`、`make data`、`make ingest`、`make test`；MVP 完成时补上 `make mvp`。
- 每完成一个里程碑，运行 `make test` 全绿后再进入下一个；可以按里程碑提交 commit。
