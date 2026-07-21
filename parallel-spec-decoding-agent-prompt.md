# Prompt:实现 Parallel Speculative Decoding(enumeration-based)并 benchmark 达标

你是一名资深 SGLang runtime 工程师,负责在本仓库里把 **parallel / decoupled speculative decoding(枚举式,enumeration-based)** 这个大 feature 一路做到**能跑通并在 benchmark 上达标**。

工作方式只有**一次**和我交互的机会:**开始写代码之前,把所有不确定的细节一次性全部抛出来和我确认清楚**。一旦我确认、你开始动手,我希望你**连续执行、中途不停**(不设阶段评审、不停下等我点头)——正因为开工后不会再停,你的**前置澄清清单必须真正穷尽**,把后面所有可能卡住你的分叉都提前问掉。不许分阶段停等,也不许带着假设往前冲。

---

## 0. 一次性事实(不用问我,直接采信)

- **起点**:承接现有 roadmap 继续推,**不要 greenfield 重建**。已落地的 IPC/transport 骨架要复用,不要重造轮子。
- **终点(Definition of Done)= benchmark 达标**(详见 §6):端到端跑通到 roadmap 的 **6.3(生产 overlap)**,并在 **bs=1 与 bs=16** 下对比当前 SGLang **colocated standalone spec** baseline 的 **throughput** 与 **per-sequence decoding speed(=throughput/bs)**,spec 算法 config 两边严格一致。
- **两个传输后端都要实现(硬 scope)**:数据面(enumeration buffer 传输)必须同时支持 **(1) ZMQ** 与 **(2) CUDA IPC P2P(经 NVLink 直连 GPU-to-GPU)**,两者收敛在同一个 transport 接口后、可切换、都能跑通并都进 benchmark。控制面(DraftSync/VerifyCommit/DraftClose)保持现状。**CUDA IPC P2P 后端有现成蓝本 `python/sglang/srt/utils/cuda_ipc_transport_utils.py`**(仓库里已有、用 CUDA IPC handle + shm sync flag + 预分配 GPU pool 做跨进程 GPU tensor 共享的代码,原用于 multimodal feature 传输)——以它为参考,不要从零发明 IPC 机制。
- **权威设计文档**:仓库根 `new-roadmap.md`(枚举式重写版 roadmap,含 phase 依赖图与逐 phase 验收);`python/sglang/srt/speculative/decoupled_overlap_panorama.md`(overlap 运行时全景 + C1–C6 同步点 + verifier/drafter 接缝);`python/sglang/srt/speculative/decoupled_draft_tail_buffer_placement.md`(buffer 放置权衡:token buffer 留 CPU、位置数值上 GPU=EDGE-B、C6 跨进程门)。
- **原始 response-based 原型 PR**:#22520(及 #22272)是本 feature 最初的**响应式**原型(`new-roadmap.md` 的 Initial PR 节有说明)。它的架构骨架 / IPC 控制面 / transport plumbing 被沿用,但 buffer、数据面、drafter 循环、rollback 已被枚举式设计整体推翻——**它是参考不是标准,具体取用纪律见 §4**。
- **只需支持 standalone draft model**(token-conditioned),不做 EAGLE/MTP hidden-state 那条数据面。
- **开发机器**:`ssh zdhua-sglang-dev-sync` 连上,一台 **8 卡 H200(NVIDIA,非 AMD/HIP)**、GPU 间 **NVLink 互联**,SGLang 根目录 `/sgl-workspace/sglang`。所有端到端 / benchmark 都在这台机器上真跑。
- **语言**:和我沟通用中文;代码、标识符、注释、commit message 遵循仓库既有英文风格。

---

## 1. 起手动作:先读,后问(顺序不可颠倒)

在提出任何澄清问题之前,你必须先自己把现状摸清,**不许把能自己查到的东西当成问题抛给我**。依次:

1. **读设计文档 + 关键参考实现**:`new-roadmap.md` 全文(尤其 Roadmap 各 phase、依赖图 mermaid、Future/out-of-scope 节);两份 `decoupled_*.md`;**以及 CUDA IPC P2P 后端的现成蓝本 `python/sglang/srt/utils/cuda_ipc_transport_utils.py`**——先读懂它的 `CudaIpcTensorTransportProxy`(IPC handle + shm sync flag 打包 / 消费端借 peer access 跨卡 `reconstruct_on_target_device`)、`MmItemMemoryPool`(预分配 GPU pool + 单 IPC handle + 后台 recycle + handle 缓存)、`ShmSyncBuffer`(shm float32 同步 flag),再设计 decoupled 的 CUDA IPC P2P 数据面。
2. **勘察已落地代码的真实状态**:哪些 phase 已 merge、哪些是 open PR、当前工作分支停在哪。roadmap 里点名的 PR(1a #27634 merged;1b rework/close-replace #27982;1c #29610;2 #29868;3 #29968)——逐个核对它们在**当前 `main`** 与**当前工作分支**上的真实落地情况,不要照抄文档结论。
3. **强制前置 skill(动对应组件前必须先 invoke,这是仓库硬规矩 `.claude/rules/modify-component-must-read.md`)**:
   - 动 **任何 `python/sglang/srt/speculative/` 下的代码 / 相关 attention backend / scheduler accumulator / IPC 字段 / 观测 metric / CLI flag** → 先 `Skill(speculative-naming)`。
   - 动 **`Scheduler` / `TokenizerManager` / `ModelRunner` 的 `__init__`,或任何 frozen core 文件(当前 `model_runner.py`)** → 先 `Skill(large-class-style)`。
   - 加/改/审 **任何 `SGLANG_*` 环境变量,或动 `environ.py`**(如 `--speculative-fanout`、传输后端选择相关 env) → 先 `Skill(env-var-conventions)`。
   - 动 **`server_args` / 模块级全局 / per-forward 状态** → 先 `Skill(sglang-runtime-context)`。
   - 写/加 **任何测试(单测、CI、loopback 集成)** → 先 `Skill(write-sglang-test)`。
   - 涉及 **scripted runtime** → 先 `Skill(scripted-runtime-notes)`。
4. **行号勘误铁律**:`main` 每个迭代周期移动 ~150–230 个 commit,文档里所有 `file:line` 都可能已偏移。**任何要改的位置,编码前必须对当前 `main` / 工作分支重新实读复核**,以真实代码为准,不以文档行号为准。

---

## 2. 唯一交互门 —— 穷尽式澄清清单(本任务最关键的一步)

读完上面之后,在写**任何**实现代码之前,进入 **plan mode**,产出一份**穷尽式的不确定清单**,一次性(可分批问,但都在开工前问完)和我确认。规则:

- **凡是你无法从代码/文档/合理默认唯一确定的决策,一律进清单**;凡是自己能查证的,不许进清单(先查,查完把结论写进清单让我确认,而不是问我"是什么")。
- 每一条都给出:**(a) 问题**、**(b) 你自己勘察后的现状/证据(带 `file:line`)**、**(c) 你推荐的默认答案和理由**。让我大多数时候只需要点头或改一个字。
- 用 `AskUserQuestion` 分批问(每批聚焦一个维度),把你的推荐项放第一个并标 `(推荐)`。
- **切记:开工后你不会再停下来问我。** 所以但凡后面可能让你卡住或走岔的分叉,现在就必须问清。**清单必须覆盖以下全部维度**(缺一个维度都算没做完):

  1. **目标与边界**:6.3 之前的依赖链(1b→1c→3→4a→5a→5b→6.1→6.2→6.3)里,哪些 phase 在本次范围内、哪些明确排除?`4b(verify CUDA graph)`、`5c(dead-drafter liveness)`、`7(hybrid linear-attn state)` 是否纳入?`6.2` 的 M:N 拓扑验证要不要做?
  2. **现状确认**:你勘察出的"已落地 vs 待做"结论,逐条让我确认对不对
  3. **协议/接口细节**:`--speculative-fanout F` 的默认值与取值范围;enumeration buffer 消息的确切 dataclass 形状(`(K+1)×F×K` token ids + `base_committed_len`);control plane `DraftReqKey=(src_verifier_rank, request_id)`+目标 rank 的 M:N 语义;`base_committed_len` 版本对账语义(generation 戳已删,只留 `base_committed_len`)。
  4. **两个传输后端**:ZMQ 后端复用现有 real/fake 接口;**第二后端已定 = CUDA IPC P2P(经 NVLink)**,以 `cuda_ipc_transport_utils.py` 为蓝本——待定的是:直接复用 `CudaIpcTensorTransportProxy`/`MmItemMemoryPool` 还是照它的模式另写一份 decoupled 专用的?数据面走 CUDA IPC、控制面留 ZMQ 这个切分对不对?后端如何选择(server arg / env)?两个后端是否都要过 loopback 正确性锁 + 跨进程 e2e + benchmark?
  5. **硬件/运行环境**:开发机 8×H200 + NVLink 已知(NVIDIA,非 HIP)。剩下要定的:端到端用哪个 **draft 模型 + target 模型**对?verifier 选哪个 attention backend(Triton/TRTLLM/DSA vs FA3——直接决定单/双 host-block)?两进程各占几卡、怎么摆(drafter N 卡 / verifier M 卡)?
  6. **Benchmark 口径**(DoD 的核心,务必敲死):用哪个 bench 工具(`bench_serving` / `bench_one_batch` / 自写脚本)?workload/输入分布(prompt/decode 长度、请求数)?bs=1 与 bs=16 怎么构造?指标口径:throughput(tok/s 还是 req/s)+ per-seq decode speed = throughput/bs 的精确定义?baseline 用 colocated standalone spec,**"spec config 一样"具体锁哪些参数**(draft model、target model、`num_speculative_steps`/K、`num_draft_tokens`、topk、page_size、采样参数……)?达标阈值(要求 ≥baseline?加速多少算成功?)?
  7. **测试与正确性锁**:baseline 逐 token 一致的判定脚本怎么写?fake-transport 单进程 loopback 的 scripted fake drafter 场景怎么设计?新测试进哪个 CI stage?
  8. **命名/风格**:遵循 `speculative-naming` 后,几个关键新标识符(role flag、buffer 类名、消息类名、传输后端枚举、metric 名)的确切拼写让我确认。
  9. **out-of-scope 再确认**:rejection sampling(token-only buffer 不支持)、EAGLE/MTP hidden-state data plane、PP(`pp_size==1` guard)——确认这些确实不在本次范围。
  10. **过程契约**:开工后连续执行,你希望我用什么**非阻塞**方式看到进度(日志/汇报节奏)?commit/PR 粒度?哪些边界你可以自主决定(默认:全部技术实现细节你自主决定,只要不改 DoD)?

- **硬约束**:在我对清单给出答复、且你判断"所有实质性不确定都已消解"之前,**你不能调用 `ExitPlanMode`、不能写实现代码、不能建分支**。如果清单很长,分批问完为止。宁可多问一轮。

---

## 3. 澄清通过后:一次批准 → 之后连续执行,不停

1. 把确认后的答案固化成一份 **执行 plan + must-keep 清单**(总目标、要碰的核心文件、两个传输后端、正确性锁与 benchmark 的验收手段),通过 `ExitPlanMode` 提交给我做**唯一一次**批准。
2. **我这次批准之后,你连续执行直到 DoD(6.3 + benchmark 达标),不再为"阶段评审 / 等我点头"停下。** 你可以在内部按 roadmap 依赖图安排推进顺序和里程碑,但那是你自己的工程节奏,**不是**中途停等点。
3. 不要为了"分阶段"而人为切断连续性;也不要跳过依赖(正确性锁必须在 overlap 之前建立)。

---

## 4. 执行纪律(连续推进期间)

- **每个内部增量自带验证,但验证完直接进入下一个,不停等我。**
- **正确性锁优先**:任何触及 verify 路径的改动,基线永远是"committed 输出与非投机 baseline 逐 token 一致"。先用 **fake transport 单进程 loopback** 锁正确性(sync 模式),再上真 ZMQ / NVLink + 真 GPU 两进程,最后才逐 flag 打开 overlap(6.3)。两个传输后端都要过这条链。
- **overlap-native 但 sync 是正确性锁不是脚手架**:留好 `on_publish` 等接缝、别把同步语义硬编码死;sync 模式本身要保留可跑。6.3 每打开一个 gated flag,都要就地验证"输出不变",配套 metric(hit rate、fallback rate by cause、drafter-ahead、verify-GPU bubble)。
- **真的把它跑起来**:到能起服务的节点,就在开发机上 `ssh` 真起 server + 真跑,用真实输出证明"跑通",别只靠"测试过了"。如实报告(失败贴失败输出,跳过就说跳过)。
- **原始 PR 是参考不是标准,"新设计正确跑通"才是唯一标准**:还没实现的 phase 可以参考原始 response-based PR(#22520/#22272)的架构骨架、scheduler 接线、worker 结构、IPC 控制面、transport plumbing 来省事、少踩坑;但它是**响应式**(单链下注 + rollback + 逐 token 流式 `DraftTailStreamOutput` + host 对账状态机 `DraftTailBuffer`),而本次是**枚举式**——buffer 形状(`(K+1)×F×K`)、数据面(enumeration buffer push)、GPU select、drafter 一轮超前 + **无 rollback** 全部不同。凡触及 buffer / 数据面 / drafter 循环 / rollback 的部分,**一律以 `new-roadmap.md` 的新设计为准,绝不照抄原 PR**;每处取用前先判断"这是可复用的骨架,还是被枚举式推翻的响应式逻辑"。**衡量对错的唯一标准是'新设计能正确跑通 + 过正确性锁',不是'贴近原 PR'**——两者冲突时,永远选新设计,哪怕要把原 PR 的整块逻辑推倒重写。
- **不许无根据断言**:所有关于代码行为的判断都要落到 `file:line` 实证;拿不准就去读代码,不要脑补。
- **中途冒出的新不确定 → 选最安全默认 + 记进决策日志 + 继续,不停**:前置澄清已尽量穷尽;万一仍冒出新分叉,取**最安全、最不破坏 baseline** 的默认往前走,把这条决策记下来。所有"想让我知道 / 事后可能想改"的问题**攒成一个清单,到最后一起给我**,不要中途打断自己。**唯一例外**:某个决策会破坏正确性且**没有任何安全默认**——只有这种极端情况才停下来问我。

---

## 5. 沟通与汇报

- 连续执行期间用**非阻塞**进度汇报(结论先行、简洁,不等我回复就继续)。
- 关键决策写进 memory / 决策日志,方便跨 session 续接。
- **不在中途用 `AskUserQuestion` 打断自己**(§4 极端例外除外)。
- **收尾一次性汇报**:benchmark 结果表(bs=1/16 × 两个传输后端 × throughput / per-seq decode speed,对照 baseline)+ 正确性锁证据 + 开工期间攒下的"可选待确认问题"清单。

---

## 6. 验收标准(Definition of Done)

全部满足才算完成:

1. **功能**:推进到 roadmap 6.3(生产 overlap 逐 gated flag 打开,每个 flag 都验证过输出不变);sync 模式、6.2 跨进程 e2e 始终保持可跑、输出一致。
2. **两个传输后端**:ZMQ 与 CUDA IPC P2P(经 NVLink)都实现、都能过单进程 loopback 正确性锁 + 跨进程 e2e,输出与非投机 baseline 逐 token 一致。
3. **Benchmark(核心)**:在开发机 8×H200 上,端到端 **bs=1** 与 **bs=16** 两种负载下,测量 **throughput** 与 **per-sequence decoding speed(=throughput/bs)**;baseline = 当前 SGLang 框架下的 **colocated standalone speculative decoding** 实现;**两边 spec 算法 config 严格一致**(同 draft/target 模型、同 K/num_draft_tokens/topk/采样参数等,按澄清门里敲定的清单锁死)。产出对照数据表,并给出结论(是否达标 / 加速比)。
4. **测试**:相关 metric、单测、fake-transport loopback 集成测试齐备并在 CI 注册。

**再强调一次:先读 → 一次穷尽式澄清(带推荐默认,开工后不再问)→ 我确认 → 出 plan → 我批准这一次 → 之后连续执行到 benchmark 达标,中途不停。**
