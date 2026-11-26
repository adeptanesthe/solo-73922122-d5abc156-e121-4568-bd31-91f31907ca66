# 双层复合膜厚度判定 API

FastAPI 服务：接收两层厚度与总厚度的整数闭区间、一棵最多 12 个叶节点的
合格判定树，以及待选测厚仪的编号与正整数步长；列出**每个可能档值只对应
一种判定**的仪表，对不合用的仪表给出**同档异判**的两组厚度见证，并支持
凭现行方案号提交精测档值。

## 业务规则

* 判定树每个分支比较**某一层**厚度是否 `<=` 整数阈值；叶节点判定为
  `pass`（合格）或 `rework`（返工）。叶节点数最多 12。
* 两层厚度为非负整数，总厚度 = 外层 + 内层，且必须落在 `total` 区间内。
* 每台仪表把**外层**厚度向下取整到自身步长的整倍数，得到"档值"。
  同一档值下若存在两种判定，该仪表不合用。
* 判断**不做厚度枚举**：只对叶节点的轴对齐矩形、步长桶与总厚度约束做
  整数区间运算，复杂度 O(叶节点数 × 仪表数)，与量程大小无关
  （测试覆盖了 `10^18` 量程）。
* 状态机：新建方案 → 可更正粗测（新范围有解则生成新方案号并作废旧号；
  无解则保留现状）→ 凭**现行**方案号提交合用仪表与实测档值；档值可行
  才返回合格/返工结论，并结束本轮。精测被拒、方案号已作废或已结束时
  均保留现状。

## 核验单（判定树识别）

质检员只能送一盘标准样片到产线，需要查清设备实际安装了哪一棵判定树。
回抄结果时最多可能抄错一片，因此：

* 凭**现行方案号**与 **1～3 棵编号各异的备选树**开单（方案树自身编号
  保留为 `plan`，备选树不得占用）；
* 每片样片两层厚度为非负整数，外层/内层/总厚度都落在方案粗测区间内，
  且任意两片厚度组合互不重复；
* 方案树与备选树中的**任意两棵**至少在 **3 片**样片上判定不同，
  保证错抄一片仍能唯一识别；
* 样片总数取**最少**；求解**不逐个枚举厚度**（见下节）；
* 无法配齐时返回 422 并列出是哪两棵树的可区分样片不足 3 片。

回抄时凭核验单号提交每片的 `pass`/`rework`：缺片、同片重复、样片编号
不属于本单一律拒收（单据保持 `open`，可重新提交）。与某棵树的预期最多
差一片时返回该树编号及不一致样片；否则报告无法识别。粗测更正后，旧
方案的核验单不能再提交结果（409，并给出现行方案号）。

### 不枚举厚度的求解方法

每棵树把粗测可行域切成若干轴对齐整数矩形（最多 12 个叶）。k 棵树
（k=2..4）的矩形网格取笛卡尔积，最多 `12^k <= 20736` 个交集矩形；
每个交集矩形对应一个固定的"判定签名"（k 维 pass/rework 向量）。

* 两棵树在某签名上是否判定不同，是签名的纯组合性质，与厚度无关；
* 每个签名区域通过**整数区间夹逼**解析取点（可行外层厚度连续、
  固定外层下可行内层厚度连续），循环次数只等于实际取点数（<=18）；
* 最少片数是一个变量数 <=16、约束数 <=6 的小型整数规划，用
  DFS 分支定界 + 迭代加深求全局最优。

复杂度与区间宽度、阈值大小完全无关（测试覆盖 `10^18` 量程与
4 棵 12 叶树的最坏网格）。

## 目录

```
app/
  engine.py        区间算法：树裁剪、档值区间、合用性、见证点
  verification.py  核验引擎：签名网格、解析取点、最少样片整数规划、回抄识别
  schemas.py       pydantic 请求模型（严格整数校验）
  service.py       方案与核验单状态机
  main.py          FastAPI 路由与统一错误格式
tests/
  test_api.py          原有引擎/API 测试（28 项）
  test_verification.py 核验引擎与 API 测试（23 项，含暴力对照）
examples/          演示脚本
Dockerfile
docker-compose.yml
```

## 本地运行

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Docker Compose

```bash
docker compose up --build
# API:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 健康检查 |
| POST | `/plans` | 创建方案号并评估仪表 |
| POST | `/assess` | 仅评估，不落库 |
| GET | `/plans/{plan_code}` | 查询方案状态与结果 |
| POST | `/plans/{plan_code}/coarse` | 更正粗测范围（换方案号/作废旧号） |
| POST | `/plans/{plan_code}/precision` | 凭现行方案号提交精测档值 |
| GET | `/plans/{plan_code}/readings/{instrument_id}` | 查询某仪表可行档值 |
| POST | `/plans/{plan_code}/verifications` | 凭现行方案号与 1~3 棵备选树开核验单 |
| GET | `/verifications/{verification_code}` | 查询核验单（样片厚度/预期判定/状态） |
| POST | `/verifications/{verification_code}/results` | 回抄样片结果并识别判定树 |

### 创建方案

```bash
curl -s -X POST http://localhost:8000/plans \
  -H 'Content-Type: application/json' \
  -d '{
    "coarse": {"outer": [40, 60], "inner": [0, 10], "total": [40, 70]},
    "tree": {
      "layer": "outer", "threshold": 50,
      "le": {"verdict": "pass"},
      "gt": {"verdict": "rework"}
    },
    "instruments": [
      {"id": 1, "step": 1},
      {"id": 2, "step": 5},
      {"id": 3, "step": 10}
    ]
  }'
```

返回中 `usable_instruments` 为合用仪表；`unusable_instruments` 的各条带
`conflict.samples`，是同一档值下分别判合格、返工的两组厚度。

### 提交精测

```bash
curl -s -X POST http://localhost:8000/plans/<plan_code>/precision \
  -H 'Content-Type: application/json' \
  -d '{"instrument_id": 1, "reading": 40}'
```

成功时 `status` 变为 `completed`，并返回 `verdict`（`pass`/`rework`）
与 `round_ended: true`。档值越界/非步长整倍数/判定不唯一、仪表不合用、
方案号已作废或已结束，都会返回 4xx 且不改动方案状态。

> 注意：判定依赖内层厚度时，只测外层的仪表在任何档值下都可能出现两种判定，
> 因此这类方案没有合用仪表（`solvable: false`）。这是题设条件下的必然结论。

### 更正粗测

```bash
curl -s -X POST http://localhost:8000/plans/<plan_code>/coarse \
  -H 'Content-Type: application/json' \
  -d '{"outer": [45, 60], "inner": [0, 10], "total": [45, 70]}'
```

新范围有解（两层区间与总厚度区间相容）则返回新 `plan_code`（`generation`
加一），旧号 `status` 变 `superseded` 并给出 `superseded_by`；新范围无解或
已被更正过/本轮已结束时返回 4xx，现有方案号保留不变。

### 开核验单

```bash
curl -s -X POST http://localhost:8000/plans/<plan_code>/verifications \
  -H 'Content-Type: application/json' \
  -d '{
    "candidates": [
      {"id": "B", "tree": {
        "layer": "outer", "threshold": 55,
        "le": {"verdict": "pass"}, "gt": {"verdict": "rework"}
      }},
      {"id": "C", "tree": {
        "layer": "inner", "threshold": 5,
        "le": {"verdict": "pass"}, "gt": {"verdict": "rework"}
      }}
    ]
  }'
```

成功返回 201 与核验单号 `vrf-xxxx`、最少样片列表（每片给出
`outer`/`inner`/`total` 与各树的 `expected_verdicts`，方案树键名为
`plan`），以及每对树的实际区分片数 `pair_difference_counts`。
无法配齐时返回 422 `verification_impossible`，`blocking_pairs` 列出
区分片不足 3 片的树对及实际可用片数。

### 回抄样片结果

```bash
curl -s -X POST http://localhost:8000/verifications/<verification_code>/results \
  -H 'Content-Type: application/json' \
  -d '{"results": [
    {"sample_index": 1, "verdict": "rework"},
    {"sample_index": 2, "verdict": "pass"}
  ]}'
```

* 缺片 → 422 `missing_samples`（附 `missing_sample_indexes`）；
* 同片重复 → 422 `duplicate_sample`；编号不属于本单 → 422 `unknown_sample`；
* 拒收不改单据状态，修正后可重新提交；
* 识别成功：`{"recognized": true, "tree_key": "B",
  "mismatches": [{"sample_index": 1, "reported": "rework",
  "expected": "pass", ...}]}`；
* 没有任何树落入 1 片容差：`{"recognized": false, "tree_key": null}`；
* 单据已回抄过 → 409 `verification_submitted`；
* 粗测已更正 → 409 `plan_superseded`，并返回 `current_plan_code`。

## 运行测试

```bash
python3 -m pytest -q
```

## 已知限制

* 方案与核验单状态存于进程内存：多副本/重启会丢失，生产部署需替换为
  共享存储（`PlanService` 的数据结构已按可替换存储设计）。
* 单进程锁只保证同进程内的一致性。

## 本次交付内容与自测

### 实现内容

* 新增 `app/verification.py`：判定签名网格（叶矩形笛卡尔积求交）、
  不枚举厚度的区间解析取点器、最少样片整数规划（DFS 分支定界 +
  迭代加深）、回抄识别（1 片容差，返回不一致样片）。
* `app/schemas.py`：新增开单/回抄请求模型，严格校验备选树数量
  （1~3）、编号互异且不得使用保留名 `plan`、样片结果枚举值。
* `app/service.py`：新增核验单状态机（`open`/`submitted`），与方案
  状态机联动——方案作废后其名下核验单一律拒绝回传。
* `app/main.py`：新增三个路由
  `POST /plans/{plan_code}/verifications`、
  `GET /verifications/{code}`、
  `POST /verifications/{code}/results`，统一错误格式。
* `tests/test_verification.py`：23 项测试，含取点器与暴力枚举对照、
  k=2/3/4 树最少片数与独立暴力多重集枚举对照、`10^18` 量程与
  4 棵 12 叶树最坏网格的毫秒级性能、缺片/重复/作废/无法识别等
  全部 HTTP 分支。
* `examples/demo.py` 与本 README：补充开单、回传的完整示例。

### 自测结果

* `pytest` 全量 **51 项通过**（原有 28 项 + 新增 23 项），约 4 秒。
* 本地 uvicorn 实跑全流程：开单（含最少片数与每对 >=3 区分片校验）、
  缺片/重复片 422 拒收、错一片仍正确识别并列出不一致样片、重复提交
  409、粗测更正后旧单 409 且新方案可正常开单、无法配齐返回阻塞树对。
* 最少性由独立暴力实现（枚举签名多重集，与生产算法不共享代码路径）
  在数百组随机小量程用例上逐例对照通过。

### 未完成项 / 说明

* 交付环境未安装 Docker，`docker compose up --build` 未能在本机实跑；
  Dockerfile 与 compose 配置未改动（新代码均在已 COPY 的 `app/` 目录内，
  镜像构建路径不受影响），仅通过本地 uvicorn 完成等价验证。
