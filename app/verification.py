"""核验样片求解引擎：在多棵判定树之间构造最少且可区分的样片集合。

问题
----
质检只能送一批标准样片到产线，需要据此识别设备实际安装了哪一棵判定树。
已知方案树与 1~3 棵编号各异的备选树（合计 k=2..4 棵），要求：

1. 每片样片 (outer, inner) 为非负整数，且落在方案粗测范围内
   （外层区间、内层区间、总厚度区间同时满足）；
2. 任意两片厚度组合互不重复；
3. 任意两棵树至少在 **3 片**样片上判定不同（回抄时最多错一条仍可识别）；
4. 样片总数取最少；
5. **不得逐个枚举厚度**。

解法（全程整数区间运算，与量程大小无关）
----------------------------------------
每棵树把粗测可行域切成若干轴对齐整数矩形（``engine.collect_cells``）。
k 棵树的矩形网格取笛卡尔积，最多 ``12^k``（k<=4 时 <=20736）个交集矩形；
每个交集矩形内所有点对每棵树的判定都相同，即对应一个"判定签名"
（长度 k 的 pass/rework 向量）。非空且与总厚度约束有交的交集矩形才保留。

* 任意两棵树在某签名上是否判定不同，是签名的纯组合性质，与厚度无关；
* 每个签名区域可解析地取点（区间夹逼，见 :class:`RectPointProducer`），
  同一次开单最多需要 18 个点，因此每个签名只探测 18 个点即可确定容量；
* 求解退化成一个至多 16 个变量、6 个"至少 3 次"约束的小型整数规划，
  用 DFS 分支定界（迭代加深）求全局最优，再按解出的签名数量解析取点。

复杂度 O(12^k + 18 × 12^k + 小规模组合搜索)，与区间宽度、阈值大小完全无关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from . import engine

# 每对树至少需要的区分片数（回抄最多错一条）。
REQUIRED_DIFFERENCES = 3
# 理论上限：C(4,2) 对 × 3，任何可行问题都不会超过该数量。
MAX_SAMPLES_HINT = 18


class VerificationError(ValueError):
    """核验开单参数不合法（树结构错误等）。"""


@dataclass(frozen=True)
class SignatureRegion:
    """一个判定签名及其覆盖的（互不相交的）可行矩形列表。"""

    signature: tuple[engine.Verdict, ...]
    rects: tuple[engine.Rect, ...]


# ---------------------------------------------------------------------------
# 矩形内解析取点：不枚举厚度
# ---------------------------------------------------------------------------


class RectPointProducer:
    """在一组互不相交矩形与总厚度约束的交集中依次取不同整数点。

    对矩形 ``e in [lo_e, hi_e]``、``i in [lo_i, hi_i]`` 且
    ``total_min <= e + i <= total_max``：

    * 可行的外层厚度 e 构成连续区间
      ``[max(lo_e, total_min - hi_i), min(hi_e, total_max - lo_i)]``；
    * 对固定的可行 e，可行内层 i 构成连续区间
      ``[max(lo_i, total_min - e), min(hi_i, total_max - e)]``。

    因此只需按 e 递增、i 递增的顺序解析取下一个点；矩形之间互不相交，
    天然不会重复。循环次数等于实际取点数（<=18），绝不随量程大小循环。
    """

    def __init__(self, rects: Sequence[engine.Rect], coarse: engine.CoarseRange) -> None:
        self._rects = list(rects)
        self._coarse = coarse
        self._idx = 0
        self._e: Optional[int] = None
        self._i: Optional[int] = None
        self._e_hi = 0

    def _advance_rect(self) -> bool:
        while self._idx < len(self._rects):
            rect = self._rects[self._idx]
            if rect.empty():
                self._idx += 1
                continue
            lo_e = max(rect.lo_e, self._coarse.total_min - rect.hi_i)
            hi_e = min(rect.hi_e, self._coarse.total_max - rect.lo_i)
            if lo_e > hi_e:
                self._idx += 1
                continue
            self._e = lo_e
            self._e_hi = hi_e
            lo_i = max(rect.lo_i, self._coarse.total_min - lo_e)
            # hi_e 的解析公式已保证 lo_i <= hi_i，这里防御性再校验一次。
            hi_i = min(rect.hi_i, self._coarse.total_max - lo_e)
            if lo_i > hi_i:  # pragma: no cover - 与区间公式互为防御
                self._idx += 1
                continue
            self._i = lo_i
            self._rect = rect
            return True
        return False

    def next_point(self) -> Optional[tuple[int, int]]:
        if self._e is None and not self._advance_rect():
            return None
        rect = self._rect
        e, i = self._e, self._i
        point = (e, i)

        # 推进到下一个不同的点：先在固定 e 下推进 i。
        hi_i = min(rect.hi_i, self._coarse.total_max - e)
        if i < hi_i:
            self._i = i + 1
            return point
        # 当前 e 的内层已取完，推进 e。可行 e 连续，直接 +1。
        if e < self._e_hi:
            next_e = e + 1
            next_lo_i = max(rect.lo_i, self._coarse.total_min - next_e)
            next_hi_i = min(rect.hi_i, self._coarse.total_max - next_e)
            if next_lo_i > next_hi_i:  # pragma: no cover - 可行性连续
                self._idx += 1
                self._e = None
                if not self._advance_rect():
                    return point
            else:
                self._e = next_e
                self._i = next_lo_i
            return point
        # 矩形取完，切换到下一个矩形。
        self._idx += 1
        self._e = None
        self._advance_rect()
        return point


def take_points(
    rects: Sequence[engine.Rect], coarse: engine.CoarseRange, limit: int
) -> list[tuple[int, int]]:
    """从矩形集合中解析取至多 limit 个互不相同的可行点。"""

    producer = RectPointProducer(rects, coarse)
    points: list[tuple[int, int]] = []
    for _ in range(limit):
        point = producer.next_point()
        if point is None:
            break
        points.append(point)
    return points


def count_capacity(
    rects: Sequence[engine.Rect], coarse: engine.CoarseRange, cap: int
) -> int:
    """返回 min(可行点总数, cap)；只取 cap 个点，不随量程循环。"""

    return len(take_points(rects, coarse, cap))


# ---------------------------------------------------------------------------
# 签名区域构造
# ---------------------------------------------------------------------------


def _intersect_rects(left: engine.Rect, right: engine.Rect) -> engine.Rect:
    return engine.Rect(
        max(left.lo_e, right.lo_e),
        min(left.hi_e, right.hi_e),
        max(left.lo_i, right.lo_i),
        min(left.hi_i, right.hi_i),
    )


def build_signature_regions(
    trees: Sequence[engine.Node], coarse: engine.CoarseRange
) -> list[SignatureRegion]:
    """把 k 棵树的叶矩形网格求交，按判定签名合并非空区域。

    每个树的叶矩形互不相交且共同覆盖整棵树在粗测范围内的可行域，
    因此不同签名区域之间、同签名的不同交集矩形之间都不会有点重叠。
    """

    cells_per_tree = [engine.collect_cells(tree, coarse) for tree in trees]
    if any(not cells for cells in cells_per_tree):
        # 粗测范围有解时理论上不会发生（至少一个叶矩形可行），防御性处理。
        return []

    # [(当前签名, 当前交集矩形)]，逐层与下一棵树的网格求交。
    frontier: list[tuple[tuple[engine.Verdict, ...], engine.Rect]] = [
        ((), engine.Rect(
            coarse.outer_min, coarse.outer_max, coarse.inner_min, coarse.inner_max
        ))
    ]
    for cells in cells_per_tree:
        nxt: list[tuple[tuple[engine.Verdict, ...], engine.Rect]] = []
        for signature, rect in frontier:
            for cell in cells:
                overlap = _intersect_rects(rect, cell.rect)
                if overlap.empty():
                    continue
                nxt.append((signature + (cell.verdict,), overlap))
        frontier = nxt

    grouped: dict[tuple[engine.Verdict, ...], list[engine.Rect]] = {}
    for signature, rect in frontier:
        # 总厚度约束的可行性在取点时统一保证；这里先按几何交集归并。
        grouped.setdefault(signature, []).append(rect)

    regions: list[SignatureRegion] = []
    for signature, rects in grouped.items():
        # 剔除与总厚度约束无交的矩形。
        feasible = [
            rect
            for rect in rects
            if engine.feasible_point(rect, coarse) is not None
        ]
        if feasible:
            regions.append(SignatureRegion(signature=signature, rects=tuple(feasible)))
    return regions


def differing_pairs(
    signature: tuple[engine.Verdict, ...],
) -> tuple[tuple[int, int], ...]:
    """该签名能区分哪些树对（两树判定不同）。"""

    pairs: list[tuple[int, int]] = []
    for a in range(len(signature)):
        for b in range(a + 1, len(signature)):
            if signature[a] != signature[b]:
                pairs.append((a, b))
    return tuple(pairs)


# ---------------------------------------------------------------------------
# 最少样片组合：小型整数规划（DFS 分支定界 + 迭代加深）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignatureOption:
    index: int
    signature: tuple[engine.Verdict, ...]
    pairs: tuple[tuple[int, int], ...]
    capacity: int  # min(区域可行点数, MAX_SAMPLES_HINT)


def _solve_multiset(
    options: list[SignatureOption], pair_count: int
) -> Optional[list[int]]:
    """求每个签名各取几片（x_i <= capacity_i），使每对树被区分 >=3 次，
    并使总片数最少。返回与 options 对齐的取点数量列表，无解返回 None。

    变量数 <= 2^(k-1) <= 16，约束数 C(k,2) <= 6，且每片只服务固定树对，
    DFS 配合下界剪枝即可瞬间完成；迭代加深保证第一个可行解就是最优解。
    """

    usable = [opt for opt in options if opt.pairs]
    pair_index = {
        pair: idx
        for idx, pair in enumerate(
            (a, b)
            for a in range(len(options[0].signature))
            for b in range(a + 1, len(options[0].signature))
        )
    }
    needs = [REQUIRED_DIFFERENCES] * pair_count

    # 先判定结构性无解：某树对没有任何签名能区分。
    coverable = [0] * pair_count
    for opt in usable:
        for pair in opt.pairs:
            coverable[pair_index[pair]] += opt.capacity
    if any(cap < REQUIRED_DIFFERENCES for cap in coverable):
        return None

    # 上界：贪心给每个签名取满，必然可行（coverable 均 >=3）。
    upper = sum(opt.capacity for opt in usable)

    # 能区分更多树对、容量更大的签名优先，便于快速剪枝。
    usable.sort(key=lambda opt: (-len(opt.pairs), -opt.capacity))

    def dfs(pos: int, remaining_need: list[int], budget: int) -> Optional[list[int]]:
        if all(need <= 0 for need in remaining_need):
            return [0] * len(usable)
        if pos == len(usable) or budget <= 0:
            return None
        # 下界 1：剩余需求点数除以"单片最多能满足的树对数"上取整。
        outstanding = sum(max(need, 0) for need in remaining_need)
        max_pairs = max((len(opt.pairs) for opt in usable[pos:]), default=1)
        if (outstanding + max_pairs - 1) // max_pairs > budget:
            return None
        # 下界 2：剩余签名的总容量必须足以覆盖剩余需求（每片至多贡献 1 次/对）。
        residual_capacity = sum(opt.capacity for opt in usable[pos:])
        if residual_capacity * max_pairs < outstanding:
            return None

        opt = usable[pos]
        touched = [pair_index[pair] for pair in opt.pairs]
        # 取超过当前剩余需求总量的片数不可能进入最优解，限制上界。
        need_cap = sum(max(remaining_need[pidx], 0) for pidx in touched)
        max_take = min(opt.capacity, budget, need_cap)
        for take in range(max_take, -1, -1):
            for pidx in touched:
                remaining_need[pidx] -= take
            rest = dfs(pos + 1, remaining_need, budget - take)
            for pidx in touched:
                remaining_need[pidx] += take
            if rest is not None:
                rest[pos] = take
                return rest
        return None

    for budget in range(1, upper + 1):
        result = dfs(0, list(needs), budget)
        if result is not None:
            by_index = [0] * len(options)
            for opt, take in zip(usable, result):
                by_index[opt.index] = take
            return by_index
    return None  # pragma: no cover - coverable 检查已覆盖无解情形


# ---------------------------------------------------------------------------
# 顶层求解
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    sample_index: int
    outer: int
    inner: int
    total: int
    expected: dict[str, engine.Verdict]


@dataclass(frozen=True)
class VerificationPlan:
    samples: tuple[Sample, ...]
    regions: tuple[SignatureRegion, ...]
    pair_difference_counts: dict[tuple[str, str], int]
    feasible: bool
    blocking_pairs: tuple[tuple[str, str, int], ...]


def _tree_keys(plan_tree_key: str, candidate_ids: Sequence[Any]) -> list[str]:
    return [plan_tree_key] + [str(cid) for cid in candidate_ids]


def design_verification(
    coarse: engine.CoarseRange,
    trees: Sequence[engine.Node],
    tree_keys: Sequence[str],
) -> VerificationPlan:
    """根据粗测范围与 k 棵树设计最少核验样片。

    ``tree_keys`` 与 ``trees`` 一一对应，第 0 棵为方案树。
    不枚举任何厚度：签名由矩形交集解析得到，取点由区间夹逼解析得到。
    """

    if len(trees) != len(tree_keys) or not (2 <= len(trees) <= 4):
        raise VerificationError("参与核验的判定树必须是 2~4 棵")

    regions = build_signature_regions(trees, coarse)
    options: list[SignatureOption] = []
    for idx, region in enumerate(regions):
        capacity = count_capacity(region.rects, coarse, MAX_SAMPLES_HINT)
        if capacity <= 0:
            continue
        options.append(
            SignatureOption(
                index=idx,
                signature=region.signature,
                pairs=differing_pairs(region.signature),
                capacity=capacity,
            )
        )

    pair_count = len(trees) * (len(trees) - 1) // 2

    # 预检：逐树对统计可区分点数（同样只取至多 3 个见证点）。
    blocking: list[tuple[str, str, int]] = []
    pair_counts: dict[tuple[str, str], int] = {}
    for a in range(len(trees)):
        for b in range(a + 1, len(trees)):
            available = 0
            for opt in options:
                if (a, b) in opt.pairs:
                    available += min(
                        opt.capacity, REQUIRED_DIFFERENCES - available
                    )
                    if available >= REQUIRED_DIFFERENCES:
                        break
            pair_counts[(tree_keys[a], tree_keys[b])] = available
            if available < REQUIRED_DIFFERENCES:
                blocking.append((tree_keys[a], tree_keys[b], available))

    if blocking:
        return VerificationPlan(
            samples=(),
            regions=tuple(regions),
            pair_difference_counts=pair_counts,
            feasible=False,
            blocking_pairs=tuple(blocking),
        )

    multiset = _solve_multiset(options, pair_count)
    if multiset is None:  # pragma: no cover - 预检通过则必然有解
        return VerificationPlan(
            samples=(),
            regions=tuple(regions),
            pair_difference_counts=pair_counts,
            feasible=False,
            blocking_pairs=tuple(blocking),
        )

    samples: list[Sample] = []
    sample_index = 1
    for opt, take in zip(options, multiset):
        if take <= 0:
            continue
        region = regions[opt.index]
        points = take_points(region.rects, coarse, take)
        if len(points) != take:  # pragma: no cover - 容量已在求解前探测
            raise AssertionError("签名区域容量与求解结果不一致")
        for outer, inner in points:
            expected = {
                key: region.signature[tree_idx]
                for tree_idx, key in enumerate(tree_keys)
            }
            samples.append(
                Sample(
                    sample_index=sample_index,
                    outer=outer,
                    inner=inner,
                    total=outer + inner,
                    expected=expected,
                )
            )
            sample_index += 1

    return VerificationPlan(
        samples=tuple(samples),
        regions=tuple(regions),
        pair_difference_counts=pair_counts,
        feasible=True,
        blocking_pairs=(),
    )


# ---------------------------------------------------------------------------
# 回抄识别
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecognitionResult:
    recognized: bool
    tree_key: Optional[str]
    mismatches: tuple[dict[str, Any], ...]
    checked_sample_count: int


def recognize(
    plan: VerificationPlan,
    tree_keys: Sequence[str],
    reported: dict[int, engine.Verdict],
) -> RecognitionResult:
    """按回抄结果识别设备实际安装的树。

    与某棵树的预期最多差一片时返回该树及不一致样片；否则无法识别。
    开单时的"任意两树 >=3 片不同"保证至多只有一棵树能落入 1 片容差。
    """

    expected_by_index = {
        sample.sample_index: sample.expected for sample in plan.samples
    }
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for key in tree_keys:
        mismatches: list[dict[str, Any]] = []
        for sample_index, verdict in sorted(reported.items()):
            if expected_by_index[sample_index][key] != verdict:
                sample = next(s for s in plan.samples if s.sample_index == sample_index)
                mismatches.append(
                    {
                        "sample_index": sample_index,
                        "outer": sample.outer,
                        "inner": sample.inner,
                        "reported": verdict,
                        "expected": expected_by_index[sample_index][key],
                    }
                )
        if len(mismatches) <= 1:
            candidates.append((key, mismatches))

    if len(candidates) == 1:
        key, mismatches = candidates[0]
        return RecognitionResult(
            recognized=True,
            tree_key=key,
            mismatches=tuple(mismatches),
            checked_sample_count=len(reported),
        )
    return RecognitionResult(
        recognized=False,
        tree_key=None,
        mismatches=(),
        checked_sample_count=len(reported),
    )
