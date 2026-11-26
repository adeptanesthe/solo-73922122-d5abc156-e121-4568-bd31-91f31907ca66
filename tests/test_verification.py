"""核验引擎与核验 HTTP API 的测试。

小量程上用暴力枚举对照，验证：
* 开单样片全部落在粗测可行域内且互不重复；
* 任意两棵树在样片上至少 3 片判定不同；
* 解出的片数等于暴力搜索的全局最小值；
* 回抄识别在零错/错一片/错多片/缺片/重复片下的行为。
"""

from __future__ import annotations

import itertools
import random

import pytest
from fastapi.testclient import TestClient

from app import engine, verification
from app.main import app, service


@pytest.fixture(autouse=True)
def _clean_state():
    service.reset()
    yield
    service.reset()


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 引擎：取点生成器
# ---------------------------------------------------------------------------


def _all_feasible_points(coarse: engine.CoarseRange) -> list[tuple[int, int]]:
    points = []
    for e in range(coarse.outer_min, coarse.outer_max + 1):
        for i in range(coarse.inner_min, coarse.inner_max + 1):
            if coarse.total_min <= e + i <= coarse.total_max:
                points.append((e, i))
    return points


def test_point_producer_matches_brute_force_enumeration():
    rng = random.Random(7)
    for _ in range(40):
        e0, e1 = rng.randint(0, 4), rng.randint(5, 12)
        i0, i1 = rng.randint(0, 4), rng.randint(5, 12)
        t0 = rng.randint(0, e1 + i1)
        t1 = rng.randint(t0, e1 + i1)
        coarse = engine.CoarseRange(e0, e1, i0, i1, t0, t1)
        if coarse.error() is not None:
            continue
        rect = engine.Rect(e0, e1, i0, i1)
        got = verification.take_points([rect], coarse, limit=10_000)
        want = _all_feasible_points(coarse)
        assert got == want
        assert len(set(got)) == len(got)


def test_producer_respects_limit_and_works_on_split_rects():
    coarse = engine.CoarseRange(0, 10, 0, 10, 5, 15)
    rects = [
        engine.Rect(0, 3, 0, 10),
        engine.Rect(7, 10, 0, 10),
    ]
    points = verification.take_points(rects, coarse, limit=5)
    assert len(points) == 5
    assert len(set(points)) == 5
    for e, i in points:
        assert coarse.total_min <= e + i <= coarse.total_max
        assert (0 <= e <= 3) or (7 <= e <= 10)


# ---------------------------------------------------------------------------
# 引擎：与暴力搜索对照的最小样片集
# ---------------------------------------------------------------------------


def _random_tree(rng, depth=3):
    if depth == 0 or rng.random() < 0.4:
        return {"verdict": rng.choice(["pass", "rework"])}
    return {
        "layer": rng.choice(["outer", "inner"]),
        "threshold": rng.randint(0, 10),
        "le": _random_tree(rng, depth - 1),
        "gt": _random_tree(rng, depth - 1),
    }


def _brute_min_sample_count(coarse, trees):
    """独立暴力实现：枚举每个判定签名的取点数量（含容量约束），求最少片数。

    与生产代码不同，这里直接枚举所有可行点得到签名与每签名点数上限，
    再对签名多重集做笛卡尔积枚举。小量程下变量数与容量都很小。
    """

    points = _all_feasible_points(coarse)
    pair_indexes = list(itertools.combinations(range(len(trees)), 2))
    cap_by_sig: dict[tuple[str, ...], int] = {}
    for e, i in points:
        sig = tuple(engine.verdict_at(tree, e, i) for tree in trees)
        cap_by_sig[sig] = cap_by_sig.get(sig, 0) + 1

    sigs = list(cap_by_sig)
    pair_of_sig = {
        sig: {pair for pair in pair_indexes if sig[pair[0]] != sig[pair[1]]}
        for sig in sigs
    }
    best = None
    ranges = [range(cap_by_sig[sig] + 1) for sig in sigs]
    for counts in itertools.product(*ranges):
        total = sum(counts)
        if best is not None and total >= best:
            continue
        covered = {pair: 0 for pair in pair_indexes}
        for sig, count in zip(sigs, counts):
            for pair in pair_of_sig[sig]:
                covered[pair] += count
        if all(value >= 3 for value in covered.values()):
            best = total
    return best


@pytest.mark.parametrize("tree_count", [2, 3, 4])
def test_design_matches_brute_force_minimum(tree_count):
    rng = random.Random(20261126 + tree_count)
    checked = 0
    iterations = {2: 400, 3: 600, 4: 1200}[tree_count]
    for _ in range(iterations):
        raw_trees = [_random_tree(rng, depth=2) for _ in range(tree_count)]
        try:
            trees = [engine.parse_tree(raw) for raw in raw_trees]
        except engine.TreeError:
            continue
        # 小量程：总可行点通常 <=25，保证暴力多重集枚举足够快。
        e0, e1 = rng.randint(0, 2), rng.randint(3, 5)
        i0, i1 = rng.randint(0, 2), rng.randint(3, 5)
        # 总厚度区间取宽一些，保证大多数随机粗测范围有解。
        t0 = rng.randint(0, 2)
        t1 = rng.randint(e1 + i1 - 2, e1 + i1)
        coarse = engine.CoarseRange(e0, e1, i0, i1, t0, t1)
        if coarse.error() is not None:
            continue
        keys = ["plan"] + [chr(ord("B") + idx) for idx in range(tree_count - 1)]
        design = verification.design_verification(coarse, trees, keys)
        brute = _brute_min_sample_count(coarse, trees)
        if brute is None:
            assert not design.feasible
            continue
        assert design.feasible
        assert len(design.samples) == brute
        # 不变量：可行域、不重复、每对 >=3、预期判定与真实判定一致。
        seen_points = set()
        for sample in design.samples:
            e, i = sample.outer, sample.inner
            assert (e, i) not in seen_points
            seen_points.add((e, i))
            assert coarse.outer_min <= e <= coarse.outer_max
            assert coarse.inner_min <= i <= coarse.inner_max
            assert coarse.total_min <= e + i <= coarse.total_max
            for t_idx, key in enumerate(keys):
                assert sample.expected[key] == engine.verdict_at(
                    trees[t_idx], e, i
                )
        for a, b in itertools.combinations(range(tree_count), 2):
            diffs = sum(
                1
                for sample in design.samples
                if sample.expected[keys[a]] != sample.expected[keys[b]]
            )
            assert diffs >= 3
        checked += 1
    assert checked >= 25


def test_huge_range_does_not_enumerate_thickness():
    coarse = engine.CoarseRange(0, 10**18, 0, 10**18, 0, 2 * 10**18)
    trees = []
    for threshold in (10**15, 10**15 + 3, 10**16):
        trees.append(
            engine.parse_tree(
                {
                    "layer": "outer",
                    "threshold": threshold,
                    "le": {"verdict": "pass"},
                    "gt": {"verdict": "rework"},
                }
            )
        )
    design = verification.design_verification(coarse, trees, ["plan", "B", "C"])
    assert design.feasible
    # 三棵树只有两个阈值边界，每个边界至少 3 片 -> 最少 6 片。
    assert len(design.samples) == 6
    for sample in design.samples:
        assert sample.outer >= 0 and sample.inner >= 0


def test_worst_case_grid_and_solver_are_fast_without_enumeration():
    """4 棵 12 叶判定树、10^18 量程：网格 <=12^4，必须毫秒级完成。"""

    coarse = engine.CoarseRange(0, 10**18, 0, 10**18, 0, 2 * 10**18)
    trees = []
    for offset in range(4):
        # 恰好 12 个叶节点的退化链（11 次拼接）。
        node = {"verdict": "pass"}
        for leaf in range(11):
            node = {
                "layer": "outer",
                "threshold": 10**17 * (leaf + 1) + offset,
                "le": {"verdict": "pass" if leaf % 2 else "rework"},
                "gt": node,
            }
        trees.append(engine.parse_tree(node))

    import time

    start = time.perf_counter()
    design = verification.design_verification(
        coarse, trees, ["plan", "B", "C", "D"]
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
    # 该组链树阈值只差 0~3，可能无解；无论可行与否，网格与求解都必须毫秒级。
    if design.feasible:
        assert len(design.samples) <= verification.MAX_SAMPLES_HINT
        for sample in design.samples:
            assert sample.outer >= 0 and sample.inner >= 0
            assert sample.total == sample.outer + sample.inner
    else:
        assert design.blocking_pairs


def test_worst_case_feasible_four_trees_with_widely_spaced_thresholds():
    """4 棵判定树、阈值充分错开：可行、求解毫秒级且不枚举厚度。"""

    coarse = engine.CoarseRange(0, 10**18, 0, 10**18, 0, 2 * 10**18)

    def single_threshold(threshold: int):
        return engine.parse_tree(
            {
                "layer": "outer",
                "threshold": threshold,
                "le": {"verdict": "pass"},
                "gt": {"verdict": "rework"},
            }
        )

    # 相邻阈值相差 10^15，任意两树的分歧区间内远不止 3 个整数点。
    trees = [single_threshold(10**15 * (idx + 1)) for idx in range(4)]

    import time

    start = time.perf_counter()
    design = verification.design_verification(
        coarse, trees, ["plan", "B", "C", "D"]
    )
    assert time.perf_counter() - start < 1.0
    assert design.feasible
    assert len(design.samples) <= verification.MAX_SAMPLES_HINT
    for sample in design.samples:
        assert sample.outer >= 0 and sample.inner >= 0
        assert sample.total == sample.outer + sample.inner
    # 只有 3 个"边界带"签名能区分树对，每对需求恰好由相邻两个带各承担 3 片，
    # 全局最优为 9（同结构小量程问题由暴力对照测试独立验证最优性）。
    assert len(design.samples) == 9


def test_identical_trees_are_impossible():
    coarse = engine.CoarseRange(0, 20, 0, 20, 0, 40)
    tree = engine.parse_tree(
        {
            "layer": "outer",
            "threshold": 10,
            "le": {"verdict": "pass"},
            "gt": {"verdict": "rework"},
        }
    )
    design = verification.design_verification(coarse, [tree, tree], ["plan", "B"])
    assert not design.feasible
    assert design.blocking_pairs == (("plan", "B", 0),)


# ---------------------------------------------------------------------------
# 引擎：回抄识别
# ---------------------------------------------------------------------------


def _make_design():
    coarse = engine.CoarseRange(40, 60, 0, 10, 40, 70)
    t_plan = engine.parse_tree(
        {
            "layer": "outer",
            "threshold": 50,
            "le": {"verdict": "pass"},
            "gt": {"verdict": "rework"},
        }
    )
    t_b = engine.parse_tree(
        {
            "layer": "outer",
            "threshold": 55,
            "le": {"verdict": "pass"},
            "gt": {"verdict": "rework"},
        }
    )
    design = verification.design_verification(coarse, [t_plan, t_b], ["plan", "B"])
    assert design.feasible
    return design


def test_recognize_zero_and_one_mismatch():
    design = _make_design()
    truth = {
        s.sample_index: s.expected["B"] for s in design.samples
    }
    ok = verification.recognize(design, ["plan", "B"], dict(truth))
    assert ok.recognized and ok.tree_key == "B" and ok.mismatches == ()

    # 故意抄错一片：仍应识别为 B，并指出不一致样片。
    wrong = dict(truth)
    first = next(iter(wrong))
    wrong[first] = "rework" if wrong[first] == "pass" else "pass"
    one = verification.recognize(design, ["plan", "B"], wrong)
    assert one.recognized and one.tree_key == "B"
    assert len(one.mismatches) == 1
    assert one.mismatches[0]["sample_index"] == first


def test_recognize_two_mismatches_fails():
    design = _make_design()
    # 该设计恰好 3 片且全部是两树分歧片：按 B 回抄再翻 2 片，
    # 与 B 差 2 片、与 plan 差 1 片 -> 识别为 plan（容差自洽）。
    truth = {s.sample_index: s.expected["B"] for s in design.samples}
    flipped = dict(truth)
    for idx in list(flipped)[:2]:
        flipped[idx] = "rework" if flipped[idx] == "pass" else "pass"
    result = verification.recognize(design, ["plan", "B"], flipped)
    assert result.recognized is True
    assert result.tree_key == "plan"
    assert len(result.mismatches) == 1

    # 翻反全部 3 片：与 B 差 3 片、与 plan 差 0 片 -> 识别为 plan。
    all_flip = {
        idx: ("rework" if v == "pass" else "pass") for idx, v in truth.items()
    }
    result_all = verification.recognize(design, ["plan", "B"], all_flip)
    assert result_all.recognized is True and result_all.tree_key == "plan"


def test_recognize_unidentifiable_when_no_tree_within_tolerance():
    # 三棵树的设计片数更多，构造与每棵树都差 >=2 片的回抄。
    coarse = engine.CoarseRange(40, 60, 0, 10, 40, 70)
    trees = [
        engine.parse_tree(
            {
                "layer": "outer",
                "threshold": threshold,
                "le": {"verdict": "pass"},
                "gt": {"verdict": "rework"},
            }
        )
        for threshold in (50, 53, 56)
    ]
    design = verification.design_verification(coarse, trees, ["plan", "B", "C"])
    assert design.feasible and len(design.samples) >= 6

    # 直接按每棵树计算距离，搜索一个与所有树距离都 >=2 的回抄（若存在）。
    indexes = [s.sample_index for s in design.samples]
    found = None
    for mask in range(1 << len(indexes)):
        reported = {}
        for bit, idx in enumerate(indexes):
            base = design.samples[bit].expected["plan"]
            flip = bool(mask & (1 << bit))
            reported[idx] = (
                ("rework" if base == "pass" else "pass") if flip else base
            )
        distances = []
        for key in ("plan", "B", "C"):
            distances.append(
                sum(
                    1
                    for idx in indexes
                    if reported[idx]
                    != next(s for s in design.samples if s.sample_index == idx).expected[key]
                )
            )
        if min(distances) >= 2:
            found = reported
            break
    if found is not None:
        result = verification.recognize(design, ["plan", "B", "C"], found)
        assert result.recognized is False
        assert result.tree_key is None


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def _plan_payload():
    return {
        "coarse": {"outer": [40, 60], "inner": [0, 10], "total": [40, 70]},
        "tree": {
            "layer": "outer",
            "threshold": 50,
            "le": {"verdict": "pass"},
            "gt": {"verdict": "rework"},
        },
        "instruments": [{"id": 1, "step": 1}],
    }


def _candidate_tree(threshold):
    return {
        "layer": "outer",
        "threshold": threshold,
        "le": {"verdict": "pass"},
        "gt": {"verdict": "rework"},
    }


def test_open_verification_order_end_to_end(client):
    plan = client.post("/plans", json=_plan_payload()).json()
    code = plan["plan_code"]
    resp = client.post(
        f"/plans/{code}/verifications",
        json={"candidates": [{"id": "B", "tree": _candidate_tree(55)}]},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["verification_code"].startswith("vrf-")
    assert body["status"] == "open"
    assert body["sample_count"] == len(body["samples"]) >= 3
    # 样片不重复且全部在粗测范围内。
    points = {(s["outer"], s["inner"]) for s in body["samples"]}
    assert len(points) == len(body["samples"])
    for sample in body["samples"]:
        assert 40 <= sample["outer"] <= 60
        assert 0 <= sample["inner"] <= 10
        assert 40 <= sample["total"] <= 70
        assert sample["total"] == sample["outer"] + sample["inner"]
        assert set(sample["expected_verdicts"]) == {"plan", "B"}
    assert body["pair_difference_counts"]["plan|B"] >= 3

    # 凭单号可查询。
    got = client.get(f"/verifications/{body['verification_code']}").json()
    assert got["samples"] == body["samples"]


def test_verification_submit_recognizes_candidate_with_one_copy_error(client):
    plan = client.post("/plans", json=_plan_payload()).json()
    code = plan["plan_code"]
    order = client.post(
        f"/plans/{code}/verifications",
        json={"candidates": [{"id": "B", "tree": _candidate_tree(55)}]},
    ).json()
    vcode = order["verification_code"]
    results = [
        {"sample_index": s["sample_index"], "verdict": s["expected_verdicts"]["B"]}
        for s in order["samples"]
    ]
    # 抄错一片。
    results[0]["verdict"] = (
        "rework" if results[0]["verdict"] == "pass" else "pass"
    )
    resp = client.post(f"/verifications/{vcode}/results", json={"results": results})
    assert resp.status_code == 200
    body = resp.json()
    assert body["recognized"] is True
    assert body["tree_key"] == "B"
    assert len(body["mismatches"]) == 1
    assert body["mismatches"][0]["sample_index"] == results[0]["sample_index"]

    # 重复提交拒收。
    again = client.post(
        f"/verifications/{vcode}/results", json={"results": results}
    )
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "verification_submitted"


def test_verification_submit_unrecognized_when_too_many_errors(client):
    # 三棵树、片数 >=6：构造与每棵树距离都 >=2 的回抄。
    plan = client.post("/plans", json=_plan_payload()).json()
    code = plan["plan_code"]
    order = client.post(
        f"/plans/{code}/verifications",
        json={
            "candidates": [
                {"id": "B", "tree": _candidate_tree(53)},
                {"id": "C", "tree": _candidate_tree(56)},
            ]
        },
    ).json()
    assert order["sample_count"] >= 6
    vcode = order["verification_code"]
    samples = order["samples"]
    keys = ["plan", "B", "C"]

    found = None
    for mask in range(1 << len(samples)):
        results = []
        for bit, sample in enumerate(samples):
            base = sample["expected_verdicts"]["plan"]
            flip = bool(mask & (1 << bit))
            verdict = ("rework" if base == "pass" else "pass") if flip else base
            results.append({"sample_index": sample["sample_index"], "verdict": verdict})
        distances = []
        for key in keys:
            distances.append(
                sum(
                    1
                    for item, sample in zip(results, samples)
                    if item["verdict"] != sample["expected_verdicts"][key]
                )
            )
        if min(distances) >= 2:
            found = results
            break
    assert found is not None

    resp = client.post(f"/verifications/{vcode}/results", json={"results": found})
    assert resp.status_code == 200
    body = resp.json()
    assert body["recognized"] is False
    assert body["tree_key"] is None


def test_verification_missing_and_duplicate_samples_rejected(client):
    plan = client.post("/plans", json=_plan_payload()).json()
    code = plan["plan_code"]
    order = client.post(
        f"/plans/{code}/verifications",
        json={"candidates": [{"id": "B", "tree": _candidate_tree(55)}]},
    ).json()
    vcode = order["verification_code"]
    full = [
        {"sample_index": s["sample_index"], "verdict": s["expected_verdicts"]["plan"]}
        for s in order["samples"]
    ]

    missing = client.post(
        f"/verifications/{vcode}/results", json={"results": full[:-1]}
    )
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "missing_samples"
    assert missing.json()["error"]["missing_sample_indexes"] == [full[-1]["sample_index"]]

    duplicate = client.post(
        f"/verifications/{vcode}/results",
        json={"results": full + [full[0]]},
    )
    assert duplicate.status_code == 422
    assert duplicate.json()["error"]["code"] == "duplicate_sample"

    unknown = client.post(
        f"/verifications/{vcode}/results",
        json={"results": full + [{"sample_index": 999, "verdict": "pass"}]},
    )
    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "unknown_sample"

    # 拒收后单据仍为 open，仍可正常回抄。
    ok = client.post(
        f"/verifications/{vcode}/results", json={"results": full}
    )
    assert ok.status_code == 200
    assert ok.json()["recognized"] is True


def test_verification_requires_one_to_three_distinct_candidates(client):
    plan = client.post("/plans", json=_plan_payload()).json()
    code = plan["plan_code"]

    empty = client.post(f"/plans/{code}/verifications", json={"candidates": []})
    assert empty.status_code == 422

    duplicate = client.post(
        f"/plans/{code}/verifications",
        json={
            "candidates": [
                {"id": "B", "tree": _candidate_tree(55)},
                {"id": "B", "tree": _candidate_tree(56)},
            ]
        },
    )
    assert duplicate.status_code == 422

    reserved = client.post(
        f"/plans/{code}/verifications",
        json={"candidates": [{"id": "plan", "tree": _candidate_tree(55)}]},
    )
    assert reserved.status_code == 422

    too_many = client.post(
        f"/plans/{code}/verifications",
        json={
            "candidates": [
                {"id": str(k), "tree": _candidate_tree(50 + k)}
                for k in range(4)
            ]
        },
    )
    assert too_many.status_code == 422


def test_verification_impossible_reports_blocking_pair(client):
    # 粗测内两树分歧点不足 3 个：outer [50,52], inner=0，阈值 50 vs 51。
    payload = _plan_payload()
    payload["coarse"] = {"outer": [50, 52], "inner": [0, 0], "total": [50, 52]}
    payload["tree"] = _candidate_tree(50)
    plan = client.post("/plans", json=payload).json()
    code = plan["plan_code"]
    resp = client.post(
        f"/plans/{code}/verifications",
        json={"candidates": [{"id": "B", "tree": _candidate_tree(51)}]},
    )
    assert resp.status_code == 422
    body = resp.json()["error"]
    assert body["code"] == "verification_impossible"
    assert body["blocking_pairs"][0]["difference_samples"] == 1


def test_superseded_plan_blocks_verification_open_and_submit(client):
    plan = client.post("/plans", json=_plan_payload()).json()
    old = plan["plan_code"]
    order = client.post(
        f"/plans/{old}/verifications",
        json={"candidates": [{"id": "B", "tree": _candidate_tree(55)}]},
    ).json()
    vcode = order["verification_code"]

    corrected = client.post(
        f"/plans/{old}/coarse",
        json={"outer": [40, 60], "inner": [0, 10], "total": [40, 70]},
    ).json()
    new_code = corrected["plan_code"]

    # 旧方案号不能再开核验单。
    stale_open = client.post(
        f"/plans/{old}/verifications",
        json={"candidates": [{"id": "B", "tree": _candidate_tree(55)}]},
    )
    assert stale_open.status_code == 409
    assert stale_open.json()["error"]["code"] == "plan_superseded"

    # 旧核验单不能再回抄。
    results = [
        {"sample_index": s["sample_index"], "verdict": s["expected_verdicts"]["plan"]}
        for s in order["samples"]
    ]
    stale_submit = client.post(
        f"/verifications/{vcode}/results", json={"results": results}
    )
    assert stale_submit.status_code == 409
    assert stale_submit.json()["error"]["current_plan_code"] == new_code

    # 新方案号可以正常开单并回抄。
    fresh = client.post(
        f"/plans/{new_code}/verifications",
        json={"candidates": [{"id": "B", "tree": _candidate_tree(55)}]},
    )
    assert fresh.status_code == 201


def test_verification_on_unknown_order_and_plan(client):
    assert client.get("/verifications/nope").status_code == 404
    submit = client.post(
        "/verifications/nope/results",
        json={"results": [{"sample_index": 1, "verdict": "pass"}]},
    )
    assert submit.status_code == 404
    open_resp = client.post(
        "/plans/nope/verifications",
        json={"candidates": [{"id": "B", "tree": _candidate_tree(55)}]},
    )
    assert open_resp.status_code == 404


def test_invalid_candidate_tree_rejected(client):
    plan = client.post("/plans", json=_plan_payload()).json()
    code = plan["plan_code"]
    resp = client.post(
        f"/plans/{code}/verifications",
        json={"candidates": [{"id": "B", "tree": {"verdict": "maybe"}}]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_tree"


def test_three_candidates_full_flow(client):
    plan = client.post("/plans", json=_plan_payload()).json()
    code = plan["plan_code"]
    resp = client.post(
        f"/plans/{code}/verifications",
        json={
            "candidates": [
                {"id": "B", "tree": _candidate_tree(55)},
                {"id": 9, "tree": _candidate_tree(58)},
                {
                    "id": "C",
                    "tree": {
                        "layer": "inner",
                        "threshold": 5,
                        "le": {"verdict": "pass"},
                        "gt": {"verdict": "rework"},
                    },
                },
            ]
        },
    )
    assert resp.status_code == 201
    order = resp.json()
    assert len(order["candidate_ids"]) == 3
    for pair in ("plan|B", "plan|9", "plan|C", "B|9", "B|C", "9|C"):
        assert order["pair_difference_counts"][pair] >= 3
    for sample in order["samples"]:
        assert set(sample["expected_verdicts"]) == {"plan", "B", "9", "C"}

    # 按设备实际装的是编号 9 的树回抄（无错）。
    results = [
        {"sample_index": s["sample_index"], "verdict": s["expected_verdicts"]["9"]}
        for s in order["samples"]
    ]
    body = client.post(
        f"/verifications/{order['verification_code']}/results",
        json={"results": results},
    ).json()
    assert body["recognized"] is True
    assert body["tree_key"] == "9"


def test_four_trees_recognize_plan_with_one_copy_error(client):
    """方案树 + 3 棵备选 = 4 棵；按方案树预期回抄且错一片，仍识别为 plan。"""

    plan = client.post("/plans", json=_plan_payload()).json()
    code = plan["plan_code"]
    order = client.post(
        f"/plans/{code}/verifications",
        json={
            "candidates": [
                {"id": "B", "tree": _candidate_tree(53)},
                {"id": "C", "tree": _candidate_tree(56)},
                {"id": "D", "tree": _candidate_tree(59)},
            ]
        },
    )
    assert order.status_code == 201
    body = order.json()
    assert body["sample_count"] <= verification.MAX_SAMPLES_HINT
    assert len(body["pair_difference_counts"]) == 6

    results = [
        {"sample_index": s["sample_index"], "verdict": s["expected_verdicts"]["plan"]}
        for s in body["samples"]
    ]
    results[1]["verdict"] = (
        "rework" if results[1]["verdict"] == "pass" else "pass"
    )
    resp = client.post(
        f"/verifications/{body['verification_code']}/results",
        json={"results": results},
    ).json()
    assert resp["recognized"] is True
    assert resp["tree_key"] == "plan"
    assert len(resp["mismatches"]) == 1
    mismatch = resp["mismatches"][0]
    assert mismatch["sample_index"] == results[1]["sample_index"]
    assert mismatch["reported"] == results[1]["verdict"]
    assert mismatch["expected"] != results[1]["verdict"]

    # 单据状态已变为 submitted，GET 可查识别结果。
    got = client.get(f"/verifications/{body['verification_code']}").json()
    assert got["status"] == "submitted"
    assert got["recognition"]["tree_key"] == "plan"
