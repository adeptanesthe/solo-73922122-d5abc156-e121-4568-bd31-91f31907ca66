"""FastAPI 请求模型。所有厚度与步长都要求严格的整数（拒绝 bool/float/字符串）。"""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

InstrumentId = Union[StrictInt, str]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RangeModel(StrictModel):
    """两层厚度与总厚度的整数闭区间。"""

    outer: tuple[StrictInt, StrictInt]
    inner: tuple[StrictInt, StrictInt]
    total: tuple[StrictInt, StrictInt]

    @field_validator("outer", "inner", "total")
    @classmethod
    def _non_negative(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] < 0 or value[1] < 0:
            raise ValueError("厚度范围必须是非负整数")
        return value


class InstrumentModel(StrictModel):
    id: InstrumentId
    step: StrictInt = Field(gt=0)

    @field_validator("id")
    @classmethod
    def _non_empty_id(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, str) and value.strip() == "":
            raise ValueError("仪表编号不能为空字符串")
        return value


class TreeModel(StrictModel):
    """判定树用自由 JSON 传递，由引擎做结构校验（叶节点数等）。"""

    tree: dict[str, Any]


class PlanRequest(StrictModel):
    coarse: RangeModel
    tree: dict[str, Any]
    instruments: list[InstrumentModel]

    @field_validator("instruments")
    @classmethod
    def _unique_ids(
        cls, value: list[InstrumentModel]
    ) -> list[InstrumentModel]:
        seen: set[Union[int, str]] = set()
        for item in value:
            if item.id in seen:
                raise ValueError(f"仪表编号重复: {item.id!r}")
            seen.add(item.id)
        return value


class AssessRequest(PlanRequest):
    pass


class PrecisionRequest(StrictModel):
    instrument_id: InstrumentId
    reading: StrictInt

    @field_validator("reading")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("实测档值必须是非负整数")
        return value


class CandidateTreeModel(StrictModel):
    """核验开单中的一棵备选判定树。"""

    id: InstrumentId
    tree: dict[str, Any]

    @field_validator("id")
    @classmethod
    def _non_empty_id(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, str) and value.strip() == "":
            raise ValueError("备选树编号不能为空字符串")
        if str(value) == "plan":
            raise ValueError("'plan' 是方案树的保留编号，备选树不得使用")
        return value


class VerificationOrderRequest(StrictModel):
    """凭现行方案号与 1~3 棵编号各异的备选树开 open 核验单。"""

    candidates: list[CandidateTreeModel] = Field(min_length=1, max_length=3)

    @field_validator("candidates")
    @classmethod
    def _unique_ids(
        cls, value: list[CandidateTreeModel]
    ) -> list[CandidateTreeModel]:
        seen: set[str] = set()
        for item in value:
            key = str(item.id)
            if key in seen:
                raise ValueError(f"备选树编号重复: {item.id!r}")
            seen.add(key)
        return value


class SampleResultModel(StrictModel):
    sample_index: StrictInt = Field(gt=0)
    verdict: str

    @field_validator("verdict")
    @classmethod
    def _known_verdict(cls, value: str) -> str:
        if value not in ("pass", "rework"):
            raise ValueError("样片结果只能是 'pass' 或 'rework'")
        return value


class VerificationSubmitRequest(StrictModel):
    """凭核验单号回抄每片样片的合格/返工结果。"""

    results: list[SampleResultModel] = Field(min_length=1)
