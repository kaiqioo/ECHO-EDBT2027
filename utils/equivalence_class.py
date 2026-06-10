"""
equivalence_class.py - 优化版
适配单cell EC（无下界）的情况
"""
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
import json

@dataclass(frozen=True)
class Cell:
    dimensions: Tuple[Any, ...]
    
    def is_contained_by(self, other: 'Cell') -> bool:
        """判断self是否被other包含（self ⊆ other）"""
        if len(self.dimensions) != len(other.dimensions):
            return False
        
        for self_dim, other_dim in zip(self.dimensions, other.dimensions):
            if other_dim != "*" and self_dim != other_dim:
                return False
        return True
    
    def contains(self, other: 'Cell') -> bool:
        return other.is_contained_by(self)
    
    def to_dict(self):
        return {"dims": list(self.dimensions)}
    
    @staticmethod
    def from_dict(d):
        return Cell(tuple(d["dims"]))


@dataclass
class MatchResult:
    """
    详细的匹配结果，包含 fallback 标记和代价估算（对应文档问题2）
    """
    matched: bool
    ec_id: Optional[str] = None
    aggregate_value: Optional[float] = None
    aggregate_type: str = "SUM"
    raw_count: Optional[int] = None
    
    # [新增] 代价估算字段（用于 cost_model 和 RL）
    estimated_scan_rows: Optional[int] = None      # 如果不物化，需要扫描的行数
    estimated_shuffle_bytes: Optional[int] = None  # 如果不物化，需要 Shuffle 的字节数
    execution_time_ms: Optional[float] = None      # 实际执行时间
    
    match_type: str = ""
    fallback_to_scan: bool = False
    reason: str = ""
    
    def to_dict(self):
        return {
            "matched": self.matched,
            "ec_id": self.ec_id,
            "aggregate_value": self.aggregate_value,
            "aggregate_type": self.aggregate_type,
            "raw_count": self.raw_count,
            # [新增] 代价字段
            "estimated_scan_rows": self.estimated_scan_rows,
            "estimated_shuffle_bytes": self.estimated_shuffle_bytes,
            "execution_time_ms": self.execution_time_ms,
            "match_type": self.match_type,
            "fallback_to_scan": self.fallback_to_scan,
            "reason": self.reason
        }


@dataclass
class EquivalenceClass:
    ec_id: str
    upper_bound: Cell
    lower_bounds: List[Cell]  # 可能为空（单cell EC）
    aggregate_value: float
    table_name: str
    partition_id: int
    raw_count: int = 0
    aggregate_type: str = "SUM"
    
    def match(self, query_cell: Cell) -> bool:
        """
        优化匹配算法：
        1. 必须包含上界
        2. 如果有下界，必须被至少一个下界包含；如果无下界，query_cell必须等于上界
        """

        # 条件1：查询单元必须包含上界（$Q_u \subseteq Q$）
        if not query_cell.contains(self.upper_bound):
            return False
        
        # 条件2：查询单元必须被至少一个下界包含（$Q \subseteq Q_l$）
        # 现在单cell EC 的下界就是上界，逻辑统一：包含上界且被上界包含 ⟹ 相等
        return any(lower.contains(query_cell) for lower in self.lower_bounds)

        # # 条件1：包含上界
        # if not query_cell.contains(self.upper_bound):
        #     return False
        
        # # 条件2：检查下界（单cell EC无下界，此时query_cell必须等于上界）
        # if not self.lower_bounds:
        #     # 单cell EC：只有完全匹配上界才算命中
        #     return query_cell == self.upper_bound
        
        # # 多cell EC：被任意下界包含即可（凸集性质）
        # return any(lower.contains(query_cell) for lower in self.lower_bounds)

    def match_with_details(self, query_cell: Cell) -> MatchResult:
        """
        详细的匹配方法，返回 MatchResult 包含完整匹配信息和 fallback 标记
        """
        # 检查是否包含上界
        if not query_cell.contains(self.upper_bound):
            return MatchResult(
                matched=False,
                match_type="miss",
                fallback_to_scan=True,
                reason="upper_bound_not_contained"
            )
        
        # # 单cell EC 情况（无下界）
        # if not self.lower_bounds:
        #     if query_cell == self.upper_bound:
        #         return MatchResult(
        #             matched=True,
        #             ec_id=self.ec_id,
        #             aggregate_value=self.aggregate_value,
        #             aggregate_type=self.aggregate_type,
        #             raw_count=self.raw_count,
        #             match_type="upper_bound",
        #             fallback_to_scan=False
        #         )
        #     else:
        #         return MatchResult(
        #             matched=False,
        #             match_type="miss",
        #             fallback_to_scan=True,
        #             reason="single_cell_mismatch"
        #         )
        
        # # 多cell EC：检查是否被任意下界包含（凸集性质）
        # for lower in self.lower_bounds:
        #     if lower.contains(query_cell):
        #         return MatchResult(
        #             matched=True,
        #             ec_id=self.ec_id,
        #             aggregate_value=self.aggregate_value,
        #             aggregate_type=self.aggregate_type,
        #             raw_count=self.raw_count,
        #             match_type="convex",
        #             fallback_to_scan=False
        #         )

        # 统一使用凸集性质检查：被任意下界包含（单cell EC 时 lowers=[upper]，自动退化为精确匹配）
        for lower in self.lower_bounds:
            if lower.contains(query_cell):
                return MatchResult(
                    matched=True,
                    ec_id=self.ec_id,
                    aggregate_value=self.aggregate_value,
                    aggregate_type=self.aggregate_type,
                    raw_count=self.raw_count,
                    match_type="convex" if len(self.lower_bounds) > 1 or lower != self.upper_bound else "exact",
                    fallback_to_scan=False
                )
        
        # 包含上界但不被任何下界包含，需要回退
        return MatchResult(
            matched=False,
            match_type="miss",
            fallback_to_scan=True,
            reason="not_covered_by_lower_bounds"
        )

    def to_dict(self):
        return {
            "ec_id": self.ec_id,
            "upper_bound": self.upper_bound.to_dict(),
            "lower_bounds": [lb.to_dict() for lb in self.lower_bounds],
            "agg_val": self.aggregate_value,
            "aggregate_type": self.aggregate_type,
            "table": self.table_name,
            "part": self.partition_id,
            "raw_count": self.raw_count
        }
    
    @staticmethod
    def from_dict(d):
        return EquivalenceClass(
            ec_id=d["ec_id"],
            upper_bound=Cell.from_dict(d["upper_bound"]),
            lower_bounds=[Cell.from_dict(lb) for lb in d["lower_bounds"]],
            aggregate_value=d["agg_val"],
            table_name=d["table"],
            partition_id=d["part"],
            raw_count=d.get("raw_count", 0),
            aggregate_type=d.get("aggregate_type", "SUM")
        )