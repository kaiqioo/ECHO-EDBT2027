#!/usr/bin/env python3
"""
tpch_workload_generator.py

"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class TPC_H_Query:
    """简化后的 TPC-H 风格查询描述。"""

    query_id: str
    timestamp: float
    template_id: str
    join_key: int
    order_status: str
    actual_arrival_time: float = 0.0
    filter_conditions: Optional[Dict[str, Any]] = None


class TPC_H_WorkloadGenerator:


    DEFAULT_TEMPLATE_MIX: Dict[str, float] = {
        "Q3": 0.40,
        "Q5": 0.30,
        "Q10": 0.20,
        "Q16": 0.05,
        "Q17": 0.05,
    }

    DEFAULT_ORDER_STATUSES = ["F", "O", "P"]
    DEFAULT_RETURN_FLAGS = ["N", "R", "A"]

    def __init__(
        self,
        seed: int = 42,
        scale_factor: int = 1,
        zipfian_skew: float = 1.5,
        dynamic_shift: bool = False,  # 保留该参数只是为了兼容旧调用；本版本不使用动态漂移。
        valid_orderkeys: Optional[Sequence[int]] = None,
        valid_orderstatus: Optional[Sequence[str]] = None,
        orderstatus_by_key: Optional[Dict[Any, Any]] = None,
        returnflags_by_key: Optional[Dict[Any, Sequence[Any]]] = None,
        quantities_by_key: Optional[Dict[Any, Sequence[Any]]] = None,
        quantities_by_key_returnflag: Optional[Dict[Any, Dict[Any, Sequence[Any]]]] = None,
    ):
        self.seed = seed
        self.scale_factor = scale_factor
        self.zipf_skew = float(zipfian_skew)
        self.query_count = 0

        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # 兼容未传入真实 key 集合的旧实验。倾斜度实验中不建议走这个分支。
        self.max_orderkey = 750000

        self.order_statuses = (
            [str(x) for x in valid_orderstatus]
            if valid_orderstatus
            else list(self.DEFAULT_ORDER_STATUSES)
        )
        self.return_flags = list(self.DEFAULT_RETURN_FLAGS)

        self.valid_orderkeys = self._normalize_orderkeys(valid_orderkeys)
        self.key_weights = self._build_key_weights(self.valid_orderkeys, self.zipf_skew)

        self.orderstatus_by_key = self._normalize_single_value_map(orderstatus_by_key)
        self.returnflags_by_key = self._normalize_list_map(returnflags_by_key, value_type=str)
        self.quantities_by_key = self._normalize_list_map(quantities_by_key, value_type=int)
        self.quantities_by_key_returnflag = self._normalize_nested_quantity_map(
            quantities_by_key_returnflag
        )

        if self.valid_orderkeys is not None:
            print(
                "[WorkloadGenerator] 使用真实 orderkey 集合: "
                f"{len(self.valid_orderkeys)} 个, "
                f"范围 {int(self.valid_orderkeys[0])} ~ {int(self.valid_orderkeys[-1])}, "
                f"Zipf α={self.zipf_skew:.2f}"
            )
        else:
            print(
                "[WorkloadGenerator] 未提供 valid_orderkeys，"
                f"退化为默认范围 1~{self.max_orderkey} 采样。"
            )

        if dynamic_shift:
            print("[WorkloadGenerator] dynamic_shift 参数已忽略：当前实验不使用动态负载漂移。")

    # ------------------------------------------------------------------
    # 元数据标准化
    # ------------------------------------------------------------------

    @staticmethod
    def _key_to_str(key: Any) -> str:
        return str(int(key))

    @classmethod
    def _normalize_orderkeys(cls, keys: Optional[Sequence[int]]) -> Optional[np.ndarray]:
        if keys is None:
            return None

        key_list = list(keys)
        if len(key_list) == 0:
            return None

        unique_keys = sorted({int(k) for k in key_list})
        if not unique_keys:
            return None

        return np.asarray(unique_keys, dtype=np.int64)

    @staticmethod
    def _build_key_weights(keys: Optional[np.ndarray], alpha: float) -> Optional[np.ndarray]:
        if keys is None or len(keys) == 0:
            return None

        if alpha <= 1.0:
            weights = np.ones(len(keys), dtype=np.float64)
        else:
            ranks = np.arange(1, len(keys) + 1, dtype=np.float64)
            weights = 1.0 / np.power(ranks, alpha)

        return weights / weights.sum()

    @classmethod
    def _normalize_single_value_map(cls, mapping: Optional[Dict[Any, Any]]) -> Dict[str, str]:
        if not mapping:
            return {}

        result: Dict[str, str] = {}
        for key, value in mapping.items():
            if isinstance(value, (list, tuple, set)):
                value_list = list(value)
                if not value_list:
                    continue
                value = value_list[0]
            result[cls._key_to_str(key)] = str(value)

        return result

    @classmethod
    def _normalize_list_map(
        cls,
        mapping: Optional[Dict[Any, Sequence[Any]]],
        value_type=str,
    ) -> Dict[str, List[Any]]:
        if not mapping:
            return {}

        result: Dict[str, List[Any]] = {}
        for key, values in mapping.items():
            if values is None:
                continue

            if not isinstance(values, (list, tuple, set, np.ndarray)):
                values = [values]

            cleaned = []
            for value in values:
                try:
                    cleaned.append(value_type(value))
                except Exception:
                    continue

            if cleaned:
                # 去重并排序，保证生成过程可复现。
                result[cls._key_to_str(key)] = sorted(set(cleaned))

        return result

    @classmethod
    def _normalize_nested_quantity_map(
        cls,
        mapping: Optional[Dict[Any, Dict[Any, Sequence[Any]]]],
    ) -> Dict[str, Dict[str, List[int]]]:
        if not mapping:
            return {}

        result: Dict[str, Dict[str, List[int]]] = {}
        for key, flag_map in mapping.items():
            if not isinstance(flag_map, dict):
                continue

            key_str = cls._key_to_str(key)
            result[key_str] = {}

            for flag, quantities in flag_map.items():
                if quantities is None:
                    continue

                if not isinstance(quantities, (list, tuple, set, np.ndarray)):
                    quantities = [quantities]

                cleaned = []
                for q in quantities:
                    try:
                        cleaned.append(int(q))
                    except Exception:
                        continue

                if cleaned:
                    result[key_str][str(flag)] = sorted(set(cleaned))

            if not result[key_str]:
                result.pop(key_str, None)

        return result

    # ------------------------------------------------------------------
    # 基础采样函数
    # ------------------------------------------------------------------

    def _next_query_id(self, template_id: str) -> str:
        self.query_count += 1
        return f"TPC_{template_id}_{self.query_count:06d}"

    def _sample_orderkey(self) -> int:
        """
        从真实 orderkey 集合中采样。

        当 alpha <= 1.0 时使用均匀采样；
        当 alpha > 1.0 时使用 rank-based Zipf 采样。
        这与 generate.py 中基于排序 key 集合的 Zipf 权重方式一致。
        """
        if self.valid_orderkeys is not None and self.key_weights is not None:
            idx = self.np_rng.choice(len(self.valid_orderkeys), p=self.key_weights)
            return int(self.valid_orderkeys[int(idx)])

        if self.zipf_skew <= 1.0:
            return self.py_rng.randint(1, self.max_orderkey)

        zipf_value = int(self.np_rng.zipf(self.zipf_skew))
        return min((zipf_value % self.max_orderkey) + 1, self.max_orderkey)

    def _status_for_key(self, orderkey: int) -> str:
        """优先使用当前 orderkey 在 orders 表中真实存在的 o_orderstatus。"""
        key = self._key_to_str(orderkey)
        if key in self.orderstatus_by_key:
            return self.orderstatus_by_key[key]
        return self.py_rng.choice(self.order_statuses)

    def _returnflag_for_key(self, orderkey: int, preferred: str) -> str:
        """
        优先选择当前 orderkey 下真实存在的 l_returnflag。
        若 preferred 不存在，则从真实 flag 集合中选一个，避免生成空查询。
        """
        key = self._key_to_str(orderkey)
        flags = self.returnflags_by_key.get(key)

        if flags:
            if preferred in flags:
                return preferred
            return self.py_rng.choice(flags)

        return preferred

    def _quantity_for_key(self, orderkey: int, returnflag: Optional[str] = None) -> int:
        """
        为 Q16 选择真实存在的 l_quantity。

        若提供了 quantities_by_key_returnflag，则优先选择当前
        (orderkey, returnflag) 组合下真实存在的 quantity。
        """
        key = self._key_to_str(orderkey)

        if returnflag is not None:
            by_flag = self.quantities_by_key_returnflag.get(key, {})
            quantities = by_flag.get(str(returnflag))
            if quantities:
                return int(self.py_rng.choice(quantities))

        quantities = self.quantities_by_key.get(key)
        if quantities:
            return int(self.py_rng.choice(quantities))

        return self.py_rng.randint(1, 50)

    def _make_query(
        self,
        template_id: str,
        orderkey: int,
        order_status: str,
        filter_conditions: Dict[str, Any],
    ) -> TPC_H_Query:
        """统一构造 TPC_H_Query，保证字段格式一致。"""
        query_id = self._next_query_id(template_id)
        conditions = dict(filter_conditions)
        conditions["template_id"] = template_id

        return TPC_H_Query(
            query_id=query_id,
            timestamp=float(self.query_count),
            template_id=template_id,
            join_key=int(orderkey),
            order_status=str(order_status),
            actual_arrival_time=0.0,
            filter_conditions=conditions,
        )

    # ------------------------------------------------------------------
    # 查询模板
    # ------------------------------------------------------------------

    def generate_q3_style(self) -> TPC_H_Query:
        """Q3 风格：标准订单收入查询。"""
        orderkey = self._sample_orderkey()
        status = self._status_for_key(orderkey)
        returnflag = self._returnflag_for_key(orderkey, preferred="N")

        return self._make_query(
            template_id="Q3",
            orderkey=orderkey,
            order_status=status,
            filter_conditions={
                "l_returnflag": returnflag,
                "o_orderstatus": status,
                "query_pattern": "standard",
            },
        )

    def generate_q5_style(self) -> TPC_H_Query:
        """Q5 风格：状态相关的连接聚合查询。"""
        orderkey = self._sample_orderkey()
        status = self._status_for_key(orderkey)
        returnflag = self._returnflag_for_key(orderkey, preferred="N")

        return self._make_query(
            template_id="Q5",
            orderkey=orderkey,
            order_status=status,
            filter_conditions={
                "l_returnflag": returnflag,
                "o_orderstatus": status,
                "query_pattern": "status_based",
            },
        )

    def generate_q10_style(self) -> TPC_H_Query:
        """Q10 风格：退货分析查询，优先选择 returnflag='R'。"""
        orderkey = self._sample_orderkey()
        status = self._status_for_key(orderkey)
        returnflag = self._returnflag_for_key(orderkey, preferred="R")

        return self._make_query(
            template_id="Q10",
            orderkey=orderkey,
            order_status=status,
            filter_conditions={
                "l_returnflag": returnflag,
                "o_orderstatus": status,
                "query_pattern": "return_analysis",
            },
        )

    def generate_q16_style(self) -> TPC_H_Query:
        """Q16 风格：带数量过滤的连接聚合查询。"""
        orderkey = self._sample_orderkey()
        status = self._status_for_key(orderkey)
        returnflag = self._returnflag_for_key(orderkey, preferred="N")
        quantity = self._quantity_for_key(orderkey, returnflag=returnflag)

        return self._make_query(
            template_id="Q16",
            orderkey=orderkey,
            order_status=status,
            filter_conditions={
                "l_returnflag": returnflag,
                "o_orderstatus": status,
                "l_quantity": quantity,
                "selectivity": 0.3,
                "query_pattern": "quantity_filter",
            },
        )

    def generate_q17_style(self) -> TPC_H_Query:
        """Q17 风格：小粒度订单查询。"""
        orderkey = self._sample_orderkey()
        status = self._status_for_key(orderkey)
        returnflag = self._returnflag_for_key(orderkey, preferred="N")

        return self._make_query(
            template_id="Q17",
            orderkey=orderkey,
            order_status=status,
            filter_conditions={
                "l_returnflag": returnflag,
                "o_orderstatus": status,
                "query_pattern": "small_quantity",
            },
        )

    # ------------------------------------------------------------------
    # 批量生成与统计
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        num_queries: int = 1000,
        template_mix: Optional[Dict[str, float]] = None,
    ) -> List[TPC_H_Query]:
        """按模板比例批量生成查询。"""
        if template_mix is None:
            template_mix = dict(self.DEFAULT_TEMPLATE_MIX)

        templates = list(template_mix.keys())
        weights = list(template_mix.values())

        dispatch = {
            "Q3": self.generate_q3_style,
            "Q5": self.generate_q5_style,
            "Q10": self.generate_q10_style,
            "Q16": self.generate_q16_style,
            "Q17": self.generate_q17_style,
        }

        queries: List[TPC_H_Query] = []
        for _ in range(num_queries):
            template = self.py_rng.choices(templates, weights=weights, k=1)[0]
            generator = dispatch.get(template, self.generate_q3_style)
            queries.append(generator())

        return queries

    def get_workload_statistics(self, queries: List[TPC_H_Query]) -> Dict[str, Any]:
        """返回负载统计信息，用于检查查询热点是否符合预期。"""
        key_freq: Dict[int, int] = {}
        status_dist: Dict[str, int] = {}
        template_dist: Dict[str, int] = {}

        for query in queries:
            key_freq[query.join_key] = key_freq.get(query.join_key, 0) + 1
            status_dist[query.order_status] = status_dist.get(query.order_status, 0) + 1
            template_dist[query.template_id] = template_dist.get(query.template_id, 0) + 1

        sorted_freq = sorted(key_freq.values(), reverse=True)
        if sorted_freq:
            top_20_percent = max(1, int(len(sorted_freq) * 0.2))
            top_20_count = sum(sorted_freq[:top_20_percent])
        else:
            top_20_count = 0

        total = len(queries)

        return {
            "total_queries": total,
            "unique_keys": len(key_freq),
            "zipfian_ratio": top_20_count / total if total > 0 else 0.0,
            "status_distribution": status_dist,
            "template_distribution": template_dist,
            "top_hot_key": max(key_freq.items(), key=lambda item: item[1]) if key_freq else (0, 0),
        }


class WorkloadAdapter:
    """将 TPC_H_Query 转换为 QuerySplitter 使用的 JoinQuery。"""

    ALLOWED_FIELDS = {
        "l_orderkey",
        "l_returnflag",
        "l_quantity",
        "l_extendedprice",
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "template_id",
        "query_pattern",
        "query_type",
        "selectivity",
    }

    QUERY_TYPE_BY_TEMPLATE = {
        "Q3": "standard",
        "Q5": "status_based",
        "Q10": "return_analysis",
        "Q16": "quantity_filter",
        "Q17": "small_quantity",
    }

    @staticmethod
    def to_join_query(tpch_query: TPC_H_Query) -> "JoinQuery":
        from query_planner.query_splitter import JoinQuery

        raw_filters = tpch_query.filter_conditions or {}
        filter_conditions = {
            key: value
            for key, value in raw_filters.items()
            if key in WorkloadAdapter.ALLOWED_FIELDS
        }

        filter_conditions["tables"] = ["lineitem", "orders"]
        filter_conditions["join_keys"] = ["l_orderkey", "o_orderkey"]
        filter_conditions["query_type"] = WorkloadAdapter.QUERY_TYPE_BY_TEMPLATE.get(
            tpch_query.template_id,
            "standard",
        )

        return JoinQuery(
            query_id=tpch_query.query_id,
            join_key=int(tpch_query.join_key),
            agg_func="SUM",
            filter_conditions=filter_conditions,
        )
