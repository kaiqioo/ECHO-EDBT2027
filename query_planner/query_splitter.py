#!/usr/bin/env python3


import sys
sys.path.insert(0, '/home/lkq/project')
from utils.query_cell import QueryCell
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field



@dataclass
class JoinQuery:
    """Join查询定义"""
    query_id: str
    join_key: int           # orderkey的具体值
    agg_func: str          # "SUM" or "COUNT"
    filter_conditions: Dict[str, Any]  # 如 {"l_returnflag": "N"}
    tables: List[str] = field(default_factory=lambda: ["lineitem", "orders"])
    template_id: str = "Q3"
    aggregation_funcs: List[str] = field(default_factory=lambda: ["sum", "count"])
    group_by_keys: List[str] = field(default_factory=lambda: ["l_orderkey"])

    def __post_init__(self):
        """初始化默认值"""
        if self.tables is None:
            self.tables = ["lineitem", "orders"]
        if self.aggregation_funcs is None:
            self.aggregation_funcs = ["sum", "count"]
        if self.group_by_keys is None:
            self.group_by_keys = ["l_orderkey"]


class QuerySplitter:
    """
    将Join查询拆分为两个子查询（lineitem和orders），生成包含完整维度信息的QueryCell
    """
    
    def __init__(self):
        # 表结构配置（3维）
        self.table_schema = {
            "lineitem": {
                "dims": ["l_orderkey", "l_returnflag", "l_quantity"],
                "bucket_config": {"l_quantity": 10},
                "dim_indices": {"orderkey": 0, "returnflag": 1, "quantity": 2}
            },
            "orders": {
                "dims": ["o_orderkey", "o_custkey", "o_orderstatus"],
                "bucket_config": {"o_orderkey": 100},
                "dim_indices": {"orderkey": 0, "custkey": 1, "orderstatus": 2}
            }
        }

        # 字段白名单（确保实验纯净性）
        self.allowed_columns = {
            "lineitem": ["l_orderkey", "l_returnflag", "l_quantity", "l_extendedprice"],
            "orders": ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice"]
        }

    # 新增字段白名单验证方法
    def _validate_filter_conditions(self, join_query: JoinQuery) -> Tuple[bool, List[str]]:
        """验证过滤条件是否只包含允许的8个字段"""
        invalid_fields = []
        for key in join_query.filter_conditions.keys():
            if key in ['query_type', 'template_id', 'quantity_threshold','query_pattern','date_range', 'start_date','end_date','tables', 'agg_func','join_keys']:
                continue
            
            found = False
            for table, allowed in self.allowed_columns.items():
                if key in allowed:
                    found = True
                    break
            
            if not found:
                invalid_fields.append(key)
        
        if invalid_fields:
            print(f"⚠️  Warning: Query {join_query.query_id} contains invalid fields: {invalid_fields}")
        
        return len(invalid_fields) == 0, invalid_fields


    def _apply_bucket_to_value(self, value, bucket_size):
        """对值进行分桶转换，保持'*'不变"""
        if value == "*" or value is None:
            return "*"
        try:
            return int(value) % bucket_size
        except (ValueError, TypeError):
            return value

    
    def split_join_query(self, join_query: JoinQuery) -> Tuple[QueryCell, QueryCell]:
        """
        将 JoinQuery 拆分为 lineitem 和 orders 两个 QueryCell。
        注意：
        1. partition_key 始终保留原始 orderkey，用于 Executor 路由。
        2. cell_dimensions 中的维度值必须与 Executor 构建格结构时的分桶规则一致。
        """
        key = int(join_query.join_key)
        filters = join_query.filter_conditions or {}

        schema_l = self.table_schema["lineitem"]
        schema_o = self.table_schema["orders"]

        self._validate_filter_conditions(join_query)

        # ========== lineitem 子查询 ==========
        # 维度顺序：
        # [l_orderkey, l_returnflag, l_quantity_bucket]
        returnflag_val = filters.get("l_returnflag", "*")

        if "l_quantity" in filters:
            quantity_val = self._apply_bucket_to_value(
                filters["l_quantity"],
                schema_l["bucket_config"]["l_quantity"]
            )
        else:
            quantity_val = "*"

        cell_l = QueryCell(
            query_id=f"{join_query.query_id}_L",
            table="lineitem",
            partition_key=key,          # 原始 key，用于路由
            cell_dimensions=[
                key,                    # l_orderkey 不分桶
                returnflag_val,
                quantity_val             # l_quantity % 10
            ],
            aggregate_func=join_query.agg_func,
            parent_join_id=join_query.query_id
        )

        # ========== orders 子查询 ==========
        # 维度顺序：
        # [o_orderkey_bucket, o_custkey, o_orderstatus]
        bucketed_orderkey = self._apply_bucket_to_value(
            key,
            schema_o["bucket_config"]["o_orderkey"]
        )

        custkey_val = filters.get("o_custkey", "*")
        orderstatus_val = filters.get("o_orderstatus", "*")

        cell_o = QueryCell(
            query_id=f"{join_query.query_id}_O",
            table="orders",
            partition_key=key,          # 原始 key，用于路由
            cell_dimensions=[
                bucketed_orderkey,       # o_orderkey % 100
                custkey_val,
                orderstatus_val
            ],
            aggregate_func=join_query.agg_func,
            parent_join_id=join_query.query_id
        )

        return cell_l, cell_o

    
    def decide_routing(self, cell: QueryCell, num_executors: int = 2) -> int:
        """
        路由规则必须与 generate.py 中保存 partition 的规则一致：
            partition = int(orderkey) % num_executors

        注意：这里用的是 QueryCell.partition_key，也就是原始 orderkey，
        不是 cell_dimensions 里的 bucketed o_orderkey。
        """
        if cell.needs_broadcast():
            return -1

        return int(cell.partition_key) % num_executors


class QueryExecutionResult:
    """
    查询执行结果（用于超图构建的数据结构）
    [MODIFIED] 新增：代价估算字段、语义特征提取方法
    """
    
    def __init__(self, query_cell: QueryCell, match_result: dict, executor_id: str):
        self.query_cell = query_cell
        self.match_result = match_result
        self.executor_id = executor_id
        
        # 关键字段（用于超图节点/边构建）
        self.query_id = query_cell.parent_join_id
        self.table = query_cell.table
        self.ec_id = match_result.get("ec_id") if match_result.get("matched") else None
        self.aggregate_value = match_result.get("aggregate_value")
        self.matched = match_result.get("matched", False)

        # 代价估算字段（用于RL奖励计算）
        self.raw_count = match_result.get("raw_count", 0)
        self.estimated_scan_rows = match_result.get("estimated_scan_rows", self.raw_count)
        self.estimated_shuffle_bytes = match_result.get("estimated_shuffle_bytes", 0)
        self.execution_time_ms = match_result.get("execution_time_ms", 0)
        self.is_fully_covered = match_result.get("is_fully_covered", self.matched)
        self.fallback_reason = match_result.get("fallback_reason", None)


    # [ADDED] 语义特征提取（关键！）
    def get_predicate_signature(self) -> dict:
        """
        提取谓词特征作为EC语义标识
        例如：{"table": "lineitem", "l_returnflag": "N", "l_quantity": 5}
        """
        dims = self.query_cell.cell_dimensions
        schema_map = {
            "lineitem": ["l_orderkey", "l_returnflag", "l_quantity"],
            "orders": ["o_orderkey", "o_custkey", "o_orderstatus"],

            # JOB-derived binary workload
            "title": ["id", "kind_id", "production_year_bucket"],
            "movie_info_idx": ["movie_id", "info_type_id", "info_bucket"],
            "movie_companies": ["movie_id", "company_id_bucket", "company_type_id"],
            "movie_keyword": ["movie_id", "keyword_id_bucket", "keyword_id"]
        }
        
        signature = {"table": self.table}
        dim_names = schema_map.get(self.table, [])
        
        for i, val in enumerate(dims):
            if i < len(dim_names) and val != "*":
                signature[dim_names[i]] = val
        
        return signature


    def to_hypergraph_node(self) -> dict:
        """
        转换为超图节点表示（供后续RL使用）
        超图节点 = 被命中的等价类（EC）
        """
        if not self.matched or not self.ec_id:
            return None
        
        return {
            "node_type": "equivalence_class",
            "ec_id": self.ec_id,
            "table": self.table,
            "executor_id": self.executor_id,
            "aggregate_value": self.aggregate_value,
            "raw_count": self.raw_count,
            "query_pattern": self.query_cell.cell_dimensions,
            # [ADDED] 语义特征
            "predicate_signature": self.get_predicate_signature(),
            "dimension_bounds": self.match_result.get("ec_bounds", {}),
            # [ADDED] 代价字段
            "estimated_scan_rows": self.estimated_scan_rows,
            "estimated_shuffle_bytes": self.estimated_shuffle_bytes,
            "execution_time_ms": self.execution_time_ms
        }
    
    def to_hypergraph_edge(self) -> dict:
        """
        转换为超图边的一部分（供后续RL使用）
        超图边 = Join查询，连接多个EC节点
        """
        # 从query_id解析模板ID（如 TPC_Q10_000001 -> Q10）
        # 这样即使QueryCell不修改，也能知道模板类型
        parts = self.query_id.split('_')
        template_id = parts[1] if len(parts) >= 2 else 'unknown'
        
        return {
            "query_id": self.query_id,
            "table": self.table,
            "executor_id": self.executor_id,
            "ec_id": self.ec_id,
            "matched": self.matched,
            "template_id": template_id,
            "raw_count": self.raw_count  # [ADDED] 用于超边权重
        }