"""
查询数据单元 (Query Cell) - Driver生成的JSON包
对应您的描述：QueryID, Table, PartitionKey, CellDimensions
"""
from dataclasses import dataclass, asdict
from typing import List, Any, Optional
import json

@dataclass
class QueryCell:
    """
    Driver生成的子查询单元，发送到Executor
    """
    query_id: str           # 所属Join查询的ID (如 "Q3_001")
    table: str              # 查哪张表 ("lineitem" 或 "orders")
    partition_key: Any      # 连接键值 (如 12345 或 "*")
    cell_dimensions: List[Any]  # 数据单元形式 (如 [12345, "*", "*"])
    aggregate_func: str = "SUM"  # 聚合函数
    parent_join_id: str = ""     # 父Join查询ID
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str: str):
        return QueryCell(**json.loads(json_str))
    
    def needs_broadcast(self) -> bool:
        """
        Driver判断：PartitionKey为*则Broadcast，否则Shuffle
        """
        return self.partition_key == "*"