#!/usr/bin/env python3
"""
generate_queries.py

基于 generate.py 生成的每个 Skew 专属元数据，生成与数据分布严格对齐的查询负载。

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# 与 train_rl_with_tpch.py 的运行方式保持一致。
sys.path.insert(0, "/home/lkq/project")

from query_planner.tpch_workload_generator import (  # noqa: E402
    TPC_H_Query,
    TPC_H_WorkloadGenerator,
    WorkloadAdapter,
)


# -----------------------------------------------------------------------------
# 配置。必须与 generate.py 中的 SKEW_CONFIGS 保持一致。
# -----------------------------------------------------------------------------

DATA_BASE = Path("/home/lkq/tpch-data")
QUERY_DIR = Path("/home/lkq/project/experiment/skew/queries")
QUERY_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_MAP: Dict[str, float] = {
    "Skew0": 1.0,
    "Skew-1": 2.0,
    "Skew1": 1.5,
    "Skew2": 2.0,
    "Skew3": 2.5,
    "Skew4": 3.0,
}

# 不使用 Python 内置 hash 生成 seed，因为 hash 会受 PYTHONHASHSEED 影响。
SEED_BY_SKEW: Dict[str, int] = {
    "Skew0": 4200,
    "Skew-1": 4199,
    "Skew1": 4201,
    "Skew2": 4202,
    "Skew3": 4203,
    "Skew4": 4204,
}

TEMPLATE_MIX: Dict[str, float] = {
    "Q3": 0.40,
    "Q5": 0.30,
    "Q10": 0.20,
    "Q16": 0.05,
    "Q17": 0.05,
}

NUM_QUERIES = 1000


# -----------------------------------------------------------------------------
# JSON 序列化工具
# -----------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """将 numpy / pandas 标量递归转换为 JSON 友好类型。"""
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return [_to_jsonable(v) for v in value.tolist()]
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return value


def serialize_query(query: TPC_H_Query) -> Dict[str, Any]:
    """将 TPC_H_Query 序列化为 JSON 字典。"""
    return {
        "query_id": str(query.query_id),
        "timestamp": float(query.timestamp),
        "template_id": str(query.template_id),
        "join_key": int(query.join_key),
        "order_status": str(query.order_status),
        "actual_arrival_time": float(query.actual_arrival_time),
        "filter_conditions": _to_jsonable(query.filter_conditions or {}),
    }


def deserialize_query(data: Dict[str, Any]) -> TPC_H_Query:
    """从 JSON 字典恢复 TPC_H_Query。"""
    return TPC_H_Query(
        query_id=str(data["query_id"]),
        timestamp=float(data.get("timestamp", 0.0)),
        template_id=str(data["template_id"]),
        join_key=int(data["join_key"]),
        order_status=str(data.get("order_status", "")),
        actual_arrival_time=float(data.get("actual_arrival_time", 0.0)),
        filter_conditions=data.get("filter_conditions") or {},
    )


# -----------------------------------------------------------------------------
# 元数据加载与校验
# -----------------------------------------------------------------------------


def metadata_path_for_skew(skew_name: str) -> Path:
    return DATA_BASE / f"query_generation_metadata_{skew_name}.json"


def load_metadata_for_skew(skew_name: str) -> Dict[str, Any]:
    """加载某个 Skew 专属元数据。缺失则直接报错，避免静默生成错查询。"""
    path = metadata_path_for_skew(skew_name)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}\n"
            f"请先运行整理后的 generate.py 生成每个 Skew 的 metadata。"
        )

    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required = [
        "valid_orderkeys",
        "valid_orderstatus",
        "orderstatus_by_key",
        "returnflags_by_key",
        "quantities_by_key",
    ]
    missing = [field for field in required if field not in metadata]
    if missing:
        raise KeyError(f"{path} 缺少字段: {missing}")

    if not metadata["valid_orderkeys"]:
        raise ValueError(f"{path} 中 valid_orderkeys 为空，不能生成对齐查询。")

    return metadata


# -----------------------------------------------------------------------------
# 查询生成与加载
# -----------------------------------------------------------------------------


def generate_queries_for_skew(
    skew_name: str,
    num_queries: int = NUM_QUERIES,
    save: bool = True,
) -> List[TPC_H_Query]:
    """为单个 Skew 生成查询负载。"""
    if skew_name not in ALPHA_MAP:
        raise ValueError(f"未知 skew_name={skew_name}，可选值: {list(ALPHA_MAP)}")

    alpha = float(ALPHA_MAP[skew_name])
    seed = int(SEED_BY_SKEW.get(skew_name, 42))
    metadata = load_metadata_for_skew(skew_name)

    print("\n" + "=" * 72)
    print(f"[{skew_name}] 生成查询 | alpha={alpha:.2f} | seed={seed} | num={num_queries}")
    print("=" * 72)
    print(
        f"  metadata: keys={len(metadata['valid_orderkeys']):,}, "
        f"lineitem_rows={metadata.get('total_lineitem_rows', 'N/A')}, "
        f"orders_rows={metadata.get('total_orders_rows', 'N/A')}"
    )
    print(f"  valid_orderstatus={metadata.get('valid_orderstatus')}")

    generator = TPC_H_WorkloadGenerator(
        seed=seed,
        zipfian_skew=alpha,
        scale_factor=1,
        dynamic_shift=False,
        valid_orderkeys=metadata.get("valid_orderkeys"),
        valid_orderstatus=metadata.get("valid_orderstatus"),
        orderstatus_by_key=metadata.get("orderstatus_by_key"),
        returnflags_by_key=metadata.get("returnflags_by_key"),
        quantities_by_key=metadata.get("quantities_by_key"),
        quantities_by_key_returnflag=metadata.get("quantities_by_key_returnflag"),
    )

    queries = generator.generate_batch(num_queries=num_queries, template_mix=TEMPLATE_MIX)
    stats = generator.get_workload_statistics(queries)
    template_stats = Counter(q.template_id for q in queries)

    print("  生成完成：")
    for template, count in sorted(template_stats.items()):
        print(f"    {template}: {count} ({count / num_queries:.1%})")
    print(
        f"    unique_join_keys={stats['unique_keys']:,}, "
        f"zipfian_ratio={stats['zipfian_ratio']:.2%}, "
        f"top_hot_key={stats['top_hot_key'][0]} ({stats['top_hot_key'][1]} 次)"
    )
    print(f"    order_status_dist={stats.get('status_distribution', {})}")

    if save:
        out_path = QUERY_DIR / f"{skew_name}_queries.json"
        payload = {
            "skew_name": skew_name,
            "zipf_alpha": alpha,
            "seed": seed,
            "num_queries": int(num_queries),
            "template_mix": TEMPLATE_MIX,
            "metadata_path": str(metadata_path_for_skew(skew_name)),
            "queries": [serialize_query(q) for q in queries],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  已保存: {out_path}")

    return queries


def load_queries_for_skew(skew_name: str) -> List[TPC_H_Query]:
    """供 train_rl_with_tpch.py 加载预生成查询。"""
    path = QUERY_DIR / f"{skew_name}_queries.json"
    if not path.exists():
        raise FileNotFoundError(
            f"查询文件不存在: {path}\n"
            f"请先运行: python3 experiment/skew/generate_queries.py {skew_name}"
        )

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    queries = [deserialize_query(item) for item in payload.get("queries", [])]
    print(
        f"[{skew_name}] 从 {path} 加载 {len(queries)} 条查询 "
        f"(alpha={payload.get('zipf_alpha', 'N/A')})"
    )
    return queries


def batch_convert_to_join_queries(skew_name: str):
    """批量转换为 QuerySplitter/DriverController 使用的 JoinQuery。"""
    tpch_queries = load_queries_for_skew(skew_name)
    join_queries = [WorkloadAdapter.to_join_query(q) for q in tpch_queries]
    print(f"[{skew_name}] 已转换 {len(join_queries)} 条 JoinQuery")
    return join_queries


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成与 Skew 数据集对齐的 TPC-H 查询负载")
    parser.add_argument(
        "skews",
        nargs="*",
        help="要生成的 Skew 名称；不指定则生成全部。例：Skew2 或 Skew0 Skew1 Skew2",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=NUM_QUERIES,
        help=f"每个 Skew 生成的查询数，默认 {NUM_QUERIES}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skews = args.skews if args.skews else list(ALPHA_MAP.keys())

    print("=" * 72)
    print("生成与数据倾斜对齐的 TPC-H 查询负载")
    print("=" * 72)
    print(f"输出目录: {QUERY_DIR}")
    print(f"查询数量/Skew: {args.num_queries}")
    print(f"模板混合: {TEMPLATE_MIX}")
    print(f"Skews: {skews}")

    for skew_name in skews:
        generate_queries_for_skew(skew_name, num_queries=args.num_queries, save=True)

    print("\n" + "=" * 72)
    print("全部完成。生成文件：")
    for skew_name in skews:
        print(f"  - {QUERY_DIR / f'{skew_name}_queries.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
