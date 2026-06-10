#!/usr/bin/env python3
"""
generate.py

为 ECHO 倾斜度敏感性实验生成 200K TPC-H 子数据集。

输出目录：
  /home/lkq/tpch-data/Skew0/partition0|1/{lineitem,orders}.parquet
  /home/lkq/tpch-data/Skew1/partition0|1/{lineitem,orders}.parquet
  ...

同时为每个 Skew 单独保存查询生成元数据：
  /home/lkq/tpch-data/query_generation_metadata_Skew0.json
  /home/lkq/tpch-data/query_generation_metadata_Skew1.json
  ...

关键约束：
1. 数据分区规则固定为 int(orderkey) % NUM_PARTITIONS。
   后续 query_splitter / Driver 路由必须使用同一规则。
2. Skew 数据和查询生成都基于同一组排序后的真实 orderkey 做 rank-based Zipf。
3. 元数据按 Skew 单独保存，不能让所有 Skew 共用同一个 key 集合。


注意：把driver的/home/lkq/tpch-data数据同步到executor
rsync -av --delete /home/lkq/tpch-data/ executor1:/home/lkq/tpch-data/
rsync -av --delete /home/lkq/tpch-data/ executor2:/home/lkq/tpch-data/

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 基础配置
# -----------------------------------------------------------------------------

TPCH_PATH = Path("/home/lkq/tpch-dbgen")
OUTPUT_BASE = Path("/home/lkq/tpch-data")
TARGET_ROWS = 200_000
NUM_PARTITIONS = 2
SEED = 42

# Skew0 是从 SF=1 中无放回随机采样得到的均匀基准。
# Skew-1 保留名称，但使用固定 alpha，避免数据和查询无法复现对齐。
SKEW_CONFIGS: Dict[str, float] = {
    "Skew-1": 2.0,
    "Skew1": 1.5,
    "Skew2": 2.0,
    "Skew3": 2.5,
    "Skew4": 3.0,
}


# -----------------------------------------------------------------------------
# 数据读取与清洗
# -----------------------------------------------------------------------------


def read_tpch_tables(tpch_path: str):
    """
    读取 TPC-H SF=1 的 lineitem.tbl 和 orders.tbl。

    注意：
    TPC-H lineitem.tbl 的字段位置为：
      0: l_orderkey
      4: l_quantity
      5: l_extendedprice
      8: l_returnflag

    不要把 usecols 写成 [0, 8, 4, 5] 后再按自定义列名强行赋值，
    否则容易把 l_returnflag='N/R/A' 读到 l_extendedprice 列。
    """

    print("Loading SF=1 lineitem ...")

    lineitem = pd.read_csv(
        os.path.join(tpch_path, "lineitem.tbl"),
        sep="|",
        header=None,
        usecols=[0, 4, 5, 8]
    )

    lineitem.columns = [
        "l_orderkey",
        "l_quantity",
        "l_extendedprice",
        "l_returnflag"
    ]

    # 统一列顺序，后续代码都按这个顺序使用
    lineitem = lineitem[
        ["l_orderkey", "l_returnflag", "l_quantity", "l_extendedprice"]
    ].copy()

    lineitem["l_orderkey"] = lineitem["l_orderkey"].astype(int)
    lineitem["l_returnflag"] = lineitem["l_returnflag"].astype(str)
    lineitem["l_quantity"] = lineitem["l_quantity"].astype(int)
    lineitem["l_extendedprice"] = lineitem["l_extendedprice"].astype(float)

    print("Loading SF=1 orders ...")

    orders = pd.read_csv(
        os.path.join(tpch_path, "orders.tbl"),
        sep="|",
        header=None,
        usecols=[0, 1, 2, 3]
    )

    orders.columns = [
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice"
    ]

    orders = orders[
        ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice"]
    ].copy()

    orders["o_orderkey"] = orders["o_orderkey"].astype(int)
    orders["o_custkey"] = orders["o_custkey"].astype(int)
    orders["o_orderstatus"] = orders["o_orderstatus"].astype(str)
    orders["o_totalprice"] = orders["o_totalprice"].astype(float)

    return lineitem, orders


# -----------------------------------------------------------------------------
# 分区与元数据保存
# -----------------------------------------------------------------------------


def partition_id_from_orderkey(orderkey: int) -> int:
    """统一分区规则。query_splitter 和 executor 路由必须保持一致。"""
    return int(orderkey) % NUM_PARTITIONS


def save_partitioned(lineitem: pd.DataFrame, orders: pd.DataFrame, out_dir: Path) -> None:
    """按照 orderkey 将 lineitem 和 orders 写入两个分区。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    li = lineitem.copy()
    od = orders.copy()

    li["part"] = li["l_orderkey"].map(partition_id_from_orderkey)
    od["part"] = od["o_orderkey"].map(partition_id_from_orderkey)

    for part in range(NUM_PARTITIONS):
        part_dir = out_dir / f"partition{part}"
        part_dir.mkdir(parents=True, exist_ok=True)

        part_li = li.loc[li["part"] == part].drop(columns=["part"])
        part_od = od.loc[od["part"] == part].drop(columns=["part"])

        part_li.to_parquet(part_dir / "lineitem.parquet", index=False)
        part_od.to_parquet(part_dir / "orders.parquet", index=False)

        print(
            f"    partition{part}: lineitem={len(part_li):,}, "
            f"orders={len(part_od):,}"
        )


def _unique_sorted(values: Iterable) -> List:
    """返回可 JSON 序列化的排序去重列表。"""
    return sorted(set(values))


def build_query_metadata(
    skew_name: str,
    zipf_alpha: float,
    lineitem: pd.DataFrame,
    orders: pd.DataFrame,
) -> Dict:
    """
    为某个 Skew 数据集构造查询生成元数据。

    这些映射用于 generate_queries.py 按 key 生成真实存在的谓词，避免：
    - orderkey 存在，但 o_orderstatus 随机选错导致 orders 侧空结果；
    - orderkey 存在，但 l_returnflag / l_quantity 组合不存在导致 lineitem 侧空结果。
    """
    valid_orderkeys = [int(k) for k in sorted(lineitem["l_orderkey"].unique())]

    # orders 表中 o_orderkey 唯一。若出现重复，保留第一条即可。
    orderstatus_by_key = {
        str(int(row.o_orderkey)): str(row.o_orderstatus)
        for row in orders[["o_orderkey", "o_orderstatus"]]
        .drop_duplicates(subset=["o_orderkey"])
        .itertuples(index=False)
    }

    returnflags_by_key = {
        str(int(key)): [str(v) for v in sorted(set(values))]
        for key, values in lineitem.groupby("l_orderkey")["l_returnflag"].unique().items()
    }

    quantities_by_key = {
        str(int(key)): [int(v) for v in sorted(set(values))]
        for key, values in lineitem.groupby("l_orderkey")["l_quantity"].unique().items()
    }

    # 更精细的映射：key -> returnflag -> quantities。
    # Q16 同时带 l_returnflag 和 l_quantity 时优先用这个映射，保证组合真实存在。
    quantities_by_key_returnflag: Dict[str, Dict[str, List[int]]] = {}
    grouped = lineitem.groupby(["l_orderkey", "l_returnflag"])["l_quantity"].unique()
    for (key, flag), values in grouped.items():
        key_str = str(int(key))
        flag_str = str(flag)
        quantities_by_key_returnflag.setdefault(key_str, {})[flag_str] = [
            int(v) for v in sorted(set(values))
        ]

    lineitem_combo_count = int(
        lineitem[["l_orderkey", "l_returnflag", "l_quantity"]].drop_duplicates().shape[0]
    )

    metadata = {
        "skew_name": skew_name,
        "zipf_alpha": float(zipf_alpha),
        "target_rows": int(len(lineitem)),
        "num_partitions": int(NUM_PARTITIONS),
        "partition_rule": "int(orderkey) % num_partitions",
        "valid_orderkeys": valid_orderkeys,
        "valid_returnflags": [str(v) for v in sorted(lineitem["l_returnflag"].unique())],
        "valid_orderstatus": [str(v) for v in sorted(orders["o_orderstatus"].unique())],
        "quantity_range": [int(lineitem["l_quantity"].min()), int(lineitem["l_quantity"].max())],
        "orderstatus_by_key": orderstatus_by_key,
        "returnflags_by_key": returnflags_by_key,
        "quantities_by_key": quantities_by_key,
        "quantities_by_key_returnflag": quantities_by_key_returnflag,
        "num_unique_orderkeys": int(lineitem["l_orderkey"].nunique()),
        "num_lineitem_combos": lineitem_combo_count,
        "total_lineitem_rows": int(len(lineitem)),
        "total_orders_rows": int(len(orders)),
    }
    return metadata


def save_query_metadata(
    skew_name: str,
    zipf_alpha: float,
    lineitem: pd.DataFrame,
    orders: pd.DataFrame,
    output_base: Path,
) -> Path:
    metadata = build_query_metadata(skew_name, zipf_alpha, lineitem, orders)
    path = output_base / f"query_generation_metadata_{skew_name}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(
        f"  [Metadata] {path.name}: keys={metadata['num_unique_orderkeys']:,}, "
        f"lineitem={metadata['total_lineitem_rows']:,}, orders={metadata['total_orders_rows']:,}"
    )
    return path


# -----------------------------------------------------------------------------
# 倾斜数据生成
# -----------------------------------------------------------------------------


def build_uniform_dataset(
    base_lineitem: pd.DataFrame,
    base_orders: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 SF=1 lineitem 中无放回采样 TARGET_ROWS，作为 Skew0。"""
    sample_n = min(TARGET_ROWS, len(base_lineitem))
    uniform_li = base_lineitem.sample(n=sample_n, replace=False, random_state=SEED)
    uniform_li = uniform_li.reset_index(drop=True)

    used_keys = set(int(k) for k in uniform_li["l_orderkey"].unique())
    uniform_od = base_orders.loc[base_orders["o_orderkey"].isin(used_keys)].copy()
    uniform_od = uniform_od.reset_index(drop=True)

    return uniform_li, uniform_od


def zipf_weights(num_keys: int, alpha: float) -> np.ndarray:
    """对排序后的真实 key 集合生成 rank-based Zipf 权重。"""
    if num_keys <= 0:
        raise ValueError("num_keys must be positive")

    if alpha <= 1.0:
        weights = np.ones(num_keys, dtype=np.float64)
    else:
        ranks = np.arange(1, num_keys + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, alpha)

    return weights / weights.sum()


def generate_skewed_lineitem(
    uniform_li: pd.DataFrame,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    基于 Skew0 的真实 key 集合，按 rank-based Zipf 有放回重采样 lineitem。

    对每个被采中的 orderkey，再从该 key 对应的真实 lineitem 行中随机抽一行。
    这样生成的 l_returnflag 和 l_quantity 一定来自真实数据。
    """
    unique_keys = np.asarray(sorted(uniform_li["l_orderkey"].unique()), dtype=np.int64)
    weights = zipf_weights(len(unique_keys), alpha)

    target_keys = rng.choice(unique_keys, size=TARGET_ROWS, replace=True, p=weights)

    # 预先建立 key -> 行号，避免 200K 次 DataFrame 过滤。
    indices_by_key = {
        int(key): np.asarray(indices, dtype=np.int64)
        for key, indices in uniform_li.groupby("l_orderkey").indices.items()
    }

    selected_indices = np.empty(TARGET_ROWS, dtype=np.int64)
    for i, key in enumerate(target_keys):
        candidate_indices = indices_by_key[int(key)]
        selected_indices[i] = int(rng.choice(candidate_indices))

    skewed_li = uniform_li.iloc[selected_indices].copy().reset_index(drop=True)
    return skewed_li


def align_orders(lineitem: pd.DataFrame, uniform_orders: pd.DataFrame) -> pd.DataFrame:
    """只保留当前 lineitem 中出现过的 orderkey 对应的 orders 行。"""
    used_keys = set(int(k) for k in lineitem["l_orderkey"].unique())
    orders = uniform_orders.loc[uniform_orders["o_orderkey"].isin(used_keys)].copy()
    return orders.reset_index(drop=True)


def print_skew_stats(skew_name: str, alpha: float, lineitem: pd.DataFrame, orders: pd.DataFrame) -> None:
    vc = lineitem["l_orderkey"].value_counts()
    top_10 = max(1, int(len(vc) * 0.1))
    top_10_cover = vc.iloc[:top_10].sum() / len(lineitem) if len(lineitem) else 0.0

    print(
        f"[{skew_name}] alpha={alpha:.2f} | "
        f"lineitem={len(lineitem):,} | orders={len(orders):,} | "
        f"unique_keys={len(vc):,} | top_key_freq={int(vc.iloc[0]) if len(vc) else 0:,} | "
        f"top_10%_keys_cover={top_10_cover:.2%}"
    )


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    base_li, base_od = read_tpch_tables(TPCH_PATH)

    print("\nBuilding Skew0 uniform 200K dataset ...")
    uniform_li, uniform_od = build_uniform_dataset(base_li, base_od, rng)
    print_skew_stats("Skew0", 1.0, uniform_li, uniform_od)

    print("  Saving Skew0 partitions ...")
    save_partitioned(uniform_li, uniform_od, OUTPUT_BASE / "Skew0")
    skew0_metadata = save_query_metadata("Skew0", 1.0, uniform_li, uniform_od, OUTPUT_BASE)

    # 保留一个 legacy 文件，避免旧代码仍读取 query_generation_metadata.json。
    legacy_path = OUTPUT_BASE / "query_generation_metadata.json"
    with open(skew0_metadata, "r", encoding="utf-8") as src, open(
        legacy_path, "w", encoding="utf-8"
    ) as dst:
        dst.write(src.read())
    print(f"  [Metadata] legacy copy: {legacy_path}")

    manifest = {
        "target_rows": TARGET_ROWS,
        "num_partitions": NUM_PARTITIONS,
        "seed": SEED,
        "datasets": {
            "Skew0": {
                "zipf_alpha": 1.0,
                "data_dir": str(OUTPUT_BASE / "Skew0"),
                "metadata": str(skew0_metadata),
            }
        },
    }

    print("\nBuilding skewed datasets ...")
    for skew_name, alpha in SKEW_CONFIGS.items():
        skewed_li = generate_skewed_lineitem(uniform_li, alpha, rng)
        skewed_od = align_orders(skewed_li, uniform_od)

        print_skew_stats(skew_name, alpha, skewed_li, skewed_od)
        print(f"  Saving {skew_name} partitions ...")
        save_partitioned(skewed_li, skewed_od, OUTPUT_BASE / skew_name)
        metadata_path = save_query_metadata(skew_name, alpha, skewed_li, skewed_od, OUTPUT_BASE)

        manifest["datasets"][skew_name] = {
            "zipf_alpha": float(alpha),
            "data_dir": str(OUTPUT_BASE / skew_name),
            "metadata": str(metadata_path),
        }

    manifest_path = OUTPUT_BASE / "skew_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nAll datasets ready under {OUTPUT_BASE}/")
    print(f"Manifest saved to {manifest_path}")
    print("\n下一步：")
    print("  1) python3 experiment/skew/generate_queries.py")
    print("  2) 在 executor1/executor2 上重建 lattice")
    print("  3) python3 train_rl_with_tpch.py Skew2")


if __name__ == "__main__":
    main()
