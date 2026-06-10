#!/usr/bin/env python3
"""
executor_main.py - 优化版（EC数量平衡 + 分布式路由）
"""
import os
import sys
import json
import pickle
import time
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, '/home/lkq/project')

from lattice_builder.quotient_cube import QuotientCubeBuilder, LocalLatticeMatcher
from utils.query_cell import QueryCell

"""
# 分发到 executor1
scp -r /home/lkq/project/lattice_builder/executor_main.py executor1:/home/lkq/project/lattice_builder
rsync -av lattice_builder/ executor1:/home/lkq/project/lattice_builder/
rsync -av utils/ executor1:/home/lkq/project/utils/

# 分发到 executor2  


rsync -av lattice_builder/ executor2:/home/lkq/project/lattice_builder/
rsync -av utils/ executor1:/home/lkq/project/utils/

# 在Executor1上
cd /home/lkq/project
export PARTITION_ID=0
python executor_main.py --mode build

python3 /home/lkq/project/lattice_builder/executor_main.py --mode build --partition 0

# 在Executor2上  
cd /home/lkq/project  
export PARTITION_ID=1
python executor_main.py --mode build

python3 /home/lkq/project/lattice_builder/executor_main.py --mode build --partition 1
"""

"""
# 在 Executor 1 上（本地执行，或 ssh 到 executor1）
for skew in Skew0 Skew1 Skew2 Skew3 Skew4 Skew-1; do
    export SKEW_NAME="$skew"
    export DATA_DIR="/home/lkq/tpch-data/$skew"
    export PARTITION_ID=0
    python3 /home/lkq/project/lattice_builder/executor_main.py --mode build
done

# 在 Executor 2 上
for skew in Skew0 Skew1 Skew2 Skew3 Skew4 Skew-1; do
    export SKEW_NAME="$skew"
    export DATA_DIR="/home/lkq/tpch-data/$skew"
    export PARTITION_ID=1
    python3 /home/lkq/project/lattice_builder/executor_main.py --mode build
done
"""




def get_table_config(data_dir=None, partition_id=None):
    if data_dir is None:
        data_dir = os.environ.get('DATA_DIR', '/home/lkq/tpch-data/Skew0')
    if partition_id is None:
        partition_id = int(os.environ.get('PARTITION_ID', 0))

    dataset_name = os.environ.get("DATASET_NAME", "tpch").lower()

    if dataset_name == "job":
        return {
            "title": {
                "parquet": f"{data_dir}/partition{partition_id}/title.parquet",
                "agg_col": "_cnt",
                "dims": ["id", "kind_id", "production_year_bucket"],
                "bucket_config": {},
                "min_ec_size": 2
            },
            "movie_info_idx": {
                "parquet": f"{data_dir}/partition{partition_id}/movie_info_idx.parquet",
                "agg_col": "_cnt",
                "dims": ["movie_id", "info_type_id", "info_bucket"],
                "bucket_config": {},
                "min_ec_size": 2
            },
            "movie_companies": {
                "parquet": f"{data_dir}/partition{partition_id}/movie_companies.parquet",
                "agg_col": "_cnt",
                "dims": ["movie_id", "company_id_bucket", "company_type_id"],
                "bucket_config": {},
                "min_ec_size": 2
            },
            "movie_keyword": {
                "parquet": f"{data_dir}/partition{partition_id}/movie_keyword.parquet",
                "agg_col": "_cnt",
                "dims": ["movie_id", "keyword_id_bucket", "keyword_id"],
                "bucket_config": {},
                "min_ec_size": 2
            },
        }

    return {
        "lineitem": {
            "parquet": f"{data_dir}/partition{partition_id}/lineitem.parquet",
            "agg_col": "l_extendedprice",
            "dims": ["l_orderkey", "l_returnflag", "l_quantity"],
            "bucket_config": {"l_quantity": 10},
            "min_ec_size": 1
        },
        "orders": {
            "parquet": f"{data_dir}/partition{partition_id}/orders.parquet",
            "agg_col": "o_totalprice",
            "dims": ["o_orderkey", "o_custkey", "o_orderstatus"],
            "bucket_config": {"o_orderkey": 100},
            "min_ec_size": 1
        }
    }
  

# [新增] 动态计算min_ec_size，平衡EC数量（对应文档问题4）
def calculate_balanced_min_ec_size(table_name: str, data_rows: int, target_ec_count: int = 15000) -> int: 
    """
    根据数据量动态计算min_ec_size，使EC数量趋近target_ec_count（默认1.5万）
    
    策略：
    - lineitem数据量大(300万)，需要更大的min_ec_size来减少EC数量到1-2万
    - orders数据量小(75万)，保持较小min_ec_size 
    """
    if data_rows == 0:
        return 2 
      
    # 粗略估算：每个EC平均覆盖的行数 = data_rows / target_ec_count 
    # 但为了安全，取2倍余量（因为聚合值分布不均匀）
    calculated_size = max(2, data_rows // (target_ec_count // 2))
     
    if table_name == "lineitem":
        # lineitem通常聚合值分布更分散，需要更大的min_ec_size
        # 原始：300万行 -> 10万EC（每EC 30行），目标：300万 -> 1.5万EC（每EC 200行）
        calculated_size = max(calculated_size, 30)  # 至少30行
        
        # 根据数据量细分
        if data_rows > 2000000:
            calculated_size = max(calculated_size, 50)  # 大数据量用50+
        elif data_rows > 500000:
            calculated_size = max(calculated_size, 20)
            
    elif table_name == "orders":  
        # orders数据量较小，保持较小值避免EC过少
        calculated_size = min(calculated_size, 10)  # 最多10
        calculated_size = max(calculated_size, 2)   # 至少2
        
    print(f"    [动态调整] {table_name}: {data_rows:,}行 -> min_ec_size={calculated_size}")
    return calculated_size


# [新增] 分布式路由逻辑（对应文档问题3）
def route_query(query_cell: QueryCell, num_executors: int = 2) -> str:
    if query_cell.partition_key == "*":
        return "broadcast"

    executor_id = 101 + (int(query_cell.partition_key) % num_executors)
    return f"executor_{executor_id}"


def batch_route(queries_json: str, num_executors: int = 2) -> Dict[str, List[dict]]:
    """
    批量路由查询到对应的 Executor
    返回：{executor_id: [query1, query2, ...]}
    """
    queries = json.loads(queries_json)
    routed = defaultdict(list)
    
    for q in queries:
        query_cell = QueryCell(**q)
        target = route_query(query_cell, num_executors)
        
        if target == "broadcast":
            for i in range(num_executors):
                routed[f"executor_{101+i}"].append(q)
        else:
            routed[target].append(q)
    
    return dict(routed)


def diagnose_paths():
    """诊断数据路径"""
    partition_id = int(os.environ.get('PARTITION_ID', 0))
    print(f"🔍 诊断分区 {partition_id}:")
    
    # 先从环境变量读出当前值，再传给函数
    data_dir = os.environ.get('DATA_DIR', '/home/lkq/tpch-data/Skew0')
    for table, cfg in get_table_config(data_dir, partition_id).items():
        path = cfg['parquet']
        exists = os.path.exists(path)
        size = os.path.getsize(path)/1024/1024 if exists else 0
        print(f"  {table}: {path} {'✅' if exists else '❌'} ({size:.1f}MB)")
        
        if not exists:
            alt_path = os.path.expanduser(f"~/tpch-data/partition{partition_id}/{table}.parquet")
            if os.path.exists(alt_path):
                print(f"    💡 发现分区路径: {alt_path}")
                TABLE_CONFIG[table]['parquet'] = alt_path


def build_phase():
    """[优化] 构建阶段 - 使用动态min_ec_size平衡EC数量"""
    partition_id = int(os.environ.get('PARTITION_ID', 0))
    executor_id = f"executor_{101 + partition_id}"
    skew_name = os.environ.get('SKEW_NAME', 'Skew0')  # ← 新增
    data_dir = os.environ.get('DATA_DIR', '/home/lkq/tpch-data/Skew0')  # ← 新增
    
    print(f"\n{'='*60}")
    print(f"[{executor_id}] Building Lattice (Partition {partition_id})")
    print(f"[Executor {partition_id}] Building from {data_dir}")
    print(f"{'='*60}")
    
    diagnose_paths()
    
    # output_dir = "/home/lkq/lattice"
    # 按倾斜度隔离保存目录
    output_dir = f"/home/lkq/lattice/{skew_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for table_name, cfg in get_table_config(data_dir, partition_id).items():
        print(f"\n📊 Processing {table_name}...")
        
        if not os.path.exists(cfg["parquet"]):
            print(f"   ❌ 跳过: 文件不存在")
            continue
            
        try:
            builder = QuotientCubeBuilder(
                partition_id, 
                table_name,
                min_ec_size=cfg["min_ec_size"],  # 直接使用TABLE_CONFIG中的值
                max_ec_size=10000,
                bucket_config=cfg.get("bucket_config", {}),
                memory_limit_mb=3000  # [新增] 3GB内存限制
            )
            
            ecs = builder.build_from_partition(
                parquet_path=cfg["parquet"],
                dim_cols=cfg["dims"],
                agg_col=cfg["agg_col"]
            )
            
            saved_path = builder.save_lattice(output_dir)
            results.append({
                'table': table_name,
                'ecs': len(ecs),
                'min_ec_size_used': cfg["min_ec_size"],  
                'data_rows': builder._table_row_count,
                'path': saved_path
            })
            print(f"   ✅ 成功: {len(ecs)} ECs (min_ec_size={cfg['min_ec_size']})")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 构建摘要
    summary = {
        'executor_id': executor_id,
        'partition_id': partition_id,
        'results': results,
        'total_ecs': sum(r['ecs'] for r in results),
        'timestamp': time.time()
    }
    with open(f"{output_dir}/summary_{partition_id}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"完成: {len(results)} 张表, 总计 {summary['total_ecs']} ECs")
    print(f"{'='*60}")


def match_phase(query_json: str):
    """匹配阶段 - 适配新版MatchResult"""
    query_cell = QueryCell.from_json(query_json)
    table_name = query_cell.table
    
    partition_id = int(os.environ.get('PARTITION_ID', 0))
    skew_name = os.environ.get('SKEW_NAME', 'Skew0')  # ← 新增
    # lattice_path = f"/home/lkq/lattice/{table_name}_p{partition_id}.pkl"
    lattice_path = f"/home/lkq/lattice/{skew_name}/{table_name}_p{partition_id}.pkl"
    
    if not os.path.exists(lattice_path):
        return {
            "matched": False,
            "fallback_to_scan": True,
            "reason": "lattice_not_found",
            "executor_id": f"executor_{101 + partition_id}"
        }
    
    matcher = LocalLatticeMatcher(lattice_path, f"executor_{101 + partition_id}")
    result = matcher.match(query_cell)
    
    if isinstance(result, dict):
        result.setdefault('fallback_to_scan', not result.get('matched', False))
        result['executor_id'] = f"executor_{101 + partition_id}"
    return result


def batch_match(queries_json: str):
    """批量匹配多个查询"""
    queries = json.loads(queries_json)
    return [match_phase(json.dumps(q)) for q in queries]


if __name__ == "__main__":
    import argparse
    import pandas as pd  # [新增] 确保在函数内可用
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['build', 'match', 'diagnose', 'route'], default='build')
    parser.add_argument('--query', type=str)
    parser.add_argument('--partition', type=int, default=None)
    
    args = parser.parse_args()
    
    # 设置分区ID
    if args.partition is not None:
        os.environ['PARTITION_ID'] = str(args.partition)
    elif 'PARTITION_ID' not in os.environ:
        os.environ['PARTITION_ID'] = '0'
    
    if args.mode == 'diagnose':
        diagnose_paths()
    elif args.mode == 'build':
        build_phase()
    elif args.mode == 'match' and args.query:
        print(json.dumps(match_phase(args.query), ensure_ascii=False, indent=2))
    elif args.mode == 'route' and args.query:
        # [新增] 测试路由功能
        qc = QueryCell.from_json(args.query)
        target = route_query(qc)
        print(f"Query {qc.query_id} -> {target}")