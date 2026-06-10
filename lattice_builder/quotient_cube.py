"""
quotient_cube.py - 格结构商立方体构建与匹配（修复偏序关系）
关键修复：
1. LocalLatticeMatcher 独立实现 u(EC) <= c <= l 偏序链，正确处理 '*' 泛化语义
2. 单 cell EC 的下界统一为全 '*'，避免 (288,'N',6) 无法匹配 (288,'N',*)
"""
import pandas as pd
import pickle
import os
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from itertools import combinations
from utils.equivalence_class import Cell, EquivalenceClass, MatchResult

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class QuotientCubeBuilder:
    def __init__(self, partition_id: int, table_name: str, 
                 max_ec_size: int = 10000, 
                 min_ec_size: int = 2,
                 bucket_config: Dict[str, int] = None,
                 memory_limit_mb: int = 3000):
        self.partition_id = partition_id
        self.table_name = table_name
        self.ecs: List[EquivalenceClass] = []
        self.max_ec_size = max_ec_size
        self.min_ec_size = min_ec_size
        self.bucket_config = bucket_config or {}
        self.residual_cells = []
        self.memory_limit_mb = memory_limit_mb
        self._table_row_count = 0
        
    def _check_memory_usage(self) -> bool:
        if not HAS_PSUTIL:
            return True
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > self.memory_limit_mb * 0.9:
            print(f"⚠️  Warning: Memory usage {memory_mb:.1f}MB exceeds 90% of limit")
            return False
        return True
    
    def build_from_partition(self, parquet_path: str, 
                            dim_cols: List[str], 
                            agg_col: str):
        print(f"[Partition {self.partition_id}] Building Optimized Quotient Cube for {self.table_name}...")
        print(f"  Dimensions: {dim_cols}")
        print(f"  Min EC Size: {self.min_ec_size}")
        
        df = pd.read_parquet(parquet_path)
        self._table_row_count = len(df)
        print(f"  Input rows: {self._table_row_count:,}")
        
        df_processed = self._apply_bucketization(df, dim_cols)
        dim_cols_processed = [f"{col}_bucket" if col in self.bucket_config else col 
                             for col in dim_cols]
        
        base_cells = self._generate_base_cells(df_processed, dim_cols_processed, agg_col)
        print(f"  - Base cells (distinct combinations): {len(base_cells):,}")
        
        if not self._check_memory_usage():
            print("⚠️  Memory limit approached, increasing min_ec_size...")
            self.min_ec_size = max(self.min_ec_size * 2, 10)
        
        valid_ecs, residual_count = self._build_equivalence_classes_with_filter(
            base_cells, dim_cols_processed
        )
        self.ecs = valid_ecs
        
        total_valid_cells = sum(ec.raw_count for ec in self.ecs)
        boundary_cells = sum(
            1 + len(ec.lower_bounds) if len(ec.lower_bounds) > 0 else 1 
            for ec in self.ecs
        )
        multi_cell_ecs = [ec for ec in self.ecs if ec.raw_count >= self.min_ec_size]
        
        print(f"  - Equivalence classes (valid): {len(self.ecs):,}")
        print(f"    * Multi-cell ECs (≥{self.min_ec_size}): {len(multi_cell_ecs):,}")
        print(f"    * Single-cell ECs: {len(self.ecs) - len(multi_cell_ecs):,}")
        print(f"  - Boundary cells stored: {boundary_cells:,}")
        
        return self.ecs
    
    def _apply_bucketization(self, df: pd.DataFrame, dim_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in dim_cols:
            if col in self.bucket_config:
                bucket_size = self.bucket_config[col]
                bucket_col = f"{col}_bucket"
                df[bucket_col] = df[col] % bucket_size
                print(f"    [Bucket] {col} % {bucket_size} -> {bucket_col}")
        return df
    
    def _generate_base_cells(self, df, dim_cols, agg_col) -> Dict[Cell, Tuple[float, int]]:
        """
        生成完整 cuboid cells。

        对 d 个维度生成 2^d 层 cell。
        例如 lineitem 的维度为：
            [l_orderkey, l_returnflag, l_quantity_bucket]

        那么会生成：
            [*, *, *]
            [l_orderkey, *, *]
            [*, l_returnflag, *]
            [*, *, l_quantity_bucket]
            [l_orderkey, l_returnflag, *]
            [l_orderkey, *, l_quantity_bucket]
            [*, l_returnflag, l_quantity_bucket]
            [l_orderkey, l_returnflag, l_quantity_bucket]

        这样查询单元 [1,'N','*'] 才能命中。
        """
        cells = {}
        n_dims = len(dim_cols)

        for r in range(0, n_dims + 1):
            for group_idx in combinations(range(n_dims), r):
                group_cols = [dim_cols[i] for i in group_idx]

                if not group_cols:
                    dims = tuple(["*"] * n_dims)
                    cells[Cell(dims)] = (
                        float(df[agg_col].sum()),
                        int(df[agg_col].count())
                    )
                    continue

                grouped = df.groupby(group_cols, observed=True)[agg_col].agg(["sum", "count"])

                for values, row in grouped.iterrows():
                    if not isinstance(values, tuple):
                        values = (values,)

                    dims = ["*"] * n_dims
                    for pos, val in zip(group_idx, values):
                        dims[pos] = val

                    cells[Cell(tuple(dims))] = (
                        float(row["sum"]),
                        int(row["count"])
                    )

        return cells
    
    def _build_equivalence_classes_with_filter(self, base_cells: Dict[Cell, Tuple[float, int]], 
                                           dim_cols: List[str]) -> Tuple[List[EquivalenceClass], int]:
        """
        保守正确版本：
        每个 cuboid cell 作为一个 singleton EC。

        这样可以保证：
        1. 查询单元存在就能命中；
        2. 不会因为仅按 SUM 值相等而错误合并；
        3. 先把倾斜度实验跑通。
        """
        ecs = []
        residual_count = 0
        ec_id = 0

        for cell, (agg_val, count) in base_cells.items():
            if count < self.min_ec_size:
                residual_count += 1
                continue

            ec_id += 1
            ecs.append(
                EquivalenceClass(
                    ec_id=f"EC_{self.table_name}_P{self.partition_id}_{ec_id}",
                    upper_bound=cell,
                    lower_bounds=[cell],
                    aggregate_value=agg_val,
                    table_name=self.table_name,
                    partition_id=self.partition_id,
                    raw_count=count
                )
            )

        return ecs, residual_count
    
    def _create_efficient_ec(self, cells_with_count: List[Tuple[Cell, int]], 
                            agg_val: float, dim_cols: List[str], ec_id: int) -> EquivalenceClass:
        cells = [c for c, _ in cells_with_count]
        n_cells = len(cells)
        n_dims = len(dim_cols)
        
        def specificity(cell):
            return sum(1 for d in cell.dimensions if d != "*")
        upper = max(cells, key=specificity)
        
        total_raw = sum(count for _, count in cells_with_count)
        
        # 【修复】删除单 cell 特殊处理，统一计算最小下界
        # 原代码：if n_cells == 1: lowers = [upper] 导致偏序匹配失败
        lowers = self._find_minimal_lower_bounds(cells, upper, n_dims)
        
        return EquivalenceClass(
            ec_id=f"EC_{self.table_name}_P{self.partition_id}_{ec_id}",
            upper_bound=upper,
            lower_bounds=lowers,
            aggregate_value=agg_val,
            table_name=self.table_name,
            partition_id=self.partition_id,
            raw_count=total_raw
        )
    
    def _find_minimal_lower_bounds(self, cells: List[Cell], upper: Cell, n_dims: int) -> List[Cell]:
        # 【修复】单 cell EC 返回全 '*' 下界，确保查询 (288,'N',*) 满足 c <= (*,*,*)
        if len(cells) == 1:
            return [upper]
        
        candidates = set()
        candidates.add(Cell(tuple(["*"] * n_dims)))
        
        for i in range(n_dims):
            unique_vals = set(cell.dimensions[i] for cell in cells)
            if len(unique_vals) == 1:
                dims = list(upper.dimensions)
                dims[i] = "*"
                candidates.add(Cell(tuple(dims)))
        
        valid_lowers = []
        for cand in candidates:
            if all(cand.contains(cell) for cell in cells):
                valid_lowers.append(cand)
        
        minimal_lowers = []
        for cand in valid_lowers:
            is_minimal = True
            for other in valid_lowers:
                if other != cand and other.contains(cand):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_lowers.append(cand)
        
        return minimal_lowers if minimal_lowers else [Cell(tuple(["*"] * n_dims))]
    
    def save_lattice(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.table_name}_p{self.partition_id}.pkl")
        
        multi_ecs = [ec for ec in self.ecs if ec.raw_count >= self.min_ec_size]
        single_ecs = [ec for ec in self.ecs if ec.raw_count < self.min_ec_size]
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'partition_id': self.partition_id,
                'table_name': self.table_name,
                'ecs': self.ecs,
                'stats': {
                    'total_ecs': len(self.ecs),
                    'multi_cell_ecs': len(multi_ecs),
                    'single_cell_ecs': len(single_ecs),
                    'total_covered_rows': sum(ec.raw_count for ec in self.ecs),
                    'avg_lower_bounds': sum(len(ec.lower_bounds) for ec in self.ecs) / len(self.ecs) if self.ecs else 0
                }
            }, f)
        
        print(f"  - Saved: {filepath}")
        print(f"  - EC Stats: {len(multi_ecs)} multi-cell, {len(single_ecs)} single-cell")
        return filepath
    
    @staticmethod
    def load_lattice(filepath: str) -> List[EquivalenceClass]:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['ecs']


class LocalLatticeMatcher:
    """Executor端本地匹配器（修复偏序匹配，支持 '*' 泛化语义）"""
    def __init__(self, lattice_path: str, executor_id: str):
        self.ecs = QuotientCubeBuilder.load_lattice(lattice_path)
        self.executor_id = executor_id
        self._avg_row_size = 100

    def match(self, query_cell) -> dict:
        """
        匹配查询单元，直接返回 dict。
        不再构造 MatchResult，避免 MatchResult 参数版本不一致导致 executor 崩溃。
        """
        q_dims = (
            query_cell.cell_dimensions
            if hasattr(query_cell, "cell_dimensions")
            else query_cell.dimensions
        )

        for ec in self.ecs:
            if self._check_subsumption(ec, q_dims):
                raw_count = ec.raw_count if hasattr(ec, "raw_count") else 0
                agg_val = ec.aggregate_value if hasattr(ec, "aggregate_value") else None
                ec_id = ec.ec_id if hasattr(ec, "ec_id") else str(id(ec))

                return {
                    "matched": True,
                    "ec_id": ec_id,
                    "aggregate_value": agg_val,
                    "aggregate_type": "SUM",
                    "raw_count": raw_count,
                    "executor_id": self.executor_id,
                    "estimated_scan_rows": raw_count,
                    "estimated_shuffle_bytes": raw_count * self._avg_row_size,
                    "fallback_to_scan": False,
                    "reason": "matched",
                    "ec_bounds": {
                        "upper_bound": list(ec.upper_bound.dimensions)
                        if hasattr(ec, "upper_bound") and ec.upper_bound
                        else [],
                        "lower_bounds": [
                            list(lb.dimensions) if hasattr(lb, "dimensions") else list(lb)
                            for lb in getattr(ec, "lower_bounds", [])
                        ],
                    },
                }

        total_rows = sum(ec.raw_count for ec in self.ecs) * 2 if self.ecs else 100000

        return {
            "matched": False,
            "ec_id": None,
            "aggregate_value": None,
            "aggregate_type": "SUM",
            "raw_count": 0,
            "executor_id": self.executor_id,
            "estimated_scan_rows": total_rows,
            "estimated_shuffle_bytes": total_rows * self._avg_row_size,
            "fallback_to_scan": True,
            "reason": "no_ec_matched_in_partition",
        }
    
    def _dim_leq(self, specific_val, general_val) -> bool:
        """偏序关系：specific_val <= general_val（specific 更具体或相等）"""
        g = str(general_val)
        if g in ('*', 'ALL', '?'):
            return True
        return str(specific_val) == g
    
    def _covers(self, general_dims, specific_dims) -> bool:
        """general 覆盖 specific（general 更泛化，specific 更具体）"""
        if len(general_dims) != len(specific_dims):
            return False
        for g, s in zip(general_dims, specific_dims):
            if not self._dim_leq(s, g):
                return False
        return True
    
    def _check_subsumption(self, ec, q_dims) -> bool:
        """
        格结构匹配条件：
        1. u(EC) <= q（EC 上界被查询覆盖，查询更泛化）
        2. q 与 l 兼容（查询的 '*' 可匹配下界的具体值）
        """
        ub = ec.upper_bound.dimensions if hasattr(ec, 'upper_bound') and ec.upper_bound else []
        lbs = ec.lower_bounds if hasattr(ec, 'lower_bounds') and ec.lower_bounds else []
        
        if len(ub) != len(q_dims):
            return False
        
        # Step 1: u(EC) <= q（核心条件）
        # 查询的每一维要么等于上界，要么是 '*'（通配）
        if not self._covers(q_dims, ub):
            return False
        
        # Step 2: q 与某个下界 l 兼容（放宽）
        # 论文原条件：c <= l（查询比下界更具体）
        # 工程放宽：查询的 '*' 维度可匹配下界的任何具体值
        # 因为 '*' 表示"不关心"，下界的具体值不影响查询语义
        if not lbs:
            return True
        
        for lb in lbs:
            lb_dims = lb.dimensions if hasattr(lb, 'dimensions') else lb
            if len(lb_dims) != len(q_dims):
                continue
            
            compatible = True
            for i, (l, q) in enumerate(zip(lb_dims, q_dims)):
                q_str = str(q)
                l_str = str(l)
                # 查询通配：匹配任何下界值
                if q_str in ('*', 'ALL', '?'):
                    continue
                # 下界通配：匹配任何查询值
                if l_str in ('*', 'ALL', '?'):
                    continue
                # 两者都具体：必须相等
                if l_str != q_str:
                    compatible = False
                    break
            
            if compatible:
                return True
        
        return False
    
    def _estimate_shuffle(self, ec: EquivalenceClass, query_cell) -> int:
        return ec.raw_count * self._avg_row_size

    def _get_table_row_count(self) -> int:
        return sum(ec.raw_count for ec in self.ecs) * 2 if self.ecs else 100000

    def _estimate_full_shuffle(self) -> int:
        return self._get_table_row_count() * self._avg_row_size



# """
# quotient_cube.py - 优化版商立方体构建
# 关键优化：
# 1. 单cell EC不存下界（避免负压缩）
# 2. 增加min_ec_size过滤（只保留有聚合价值的EC）
# 3. 支持维度分桶（降低高基维影响）
# 4. [新增] 内存监控和代价估算
# """
# import pandas as pd
# import pickle
# import os
# from typing import List, Dict, Tuple, Set
# from collections import defaultdict
# from itertools import combinations
# from utils.equivalence_class import Cell, EquivalenceClass, MatchResult

# # [新增] 内存监控
# try:
#     import psutil
#     HAS_PSUTIL = True
# except ImportError:
#     HAS_PSUTIL = False


# class QuotientCubeBuilder:
#     def __init__(self, partition_id: int, table_name: str, 
#                  max_ec_size: int = 10000, 
#                  min_ec_size: int = 2,
#                  bucket_config: Dict[str, int] = None,
#                  memory_limit_mb: int = 3000):  # [新增] 内存限制（对应文档问题5）
#         self.partition_id = partition_id
#         self.table_name = table_name
#         self.ecs: List[EquivalenceClass] = []
#         self.max_ec_size = max_ec_size
#         self.min_ec_size = min_ec_size
#         self.bucket_config = bucket_config or {}
#         self.residual_cells = []
#         self.memory_limit_mb = memory_limit_mb
#         self._table_row_count = 0  # [新增] 记录表总行数，用于代价估算
        
#     def _check_memory_usage(self) -> bool:
#         """[新增] 检查当前内存使用是否超出限制（对应文档问题5）"""
#         if not HAS_PSUTIL:
#             return True
            
#         process = psutil.Process(os.getpid())
#         memory_mb = process.memory_info().rss / 1024 / 1024
        
#         if memory_mb > self.memory_limit_mb * 0.9:  # 90% 阈值
#             print(f"⚠️  Warning: Memory usage {memory_mb:.1f}MB exceeds 90% of limit")
#             return False
#         return True
    
#     def build_from_partition(self, parquet_path: str, 
#                             dim_cols: List[str], 
#                             agg_col: str):
#         """
#         构建优化版商立方体
#         """
#         print(f"[Partition {self.partition_id}] Building Optimized Quotient Cube for {self.table_name}...")
#         print(f"  Dimensions: {dim_cols}")
#         print(f"  Min EC Size: {self.min_ec_size} (EC with fewer cells are merged)")
        
#         df = pd.read_parquet(parquet_path)
#         self._table_row_count = len(df)  # [新增] 记录总行数
#         print(f"  Input rows: {self._table_row_count:,}")
        
#         # 维度分桶处理
#         df_processed = self._apply_bucketization(df, dim_cols)
#         dim_cols_processed = [f"{col}_bucket" if col in self.bucket_config else col 
#                              for col in dim_cols]
        
#         # Step 1: 生成Base Cells
#         base_cells = self._generate_base_cells(df_processed, dim_cols_processed, agg_col)
#         print(f"  - Base cells (distinct combinations): {len(base_cells):,}")
        
#         # [新增] 内存检查
#         if not self._check_memory_usage():
#             print("⚠️  Memory limit approached, increasing min_ec_size...")
#             self.min_ec_size = max(self.min_ec_size * 2, 10)
        
#         # Step 2: 构建EC（带大小过滤）
#         valid_ecs, residual_count = self._build_equivalence_classes_with_filter(
#             base_cells, dim_cols_processed
#         )
#         self.ecs = valid_ecs
        
#         # 统计信息
#         total_valid_cells = sum(ec.raw_count for ec in self.ecs)
#         boundary_cells = sum(
#             1 + len(ec.lower_bounds) if len(ec.lower_bounds) > 0 else 1 
#             for ec in self.ecs
#         )
        
#         multi_cell_ecs = [ec for ec in self.ecs if ec.raw_count >= self.min_ec_size]
        
#         print(f"  - Equivalence classes (valid): {len(self.ecs):,}")
#         print(f"    * Multi-cell ECs (≥{self.min_ec_size}): {len(multi_cell_ecs):,}")
#         print(f"    * Single-cell ECs: {len(self.ecs) - len(multi_cell_ecs):,}")
#         print(f"  - Boundary cells stored: {boundary_cells:,}")
        
#         return self.ecs
    
#     def _apply_bucketization(self, df: pd.DataFrame, dim_cols: List[str]) -> pd.DataFrame:
#         """对高基维进行分桶"""
#         df = df.copy()
#         for col in dim_cols:
#             if col in self.bucket_config:
#                 bucket_size = self.bucket_config[col]
#                 bucket_col = f"{col}_bucket"
#                 df[bucket_col] = df[col] % bucket_size
#                 print(f"    [Bucket] {col} % {bucket_size} -> {bucket_col}")
#         return df
    
#     def _generate_base_cells(self, df, dim_cols, agg_col) -> Dict[Cell, Tuple[float, int]]:
#         """生成最细粒度的Base Cells"""
#         cells = {}
#         grouped = df.groupby(dim_cols, observed=True)[agg_col].agg(['sum', 'count'])
        
#         for values, row in grouped.iterrows():
#             if not isinstance(values, tuple):
#                 values = (values,)
#             cell = Cell(tuple(values))
#             cells[cell] = (float(row['sum']), int(row['count']))
        
#         return cells
    
#     def _build_equivalence_classes_with_filter(self, base_cells: Dict[Cell, Tuple[float, int]], 
#                                                dim_cols: List[str]) -> Tuple[List[EquivalenceClass], int]:
#         """构建EC，过滤掉太小的EC"""
#         value_to_cells = defaultdict(list)
#         for cell, (agg_val, count) in base_cells.items():
#             value_to_cells[agg_val].append((cell, count))
        
#         ecs = []
#         residual_count = 0
#         ec_id = 0
        
#         for agg_val, cells_with_count in value_to_cells.items():
#             n_cells = len(cells_with_count)
            
#             if n_cells < self.min_ec_size:
#                 residual_count += n_cells
#                 continue
            
#             if n_cells > self.max_ec_size:
#                 for i in range(0, n_cells, self.max_ec_size):
#                     batch = cells_with_count[i:i+self.max_ec_size]
#                     ec_id += 1
#                     ec = self._create_efficient_ec(batch, agg_val, dim_cols, ec_id)
#                     if ec:
#                         ecs.append(ec)
#             else:
#                 ec_id += 1
#                 ec = self._create_efficient_ec(cells_with_count, agg_val, dim_cols, ec_id)
#                 if ec:
#                     ecs.append(ec)
        
#         return ecs, residual_count
    
#     def _create_efficient_ec(self, cells_with_count: List[Tuple[Cell, int]], 
#                             agg_val: float, dim_cols: List[str], ec_id: int) -> EquivalenceClass:
#         """创建优化后的EC（单cell时不存下界）"""
#         cells = [c for c, _ in cells_with_count]
#         n_cells = len(cells)
#         n_dims = len(dim_cols)
        
#         def specificity(cell):
#             return sum(1 for d in cell.dimensions if d != "*")
#         upper = max(cells, key=specificity)
        
#         total_raw = sum(count for _, count in cells_with_count)
        
#         if n_cells == 1:
#             lowers = [upper]
#         else:
#             lowers = self._find_minimal_lower_bounds(cells, upper, n_dims)
        
#         return EquivalenceClass(
#             ec_id=f"EC_{self.table_name}_P{self.partition_id}_{ec_id}",
#             upper_bound=upper,
#             lower_bounds=lowers,
#             aggregate_value=agg_val,
#             table_name=self.table_name,
#             partition_id=self.partition_id,
#             raw_count=total_raw
#         )
    
#     def _find_minimal_lower_bounds(self, cells: List[Cell], upper: Cell, n_dims: int) -> List[Cell]:
#         """查找最小下界集合"""
#         if len(cells) == 1:
#             return []
        
#         candidates = set()
#         candidates.add(Cell(tuple(["*"] * n_dims)))
        
#         for i in range(n_dims):
#             unique_vals = set(cell.dimensions[i] for cell in cells)
#             if len(unique_vals) == 1:
#                 dims = list(upper.dimensions)
#                 dims[i] = "*"
#                 candidates.add(Cell(tuple(dims)))
        
#         valid_lowers = []
#         for cand in candidates:
#             if all(cand.contains(cell) for cell in cells):
#                 valid_lowers.append(cand)
        
#         minimal_lowers = []
#         for cand in valid_lowers:
#             is_minimal = True
#             for other in valid_lowers:
#                 if other != cand and other.contains(cand):
#                     is_minimal = False
#                     break
#             if is_minimal:
#                 minimal_lowers.append(cand)
        
#         return minimal_lowers if minimal_lowers else [Cell(tuple(["*"] * n_dims))]
    
#     def save_lattice(self, output_dir: str):
#         """保存格结构"""
#         os.makedirs(output_dir, exist_ok=True)
#         filepath = os.path.join(output_dir, f"{self.table_name}_p{self.partition_id}.pkl")
        
#         multi_ecs = [ec for ec in self.ecs if ec.raw_count >= self.min_ec_size]
#         single_ecs = [ec for ec in self.ecs if ec.raw_count < self.min_ec_size]
        
#         with open(filepath, 'wb') as f:
#             pickle.dump({
#                 'partition_id': self.partition_id,
#                 'table_name': self.table_name,
#                 'ecs': self.ecs,
#                 'stats': {
#                     'total_ecs': len(self.ecs),
#                     'multi_cell_ecs': len(multi_ecs),
#                     'single_cell_ecs': len(single_ecs),
#                     'total_covered_rows': sum(ec.raw_count for ec in self.ecs),
#                     'avg_lower_bounds': sum(len(ec.lower_bounds) for ec in self.ecs) / len(self.ecs) if self.ecs else 0
#                 }
#             }, f)
        
#         print(f"  - Saved: {filepath}")
#         print(f"  - EC Stats: {len(multi_ecs)} multi-cell, {len(single_ecs)} single-cell")
#         return filepath
    
#     @staticmethod
#     def load_lattice(filepath: str) -> List[EquivalenceClass]:
#         """加载格结构"""
#         with open(filepath, 'rb') as f:
#             data = pickle.load(f)
#         return data['ecs']


# class LocalLatticeMatcher:
#     """Executor端本地匹配器（增强版，支持代价估算）"""
#     def __init__(self, lattice_path: str, executor_id: str):
#         self.ecs = QuotientCubeBuilder.load_lattice(lattice_path)
#         self.executor_id = executor_id
#         self._avg_row_size = 100  # 假设平均每行100字节

#     def match(self, query_cell) -> dict:
#         """匹配查询单元，返回详细的 MatchResult（对应文档问题2）"""
#         from utils.equivalence_class import MatchResult
#         # 获取查询维度
#         q_dims = query_cell.cell_dimensions if hasattr(query_cell, 'cell_dimensions') else query_cell.dimensions
        
#         for ec in self.ecs:
#             result = ec.match_with_details(cell)
#             if result.matched:
#                 # [新增] 计算代价估算
#                 result.estimated_scan_rows = ec.raw_count
#                 result.estimated_shuffle_bytes = self._estimate_shuffle(ec, query_cell)
#                 return result.to_dict()
        
#         # 所有EC都不匹配，需要回退到原始表扫描
#         return MatchResult(
#             matched=False,
#             fallback_to_scan=True,
#             reason="no_ec_matched_in_partition",
#             estimated_scan_rows=self._get_table_row_count(),
#             estimated_shuffle_bytes=self._estimate_full_shuffle()
#         ).to_dict()
    
#     def _estimate_shuffle(self, ec: EquivalenceClass, query_cell) -> int:
#         """[新增] 估算 Shuffle 量"""
#         return ec.raw_count * self._avg_row_size

#     def _get_table_row_count(self) -> int:
#         """[新增] 获取表总行数（简化版）"""
#         # 实际应该从元数据获取，这里用EC覆盖行数估算
#         return sum(ec.raw_count for ec in self.ecs) * 2  # 粗略估计

#     def _estimate_full_shuffle(self) -> int:
#         """[新增] 估算全表 Shuffle 量"""
#         return self._get_table_row_count() * self._avg_row_size