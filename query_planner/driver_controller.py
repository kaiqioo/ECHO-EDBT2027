#!/usr/bin/env python3
"""
driver_controller.py - Driver端主控制器
功能：
1. 整合TPC-H工作负载生成（Zipfian分布）
2. 拆分为子查询/数据单元（QuerySplitter）
3. 路由到Executor并收集结果
4. 构建超图数据结构（支持Fallback标记）

设计：与超图/RL模块解耦，通过标准接口传递数据
"""

import json
import subprocess
import sys
import shlex
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter

sys.path.insert(0, '/home/lkq/project')
from utils.query_cell import QueryCell
from query_planner.query_splitter import QuerySplitter, JoinQuery, QueryExecutionResult
from query_planner.tpch_workload_generator import TPC_H_WorkloadGenerator, WorkloadAdapter
from costs.cost_model import CostModel, CostConfig  # [ADDED] 导入cost_model



class ExecutorClient:
    """Executor客户端：负责与Executor节点的通信"""
    
    def __init__(self, executor_hosts: List[str], skew_name: str = "Skew0"):
        self.executor_hosts = executor_hosts
        self.timeout = 30
        self.skew_name = skew_name
         
    def execute_match(self, query_cell: QueryCell, executor_id: int) -> dict:
        """在指定Executor上执行QueryCell匹配"""
        if executor_id >= len(self.executor_hosts):
            return {"error": "Invalid executor_id", "matched": False}
        
        host = self.executor_hosts[executor_id]
        query_json = query_cell.to_json()

        # 注意：环境变量必须放在 python3 前面，而不是放在 cd 前面
        cmd = [
            "ssh",
            host,
            (
                "cd /home/lkq/project && "
                f"SKEW_NAME={shlex.quote(self.skew_name)} "
                f"PARTITION_ID={executor_id} "
                f"python3 lattice_builder/executor_main.py "
                f"--mode match --query {shlex.quote(query_json)}"
            )
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                return {
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "matched": False,
                    "fallback_to_scan": True,
                    "reason": "executor_command_failed"
                }

            out = result.stdout.strip()

            # 正常情况下 executor_main 只输出 JSON；
            # 如果前面混入少量日志，也尝试截取 JSON。
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                start = out.find("{")
                end = out.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(out[start:end + 1])

                return {
                    "error": "no_json_output",
                    "stdout": out,
                    "matched": False,
                    "fallback_to_scan": True,
                    "reason": "invalid_executor_output"
                }

        except Exception as e:
            return {
                "error": str(e),
                "matched": False,
                "fallback_to_scan": True,
                "reason": "executor_exception"
            }



class ResultCollector:
    """结果收集器：收集所有子查询的执行结果，为超图构建准备数据结构"""
    
    def __init__(self):
        self.join_results: Dict[str, List[QueryExecutionResult]] = defaultdict(list)
        self.ec_statistics: Dict[str, dict] = defaultdict(lambda: {
            "hit_count": 0,
            "queries": set(),
            "tables": set(),
            "executors": set(),
            "total_raw_count": 0,
            # [ADDED] 语义特征
            "predicate_signature": None,
            "dimension_bounds": None,
            # [ADDED] 累计代价统计
            "total_scan_rows": 0,
            "total_shuffle_bytes": 0,
            "total_execution_time_ms": 0
        })
        
    def collect(self, result: QueryExecutionResult):
        """收集结果（含语义特征提取）"""
        self.join_results[result.query_id].append(result)
        
        if result.matched and result.ec_id:
            stats = self.ec_statistics[result.ec_id]
            stats["hit_count"] += 1
            stats["queries"].add(result.query_id)
            stats["tables"].add(result.table)
            stats["executors"].add(result.executor_id)
            stats["total_raw_count"] += result.raw_count
            
            # [ADDED] 提取语义特征（只提取一次）
            if stats["predicate_signature"] is None:
                stats["predicate_signature"] = result.get_predicate_signature()
                stats["dimension_bounds"] = result.match_result.get("ec_bounds", {})
            
            # [ADDED] 累计代价
            stats["total_scan_rows"] += result.estimated_scan_rows
            stats["total_shuffle_bytes"] += result.estimated_shuffle_bytes
            stats["total_execution_time_ms"] += result.execution_time_ms or 0


    def get_hypergraph_data(self) -> dict:
        """获取超图数据（含语义特征和代价）"""
        nodes = []
        edges = []
        
        for ec_id, stats in self.ec_statistics.items():
            nodes.append({
                "ec_id": ec_id,
                "hit_frequency": stats["hit_count"],
                "tables": list(stats["tables"]),
                "executors": list(stats["executors"]),
                "total_raw_count": stats["total_raw_count"],
                "view_cost": self._estimate_view_cost(stats),
                # [ADDED] 语义特征（供RL使用）
                "predicate_signature": stats["predicate_signature"],
                "dimension_bounds": stats["dimension_bounds"],
                # [ADDED] 平均代价
                "avg_scan_rows": stats["total_scan_rows"] / max(stats["hit_count"], 1),
                "avg_shuffle_bytes": stats["total_shuffle_bytes"] / max(stats["hit_count"], 1),
                "avg_execution_time_ms": stats["total_execution_time_ms"] / max(stats["hit_count"], 1)
            })
        
        for query_id, results in self.join_results.items():
            edge_nodes = [r.ec_id for r in results if r.matched and r.ec_id]
            
            if len(edge_nodes) >= 1:
                has_fallback = any(r.match_result.get('fallback_to_scan', False) for r in results)
                edges.append({
                    "query_id": query_id,
                    "nodes": edge_nodes,
                    "tables": list(set(r.table for r in results)),
                    "full_coverage": len(edge_nodes) >= 2 and not has_fallback,
                    "requires_fallback": has_fallback
                })
        
        return {"nodes": nodes, "edges": edges}

    
    def _estimate_view_cost(self, stats: dict) -> float:
        """估算物化视图的成本（供RL奖励函数使用）"""
        storage_cost = stats["total_raw_count"] * 0.001
        maintenance_cost = len(stats["tables"]) * 10
        return storage_cost + maintenance_cost
    
    def print_statistics(self):
        print(f"\n📊 Result Collection Statistics:")
        print(f"   Total Join queries: {len(self.join_results)}")
        print(f"   Unique ECs hit: {len(self.ec_statistics)}")




class DriverController:
    """Driver端主控制器"""
    
    def __init__(self, executor_hosts: List[str] = None, num_executors: int = 2, skew_name: str = "Skew0"):
        self.skew_name = skew_name
        self.query_generator = TPC_H_WorkloadGenerator(seed=42, zipfian_skew=1.5)
        self.query_splitter = QuerySplitter()
        self.executor_client = ExecutorClient(
            executor_hosts or [f"executor{i+1}" for i in range(num_executors)],
            skew_name=skew_name
        )
        self.result_collector = ResultCollector()
        self.num_executors = num_executors

        
        # [ADDED] 集成cost_model
        self.cost_model = CostModel(CostConfig())
        self.current_materialized_views = set()
        self.total_queries_processed = 0
        
        print(f"[DriverController] Initialized with {num_executors} executors, skew={skew_name}")


    # [ADDED] RL状态接口（供train_rl_gnn.py使用）
    def get_rl_state(self) -> Dict:
        """将超图状态转换为RL State"""
        hypergraph_data = self.result_collector.get_hypergraph_data()
        
        state = {
            "node_features": [],
            "edge_features": [],
            "adjacency_info": [],
            "resource_constraints": {
                "driver_memory_mb": 6000,
                "executor_memory_mb": 3000,
                "num_executors": self.num_executors
            },
            "current_materialized_views": list(self.current_materialized_views),
            "total_queries_processed": sum(len(r) for r in self.result_collector.join_results.values())
        }
        
        for node in hypergraph_data["nodes"]:
            state["node_features"].append({
                "ec_id": node["ec_id"],
                "hit_frequency": node["hit_frequency"],
                "total_raw_count": node["total_raw_count"],
                "view_cost": node["view_cost"],
                "table": node["tables"][0] if node["tables"] else "unknown",
                "predicate_signature": node.get("predicate_signature", {}),
                "avg_scan_rows": node.get("avg_scan_rows", 0),
                "avg_shuffle_bytes": node.get("avg_shuffle_bytes", 0)
            })
        
        for edge in hypergraph_data["edges"]:
            state["edge_features"].append({
                "query_id": edge["query_id"],
                "num_nodes": len(edge["nodes"]),
                "full_coverage": edge.get("full_coverage", False),
                "requires_fallback": edge.get("requires_fallback", False)
            })
            state["adjacency_info"].append({
                "edge_id": edge["query_id"],
                "connected_nodes": edge["nodes"]
            })
        
        return state


    # [ADDED] 使用cost_model计算RL奖励
    def calculate_rl_reward(self, selected_ecs: List[str]) -> float:
        """
        计算RL奖励（调用cost_model）
        Reward = Σ(Benefit) - α·C_store
        """
        total_reward = 0.0
        
        for ec_id in selected_ecs:
            if ec_id in self.result_collector.ec_statistics:
                stats = self.result_collector.ec_statistics[ec_id]
                
                # 构造伪EC对象（用于cost_model计算）
                class PseudoEC:
                    def __init__(self, raw_count):
                        self.raw_count = raw_count
                        self.upper_bound = type('Bound', (), {'dimensions': ['*'] * 3})()
                        self.lower_bounds = []
                
                pseudo_ec = PseudoEC(stats['total_raw_count'])
                
                # 使用cost_model计算奖励
                reward = self.cost_model.calculate_reward(
                    ec=pseudo_ec,
                    hit_count=stats['hit_count']
                )
                total_reward += reward
        
        return total_reward


    # [ADDED] 评估EC（用于调试）
    def evaluate_ec(self, ec_id: str) -> Dict:
        """使用cost_model评估特定EC"""
        if ec_id not in self.result_collector.ec_statistics:
            return {}
        
        stats = self.result_collector.ec_statistics[ec_id]
        
        class PseudoEC:
            def __init__(self, raw_count):
                self.raw_count = raw_count
                self.upper_bound = type('Bound', (), {'dimensions': ['*'] * 3})()
                self.lower_bounds = []
        
        pseudo_ec = PseudoEC(stats['total_raw_count'])
        return self.cost_model.evaluate_ec(pseudo_ec, stats['hit_count'])


    def process_single_query(self, join_query: JoinQuery) -> Dict:
        """
        处理单个Join查询的完整流程
        """
        print(f"\n{'='*60}")
        print(f"Processing {join_query.query_id} (key={join_query.join_key})")
        print(f"{'='*60}")
        
        # Step 1: 拆分为子查询
        cell_l, cell_o = self.query_splitter.split_join_query(join_query)
        print(f"1️⃣  Split: L:{cell_l.cell_dimensions}, O:{cell_o.cell_dimensions}")
        
        # Step 2: 路由决策
        exec_id_l = self.query_splitter.decide_routing(cell_l, self.num_executors)
        exec_id_o = self.query_splitter.decide_routing(cell_o, self.num_executors)
        print(f"2️⃣  Route: L->Executor{exec_id_l}, O->Executor{exec_id_o}")
        
        # Step 3: 在Executor上执行匹配（传入 skew_name）
        results = []
        
        # Lineitem子查询
        if exec_id_l >= 0:
            match_res_l = self.executor_client.execute_match(cell_l, exec_id_l)
            qer_l = QueryExecutionResult(
                query_cell=cell_l,
                match_result=match_res_l,
                executor_id=f"executor_{101 + exec_id_l}"
            )
            results.append(qer_l)
            status_l = "✓" if match_res_l.get("matched") else "✗"
            print(f"3️⃣  Match L: {status_l} EC={match_res_l.get('ec_id', 'N/A')}, "
                  f"val={match_res_l.get('aggregate_value', 'N/A')}")
        else:
            print(f"3️⃣  Match L: Broadcast not implemented")
            
        # Orders子查询
        if exec_id_o >= 0:
            match_res_o = self.executor_client.execute_match(cell_o, exec_id_o)
            qer_o = QueryExecutionResult(
                query_cell=cell_o,
                match_result=match_res_o,
                executor_id=f"executor_{101 + exec_id_o}"
            )
            results.append(qer_o)
            status_o = "✓" if match_res_o.get("matched") else "✗"
            print(f"3️⃣  Match O: {status_o} EC={match_res_o.get('ec_id', 'N/A')}, "
                  f"val={match_res_o.get('aggregate_value', 'N/A')}")
        else:
            print(f"3️⃣  Match O: Broadcast not implemented")
        
        # Step 4: 收集结果
        for result in results:
            self.result_collector.collect(result)
        
        self.total_queries_processed += 1
        
        # 检查是否需要Fallback（关键：用于超图过滤）
        fallback_required = any(
            r.match_result.get('fallback_to_scan', False) 
            for r in results
        )
        
        # 判断状态
        if len(results) == 2 and all(r.matched for r in results):
            status = "SUCCESS"
        elif fallback_required:
            status = "FALLBACK_TO_SCAN"  # 明确标记需要回退到原始表
        else:
            status = "PARTIAL_MISS"
        
        print(f"4️⃣  Status: {status}")
        
        return {
            "query_id": join_query.query_id,
            "status": status,
            "fallback": fallback_required,
            "results": results
        }
    
    def process_batch(self, num_queries: int = 1000, template_mix: Dict[str, float] = None) -> Dict:
        """
        批量处理TPC-H查询（支持5模板混合）
        
        Args:
            num_queries: 查询数量
            template_mix: 模板混合比例，默认5模板混合
        """
        print(f"\n{'='*60}")
        print(f"[Batch Processing] Generating {num_queries} TPC-H queries...")
        print(f"{'='*60}")
        
        # [关键修改] 默认使用5模板混合，不再硬编码Q3
        if template_mix is None:
            template_mix = {
                'Q3': 0.40,   # 30% 标准查询
                'Q5': 0.3,   # 25% 复杂Join
                'Q10': 0.20,  # 20% 退货分析（部分覆盖场景）
                'Q16': 0.05,  # 15% 复杂过滤
                'Q17': 0.05   # 10% 小量订单
            }
        
        # 生成TPC-H查询（Zipfian分布）
        queries = self.query_generator.generate_batch(
            num_queries=num_queries,
            template_mix=template_mix
        )
        
        # [新增] 打印模板分布统计（验证5模板生效）
        from collections import Counter
        template_dist = Counter([q.template_id for q in queries])
        print(f"\n📊 模板分布:")
        for t, c in sorted(template_dist.items()):
            print(f"   {t}: {c}条 ({c/num_queries*100:.1f}%)")
        
        # 转换为JoinQuery格式
        join_queries = [WorkloadAdapter.to_join_query(q) for q in queries]
        
        # 打印Workload统计（验证Zipfian效果）
        stats = self.query_generator.get_workload_statistics(queries)
        print(f"工作负载统计:")
        print(f"  总查询数: {stats['total_queries']}")
        print(f"  唯一JoinKey数: {stats['unique_keys']}")
        print(f"  Zipfian程度: {stats['zipfian_ratio']:.2%} (Top 20% Key占比)")
        if stats['top_hot_key'][1] > 0:
            print(f"  最热Key: {stats['top_hot_key'][0]} 出现 {stats['top_hot_key'][1]} 次")
        print()
        
        # 逐个处理查询
        success_count = 0
        fallback_count = 0
        
        for query in join_queries:
            result = self.process_single_query(query)
            if result.get('status') == 'SUCCESS':
                success_count += 1
            if result.get('fallback'):
                fallback_count += 1
        
        # 打印统计
        print(f"\n{'='*60}")
        print(f"Batch Complete: {success_count}/{num_queries} successful")
        print(f"Fallback to scan: {fallback_count}/{num_queries}")
        self.result_collector.print_statistics()
        
        # 构建并返回超图数据
        hypergraph_data = self.result_collector.get_hypergraph_data()
        print(f"\n📊 Hypergraph Data Prepared:")
        print(f"   Nodes (ECs): {len(hypergraph_data['nodes'])}")
        print(f"   Edges (Joins): {len(hypergraph_data['edges'])}")
        
        return self.result_collector.get_hypergraph_data()
    
    def save_hypergraph_data(self, filepath: str = "/home/lkq/hypergraph_data.json"):
        """保存超图数据到文件（供后续RL模块加载）"""
        data = self.result_collector.get_hypergraph_data()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Hypergraph data saved to: {filepath}")


        

if __name__ == "__main__":
    # 测试运行
    controller = DriverController(
        executor_hosts=["executor1", "executor2"],
        num_executors=2
    )
    
    # 启用5模板批量处理测试
    print("测试5模板混合生成...")
    hypergraph = controller.process_batch(num_queries=100)  # 先生成100条测试
    controller.save_hypergraph_data("/home/lkq/test_hypergraph_5templates.json")

    # [ADDED] 测试cost_model集成
    if hypergraph.get('nodes'):
        sample_ec = hypergraph['nodes'][0]['ec_id']
        evaluation = controller.evaluate_ec(sample_ec)
        print(f"\n📊 Sample EC Evaluation ({sample_ec}):")
        for k, v in evaluation.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")