### 引入了三级覆盖策略
"""
超图构建器 - 与DriverController集成
"""
import networkx as nx
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import pickle


def _nested_defaultdict_int():
    return defaultdict(int)

class HypergraphBuilder:
  
    def __init__(self):
        # self.graph = nx.Graph()
        self.graph = nx.MultiGraph()
        self.join_edges: Dict[str, Dict] = {}  # join_id -> {ec_nodes, query_info, same_executor}
        self.ec_metadata: Dict[str, Dict] = {}  # ec_id -> 统计信息
        # Driver内存阈值（6GB的30%，用于判断是否能拉到Driver）
        self.driver_threshold = 0.3 * 6e9  # 1.8GB，单位与raw_count一致
        # [ADDED] 单边覆盖统计（用于节点特征编码，方案D）
        self.partial_coverage_count: Dict[str, int] = defaultdict(int)  # ec_id -> 单边查询数量
        self.partial_coverage_partners = defaultdict(_nested_defaultdict_int)  # ec_id -> {table -> count}


    def build_from_results(self, hypergraph_data: Dict):
      
        # 添加节点（等价类）
        for node_data in hypergraph_data.get('nodes', []):
            ec_id = node_data['ec_id']
            self.graph.add_node(
                ec_id,
                hit_frequency=node_data.get('hit_frequency', 0),
                tables=node_data.get('tables', []),
                executors=node_data.get('executors', []),
                raw_count=node_data.get('total_raw_count', 0),
                view_cost=node_data.get('view_cost', 0),
                table=node_data.get('tables', ['unknown'])[0] if node_data.get('tables') else 'unknown'
            )
            self.ec_metadata[ec_id] = node_data


        # 标记需要Fallback的查询（不参与超图构建）
        self.fallback_queries = set()
        
        # 添加边时过滤Fallback查询
        for edge_data in hypergraph_data.get('edges', []):
            join_id = edge_data['query_id']
            ec_nodes = edge_data.get('nodes', [])
            
            # 检查是否包含Fallback标记
            if edge_data.get('requires_fallback', False):
                self.fallback_queries.add(join_id)
                continue  # 跳过需要回退到原始表的查询
        

        # 添加边（Join查询）
        for edge_data in hypergraph_data.get('edges', []):
            join_id = edge_data['query_id']
            ec_nodes = edge_data.get('nodes', [])
            tables = edge_data.get('tables', [])
            
            # 支持单边记录，但只双边才添加图边
            if len(ec_nodes) >= 1:
                # 判断是否为同节点完全覆盖
                same_executor = False
                total_size = 0
                
                if len(ec_nodes) == 2:
                    # 获取两个EC的Executor集合
                    ec1_exec = set(self.ec_metadata.get(ec_nodes[0], {}).get('executors', []))
                    ec2_exec = set(self.ec_metadata.get(ec_nodes[1], {}).get('executors', []))
                    # 检查是否有交集（同一Executor）
                    same_executor = bool(ec1_exec & ec2_exec)
                    # 计算总数据量（用于后续阈值判断）
                    total_size = sum(
                        self.ec_metadata.get(ec, {}).get('total_raw_count', 100)
                        for ec in ec_nodes
                    )

                # 策略类型判断（适配基于连接键分区）
                # 说明：基于连接键分区时，相同key必然路由到同一Executor，因此只要双边命中就是同节点
                if len(ec_nodes) == 1:
                    strategy_type = 'partial_coverage'
                    # [ADDED] 记录单边信息（方案D：编码为节点特征）
                    ec_id = ec_nodes[0]
                    self.partial_coverage_count[ec_id] += 1
                
                    # 记录可能配对的表（从 tables 中推断缺失的表）
                    if tables:
                        current_table = tables[0] if isinstance(tables[0], str) else 'unknown'
                        # 推断缺失的表（简化：假设是 Lineitem 或 Orders）
                        if 'lineitem' in current_table.lower():
                            missing_table = 'orders'
                        elif 'orders' in current_table.lower():
                            missing_table = 'lineitem'
                        else:
                            missing_table = 'unknown'
                        
                        self.partial_coverage_partners[ec_id][missing_table] += 1
                            
                elif len(ec_nodes) == 2:
                    # 基于连接键分区：双边命中必然同节点 -> local_join
                    strategy_type = 'local_join'

                # 记录超边信息（包含同节点标记和数据量）
                self.join_edges[join_id] = {
                    'nodes': set(ec_nodes),
                    'tables': tables,
                    'full_coverage': len(ec_nodes) >= 2,
                    'same_executor': same_executor,  # 是否同节点
                    'total_size': total_size,        # 总数据量
                    'strategy_type': strategy_type,      # 策略类型标记（关键）
                    'requires_fallback': edge_data.get('requires_fallback', False)
                }
                
                # [MODIFIED] 只在图中添加双边（删除单边自环）
                if len(ec_nodes) == 2:
                    ec1, ec2 = ec_nodes[0], ec_nodes[1]
                    if self.graph.has_node(ec1) and self.graph.has_node(ec2):
                        self.graph.add_edge(
                            ec1, ec2, 
                            join_id=join_id,
                            weight=1
                        )
        
        # 统计信息更新，区分当前模式和备用模式
        strategy_counts = {}
        for join_info in self.join_edges.values():
            st = join_info.get('strategy_type', 'unknown')
            strategy_counts[st] = strategy_counts.get(st, 0) + 1

        # print(f"[HypergraphBuilder] 构建完成: {self.graph.number_of_nodes()} 节点, "
        #       f"{len(self.join_edges)} 超边, "
        #       f"{self.graph.number_of_edges()} 图边(含多重)")
        bilateral_count = sum(1 for j in self.join_edges.values() if len(j['nodes']) == 2)
        print(f"[HypergraphBuilder] 构建完成: {self.graph.number_of_nodes()} 节点, "
            f"{bilateral_count} 双边超边(供GNN), "
            f"{len(self.join_edges) - bilateral_count} 单边查询(作节点特征)")

        # 打印单边统计信息（方案D验证）
        if self.partial_coverage_count:
            total_partial = sum(self.partial_coverage_count.values())
            print(f"[HypergraphBuilder] 单边覆盖统计: {len(self.partial_coverage_count)} 个EC有单边记录, "
                  f"总计 {total_partial} 次单边查询")

        # 打印策略分布（基于连接键分区时应该只有 local_join 和 partial_coverage）
        print(f"[HypergraphBuilder] 策略类型分布 (基于连接键分区):")
        for st, count in sorted(strategy_counts.items()):
            percentage = count / len(self.join_edges) * 100 if self.join_edges else 0
            print(f"  - {st}: {count} ({percentage:.1f}%)")
        
        # 提示：如果看到 driver_join 或 distributed_join，说明分区方式可能不是基于连接键
        if 'driver_join' in strategy_counts or 'distributed_join' in strategy_counts:
            print(f"[警告] 检测到跨节点策略，请检查是否使用了非连接键分区方法")

       
       
        # 关键统计：双边查询占比（用于诊断完全覆盖问题）
        total_edges = len(self.join_edges)
        bilateral_edges = sum(1 for j in self.join_edges.values() if len(j['nodes']) == 2)
        unilateral_edges = total_edges - bilateral_edges
        
        if total_edges > 0:
            print(f"[HypergraphBuilder] 超边结构分析:")
            # print(f"  - 总查询数: {total_edges}")
            # print(f"  - 双边查询(可完全覆盖): {bilateral_edges} ({bilateral_edges/total_edges:.1%})")
            # print(f"  - 单边查询(部分覆盖): {unilateral_edges} ({unilateral_edges/total_edges:.1%})")
            print(f"  - 双边超边数(GNN嵌入用): {bilateral_edges} ({bilateral_edges/total_edges:.1%})")  
            print(f"  - 单边查询数(节点特征): {unilateral_edges} ({unilateral_edges/total_edges:.1%})")   
            print(f"  - 总查询数: {total_edges}")                     
            print(f"  - 双边覆盖率: {bilateral_edges/total_edges:.1%}")
            
            if bilateral_edges == 0:
                print(f"[警告] 没有双边查询！完全覆盖率必然为0，请检查数据生成或EC划分逻辑")
            elif bilateral_edges / total_edges < 0.2:
                print(f"[警告] 双边查询占比过低(<20%)，完全覆盖很难实现")



    # 计算EC的分布式价值
    def get_ec_distributed_value(self, ec_id: str) -> Dict:
        """
        计算EC的分布式价值：
        - 本地查询频率（EC所在Executor发起的查询）
        - 远程查询频率（其他Executor发起的查询，物化后可节省Shuffle）
        """
        ec_data = self.ec_metadata.get(ec_id, {})
        ec_executors = set(ec_data.get('executors', []))
        
        local_hits = 0
        remote_hits = 0
        
        # 遍历所有包含该EC的Join查询
        for join_id, join_info in self.join_edges.items():
            if ec_id in join_info['nodes']:
                # 简化：假设查询均匀分布，或者从join_info中获取
                # 实际应该从查询日志中统计
                query_executors = join_info.get('source_executors', ec_executors)
                for exe in query_executors:
                    if exe in ec_executors:
                        local_hits += 1
                    else:
                        remote_hits += 1
        
        return {
            'local_hits': local_hits,
            'remote_hits': remote_hits,  # 物化后可节省的Shuffle次数
            'shuffle_saving_potential': remote_hits * ec_data.get('size', 0)  # 节省的总数据量
        }



    def get_ec_candidates(self) -> List[str]:
        """获取候选视图（所有节点）"""
        return list(self.graph.nodes())
    
    def get_ec_frequency(self, ec_id: str) -> int:
        """获取EC被查询命中的频率"""
        return self.graph.nodes[ec_id].get('hit_frequency', 0)
    
    def get_join_coverage(self, selected_ecs: Set[str]) -> Dict:
        """
        计算物化视图集合覆盖的Join查询
        
        Returns:
            {
                'covered_joins': [...],  # 完全覆盖的Join（两边EC都被物化）
                'partial_joins': [...],  # 部分覆盖（只物化一边）
                'coverage_ratio': 0.8     # 覆盖率
            }
        """

        # 确保 selected_ecs 是 set 类型（兼容 list 输入）
        if isinstance(selected_ecs, list):
            selected_ecs = set(selected_ecs)
        elif not isinstance(selected_ecs, set):
            # 如果是其他类型（如 dict_keys），也转换为 set
            selected_ecs = set(selected_ecs)

        covered = []
        partial = []
        
        for join_id, join_info in self.join_edges.items():
            join_nodes = join_info['nodes']
            intersection = join_nodes & selected_ecs
            
            if len(intersection) == len(join_nodes) and len(join_nodes) >= 2:
                covered.append(join_id)
            elif len(intersection) > 0:
                partial.append(join_id)
        
        total = len(self.join_edges)
        return {
            'covered_joins': covered,
            'partial_joins': partial,
            'coverage_ratio': len(covered) / total if total > 0 else 0,
            'total_joins': total
        }
    
    def get_table_distribution(self, selected_ecs: Set[str]) -> Dict[str, List[str]]:
        """按表分类物化视图"""
        result = defaultdict(list)
        for ec_id in selected_ecs:
            if self.graph.has_node(ec_id):
                table = self.graph.nodes[ec_id].get('table', 'unknown')
                result[table].append(ec_id)
        return dict(result)
    
    def evaluate_view_set(self, selected_ecs: Set[str], cost_model) -> Dict:
        """
        评估物化视图集合的综合性能
        
        Returns:
            {
                'total_storage_cost': float,
                'total_benefit': float,
                'net_utility': float,
                'coverage_ratio': float,
                'avg_query_cost': float
            }
        """
        total_storage = 0
        total_benefit = 0
        
        for ec_id in selected_ecs:
            # 这里需要从原始EC对象获取信息，简化起见使用metadata
            raw_count = self.ec_metadata.get(ec_id, {}).get('total_raw_count', 100)
            hit_freq = self.get_ec_frequency(ec_id)
            
            # 构建临时EC对象用于成本计算
            class TempEC:
                def __init__(self, raw_count):
                    self.raw_count = raw_count
                    self.upper_bound = type('obj', (object,), {'dimensions': [0,0,0]})()
                    self.lower_bounds = []
            
            temp_ec = TempEC(raw_count)
            total_storage += cost_model.c_store(temp_ec)
            total_benefit += cost_model.benefit(temp_ec) * hit_freq
        
        coverage = self.get_join_coverage(selected_ecs)
        
        return {
            'total_storage_cost': total_storage,
            'total_benefit': total_benefit,
            'net_utility': total_benefit - 0.1 * total_storage,  # alpha=0.1
            'coverage_ratio': coverage['coverage_ratio'],
            'covered_joins': len(coverage['covered_joins'])
        }







