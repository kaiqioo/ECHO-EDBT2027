"""
图强化学习环境 - 基于PyTorch Geometric
完全兼容现有HypergraphBuilder和CostModel
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Set, Dict, List
from collections import deque
from online_phase.view_pool_manager import estimate_view_size_bytes


class GraphViewSelectionEnv(gym.Env):
    """
    图强化学习环境
    
    观测空间（Dict）：
    - node_features: [num_candidates, 6] - ROI, freq, size, selected_flag, partial_ratio, top_partner
    - edge_index: [2, num_edges] - 超边连接（Join查询，仅双边）
    - edge_attr: [num_edges, 1] - 边权重（查询频率）
    - global_features: [2] - 存储压力、剩余预算
    - action_mask: [num_candidates] - 有效动作掩码（0=无效，1=有效）
    节点特征从4维扩展到6维，包含单边配对潜力信息（方案D）
    """
    
    def __init__(self, 
                 hypergraph_builder, 
                 cost_model,
                 candidates: List[str] = None,
                 storage_capacity: float = 100.0,
                 alpha: float = 0.1,
                 device: str = 'cpu'):
        super().__init__()
        
        self.hg = hypergraph_builder
        self.cost_model = cost_model
        self.capacity = storage_capacity
        self.alpha = alpha
        self.device = device
        
        # 如果没有提供candidates，使用所有EC
        if candidates is None:
            candidates = list(self.hg.ec_metadata.keys())
        self.candidates = candidates
        self.num_candidates = len(candidates)
        
        # 构建静态图结构（仅双边，单边信息编码在节点特征中）
        self._build_graph_structure()
        
        # 动作空间：选择候选索引
        self.action_space = spaces.Discrete(self.num_candidates)
        
        # 观测空间：节点特征从4维扩展到6维（方案D）
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(
                low=0, high=1, 
                shape=(self.num_candidates, 6),  # [FIXED] 4 -> 6，增加partial_ratio和top_partner
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0, high=self.num_candidates,
                shape=(2, len(self.edge_list)),
                dtype=np.int64
            ),
            'edge_attr': spaces.Box(
                low=0, high=10,
                shape=(len(self.edge_list), 1),
                dtype=np.float32
            ),
            'global_features': spaces.Box(
                low=0, high=1, 
                shape=(2,), 
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1, 
                shape=(self.num_candidates,), 
                dtype=np.float32
            )
        })
        
        # 预计算EC属性（包含单边信息）
        self.ec_properties = self._precompute_ec_properties()
        
        self.reset()
    
    def _build_graph_structure(self):
        """构建静态图拓扑（边和邻接关系）
        [MODIFIED] 方案D：只添加双边到图结构，单边信息编码在节点特征中
        """
        self.edge_list = []  # [(src, dst), ...]
        self.edge_weights = []  # 对应查询频率
        self.join_edge_map = {}  # edge_key -> join_info，用于快速查询
        
        # 构建EC到索引的映射
        self.ec_to_idx = {ec: i for i, ec in enumerate(self.candidates)}
        
        # 遍历所有Join边构建图（仅双边，过滤单边）
        for join_id, join_info in self.hg.join_edges.items():
            nodes = join_info['nodes']
            # [MODIFIED] 方案D：只处理双边，单边不进图结构（但保留在节点特征中）
            if len(nodes) != 2:
                continue
                
            ec1, ec2 = list(nodes)
            # 只保留两个EC都在候选中的边
            if ec1 not in self.ec_to_idx or ec2 not in self.ec_to_idx:
                continue
                
            idx1 = self.ec_to_idx[ec1]
            idx2 = self.ec_to_idx[ec2]
            
            # 双向边（无向图）
            self.edge_list.append([idx1, idx2])
            self.edge_list.append([idx2, idx1])
            
            # 边权重：使用Join查询的频率（从metadata估算）
            weight = 1.0  # 默认
            # 尝试从EC的hit_frequency估算
            freq1 = self.hg.ec_metadata.get(ec1, {}).get('hit_frequency', 0)
            freq2 = self.hg.ec_metadata.get(ec2, {}).get('hit_frequency', 0)
            weight = min((freq1 + freq2) / 2.0, 10.0)  # 归一化到10以内
            
            self.edge_weights.extend([weight, weight])
            
            # 记录Join边信息
            edge_key = tuple(sorted([idx1, idx2]))
            self.join_edge_map[edge_key] = join_info
        
        # 转换为numpy
        self.edge_index = np.array(self.edge_list, dtype=np.int64).T  # [2, num_edges]
        self.edge_attr = np.array(self.edge_weights, dtype=np.float32).reshape(-1, 1)
        
        # 构建邻接表（用于快速查询邻居）
        self.adj_list = [set() for _ in range(self.num_candidates)]
        for src, dst in self.edge_list:
            self.adj_list[src].add(dst)
        
        print(f"[GraphEnv] 图构建完成: {self.num_candidates} 节点, {len(self.edge_list)//2} 无向边 "
              f"(已过滤单边查询，单边信息编码在节点特征中)")
    
    def _precompute_ec_properties(self):

        props = {}

        max_partial_count = max(self.hg.partial_coverage_count.values(), default=1)
        if max_partial_count == 0:
            max_partial_count = 1

        class Bound:
            def __init__(self, dims):
                self.dimensions = dims

        class CompactEC:
            def __init__(self, upper_dims, lower_dims_list, raw_count):
                self.raw_count = raw_count
                self.upper_bound = Bound(upper_dims)
                self.lower_bounds = [Bound(dims) for dims in lower_dims_list]

        for i, ec_id in enumerate(self.candidates):
            data = self.hg.ec_metadata.get(ec_id, {})
            raw_count = data.get("total_raw_count", 100)
            hit_freq = data.get("hit_frequency", 0)

            bounds = data.get("dimension_bounds", {})
            upper_dims = bounds.get("upper_bound", [0, 0, 0])
            lower_dims_list = bounds.get("lower_bounds", [upper_dims])

            compact_ec = CompactEC(upper_dims, lower_dims_list, raw_count)
            c_store = float(estimate_view_size_bytes(compact_ec))

            partial_count = self.hg.partial_coverage_count.get(ec_id, 0)
            partners = self.hg.partial_coverage_partners.get(ec_id, {})
            top_partner = max(partners, key=partners.get) if partners else "unknown"

            # 这里的 benefit 只用于节点排序和 ROI 特征，不改变真实在线评估逻辑
            view_cost = data.get("view_cost", 1.0)
            benefit = view_cost * max(hit_freq, 1)

            props[i] = {
                "ec_id": ec_id,
                "size": c_store,
                "raw_count": raw_count,
                "hit_freq": hit_freq,
                "is_lineitem": "lineitem" in ec_id,
                "is_orders": "orders" in ec_id,
                "roi": benefit / max(c_store / 1024.0, 1.0),
                "partial_count": partial_count,
                "partial_ratio": partial_count / max_partial_count,
                "top_partner": top_partner
            }

        return props

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.selected = set()  # 已选EC索引
        self.storage_usage = 0.0
        self.covered_join_edges = set()  # 已完全覆盖的Join边（edge_key）
        self.step_count = 0
        
        return self._get_obs(), {}
    
    
    def action_masks(self):
        """
        返回动作掩码（用于MaskablePPO）
        返回: np.ndarray, shape=(num_candidates,), dtype=bool
              True=动作有效, False=动作无效（已选或超预算）
        """
        mask = np.ones(self.num_candidates, dtype=bool)
        
        for i in range(self.num_candidates):
            if i in self.selected:
                mask[i] = False
            elif self.storage_usage + self.ec_properties[i]['size'] > self.capacity:
                mask[i] = False
        
        return mask
    

    def _get_obs(self):

        # [MODIFIED] 节点特征：[ROI, freq, normalized_size, selected_flag, partial_ratio, top_partner]
        node_features = np.zeros((self.num_candidates, 6), dtype=np.float32)
        
        for i in range(self.num_candidates):
            prop = self.ec_properties[i]
            node_features[i, 0] = min(prop['roi'] / 10.0, 1.0)  # ROI
            node_features[i, 1] = min(prop['hit_freq'] / 10.0, 1.0)  # freq
            node_features[i, 2] = min(prop['size'] / max(self.capacity, 1.0), 1.0)  # size
            node_features[i, 3] = 1.0 if i in self.selected else 0.0  # selected flag
            # [ADDED] 方案D：单边特征编码
            node_features[i, 4] = prop['partial_ratio']  # 单边查询比例（配对潜力信号）
            # top_partner编码：lineitem=1.0, orders=0.5, unknown/other=0.0
            node_features[i, 5] = 1.0 if prop['top_partner'] == 'lineitem' else (
                0.5 if prop['top_partner'] == 'orders' else 0.0
            )


        # 全局特征
        storage_pressure = min(self.storage_usage / self.capacity, 1.0)
        global_feat = np.array([
            storage_pressure,
            1.0 - storage_pressure
        ], dtype=np.float32)
        
        # 动作掩码：已选或超预算的EC不能选
        action_mask = np.ones(self.num_candidates, dtype=np.float32)
        for i in range(self.num_candidates):
            if i in self.selected:
                action_mask[i] = 0.0
            elif self.storage_usage + self.ec_properties[i]['size'] > self.capacity:
                action_mask[i] = 0.0
        
        return {
            'node_features': node_features,
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'global_features': global_feat,
            'action_mask': action_mask
        }
    
    def step(self, action):
        """
        执行动作
        奖励 = 边际完全覆盖*25 + 双表平衡奖励 - 存储成本 - 孤立惩罚
        [NOTE] 奖励仍只基于双边（完全覆盖），但GNN通过节点特征学习单边配对潜力
        """
        self.step_count += 1
        terminated = False
        truncated = False
        info = {}
        
        # 检查动作有效性
        if action < 0 or action >= self.num_candidates:
            reward = -100.0
            info['error'] = 'invalid_action'
        elif action in self.selected:
            reward = -10.0
            info['error'] = 'already_selected'
        elif self.storage_usage + self.ec_properties[action]['size'] > self.capacity:
            reward = -10.0
            info['error'] = 'storage_overflow'
        else:
            # 执行选择
            current_idx = action   # 选择单个节点
            current_ec_id = self.candidates[current_idx]
            self.selected.add(current_idx)    # 加入已选集合
            self.storage_usage += self.ec_properties[current_idx]['size']
            
            # 计算奖励组件
            reward_components = {}
            
            # 1. 基础存储成本（负）
            base_cost = -self.alpha * (self.ec_properties[current_idx]['size'] / (1024 * 1024))
            
            # 2. [核心] 边际完全覆盖奖励（只基于双边，与方案D一致）
            # 检查这个EC与已选EC形成了多少新的完全覆盖
            new_covers = 0
            connected_selected = []
            
            for neighbor_idx in self.adj_list[current_idx]:
                if neighbor_idx in self.selected and neighbor_idx != current_idx:
                    # 检查这条Join边是否已经被覆盖
                    edge_key = tuple(sorted([current_idx, neighbor_idx]))
                    
                    if edge_key not in self.covered_join_edges:
                        # 新的完全覆盖！
                        new_covers += 1
                        self.covered_join_edges.add(edge_key)
                        connected_selected.append(neighbor_idx)
            
            cover_reward = new_covers * 100.0  # 每个新完全覆盖+25
            
            # 3. 双表平衡奖励（首次形成配对时）
            diversity_reward = 0.0
            prop = self.ec_properties[current_idx]
            
            if new_covers > 0:
                # 形成了新的完全覆盖，给予额外奖励
                diversity_reward = 50.0
                info['formed_pairs'] = new_covers
            
            # 4. 孤立选择惩罚（如果选了但没形成任何连接）
            isolation_penalty = 0.0
            if new_covers == 0 and len(self.selected) > 1:
                # 检查是否与任何已选EC有潜在连接（即使没形成完全覆盖）
                has_connection = any(
                    neighbor in self.selected 
                    for neighbor in self.adj_list[current_idx]
                )
                if not has_connection:
                    isolation_penalty = -20.0  # 孤立选择惩罚
                    info['isolation_warning'] = True
            
            # 5. 本地传输收益（原有逻辑简化）
            hit_freq = prop['hit_freq']
            local_bonus = hit_freq * 0.1  # 简化本地收益
            
            total_reward = base_cost + cover_reward + diversity_reward + isolation_penalty + local_bonus
            
            reward = total_reward
            info['action'] = 'materialized'
            info['ec_id'] = current_ec_id
            info['new_covers'] = new_covers
            info['total_covered'] = len(self.covered_join_edges)
            info['storage_used'] = self.storage_usage
            info['num_selected'] = len(self.selected)

            # 记录奖励组件用于调试
            info['reward_breakdown'] = {
                'base_cost': base_cost,
                'cover_reward': cover_reward,
                'diversity_reward': diversity_reward,
                'isolation_penalty': isolation_penalty,
                'local_bonus': local_bonus
            }
        
        # 检查终止条件
        # 1. 存储已满（无法选择更多）
        remaining_budget = self.capacity - self.storage_usage
        min_ec_size = min(p['size'] for p in self.ec_properties.values())
        
        if remaining_budget < min_ec_size * 0.5:  # 剩余空间不足以选最小EC
            terminated = True
        # 2. 所有候选已选
        elif len(self.selected) >= self.num_candidates:
            terminated = True
        # 3. 步数限制（安全机制）
        elif self.step_count >= self.num_candidates * 2:
            truncated = True
        
        # 终止时统计最终覆盖（使用与验证代码相同的逻辑）
        if terminated or truncated:
            local_full = 0
            partial_count = 0
            
            # 使用与hypergraph_builder相同的统计逻辑
            selected_ec_ids = {self.candidates[i] for i in self.selected}
            
            for join_id, join_info in self.hg.join_edges.items():
                nodes = join_info['nodes']
                intersection = nodes & selected_ec_ids
                
                if len(intersection) == len(nodes) and len(nodes) == 2:
                    local_full += 1
                elif len(intersection) > 0:
                    partial_count += 1
            
            info['final_local_full'] = local_full
            info['final_partial'] = partial_count
            info['final_coverage'] = local_full / max(len(self.hg.join_edges), 1)
            info['num_selected'] = len(self.selected)
            info['lineitem_count'] = sum(1 for i in self.selected if 'lineitem' in self.candidates[i])
            info['orders_count'] = sum(1 for i in self.selected if 'orders' in self.candidates[i])
            
            # 最终奖励加成：基于完全覆盖率
            coverage_bonus = local_full * 50.0  # 终止时大奖励
            reward += coverage_bonus
            info['coverage_bonus'] = coverage_bonus
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def get_selection_result(self) -> Dict:
        """获取最终选择结果（兼容原有接口）"""
        selected_ids = [self.candidates[i] for i in self.selected]
        return {
            'selected': selected_ids,
            'storage_usage': self.storage_usage,
            'storage_capacity': self.capacity,
            'utilization': self.storage_usage / self.capacity,
            'coverage': self.hg.get_join_coverage(set(selected_ids))
        }