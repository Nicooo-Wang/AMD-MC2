#!/usr/bin/env python3
"""
单文件运行脚本 - 测试 AllToAll Dispatch 功能
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import dataclasses
import time
from typing import List

# ---------------- 配置和数据定义 ----------------
@dataclasses.dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16

class RankTestData:
    def __init__(self, cfg: MoEConfig, rng: torch.Generator, rank: int):
        device = torch.device(f"cuda:{rank}")
        self.num_tokens = int(
            torch.randint(
                1, cfg.max_num_tokens, [1], generator=rng, device=device
            ).item()
        )
        # token expert 映射
        self.indices = torch.empty(
            self.num_tokens, cfg.experts_per_token, dtype=torch.int32, device=device
        )
        for i in range(self.num_tokens):
            perm = torch.randperm(cfg.num_experts, generator=rng, device=device)
            self.indices[i] = perm[: cfg.experts_per_token]
        # topk 权重
        self.weights = torch.rand(
            self.num_tokens,
            cfg.experts_per_token,
            dtype=torch.float32,
            generator=rng,
            device=device,
        )
        # dp tokens, dispatch 的输入
        self.x = torch.randn(
            self.num_tokens,
            cfg.hidden_dim,
            dtype=cfg.in_dtype,
            generator=rng,
            device=device,
        )

# ---------------- AllToAll 实现 ----------------
class PyTorchAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    def __init__(self, cfg: MoEConfig, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # 1. 计算发送计数
        send_counts = [0] * self.world_size
        token_map = [[] for _ in range(self.world_size)]
        meta_map = [[] for _ in range(self.world_size)]
        
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                dst_rank = e // self.num_local_experts
                send_counts[dst_rank] += 1
                token_map[dst_rank].append(t)
                meta_map[dst_rank].extend([e, self.rank, t, k, 0])

        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        
        # 第一次 AllToAll - 交换计数
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        
        # 2. 准备发送缓冲区
        send_buf = torch.cat([dp_x[idx_list] for idx_list in token_map], dim=0)
        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device
        )

        # 准备元数据
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )
        
        # 第二次 AllToAll - 交换数据
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )

        # 第三次 AllToAll - 交换元数据
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)
        
        # 3. 组织到专家缓冲区
        expert_num_tokens = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device=device
        )
        expert_x = torch.empty(
            (self.num_local_experts, self.max_recv, cfg.hidden_dim),
            dtype=cfg.in_dtype,
            device=device,
        )
        expert_meta = torch.empty(
            (self.num_local_experts, self.max_recv, self.META_DIM),
            dtype=torch.int32,
            device=device,
        )
        
        for i in range(total_recv):
            global_eid = int(recv_meta[i, 0].item())
            local_eid = global_eid % self.num_local_experts
            expert_x[local_eid, expert_num_tokens[local_eid]] = recv_buf[i]
            expert_meta[local_eid, expert_num_tokens[local_eid]] = recv_meta[i]
            expert_num_tokens[local_eid] += 1

        return expert_num_tokens, expert_x, expert_meta

# ---------------- 分布式设置和运行函数 ----------------
def setup_distributed(rank: int, world_size: int):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: 分布式环境初始化完成")

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def run_dispatch_test(rank: int, world_size: int, cfg_params: dict):
    """在每个 rank 上运行 dispatch 测试"""
    try:
        setup_distributed(rank, world_size)
        
        # 创建配置
        cfg = MoEConfig(**cfg_params)
        
        # 生成测试数据
        device = torch.device(f"cuda:{rank}")
        rng = torch.Generator(device=device)
        rng.manual_seed(1234 + rank)
        rank_data = RankTestData(cfg, rng, rank)
        
        # 创建 AllToAll 实例
        ata = PyTorchAllToAll(cfg, rank, world_size)
        
        # 同步所有 rank
        dist.barrier()
        
        # 运行 dispatch
        start_time = time.time()
        expert_num_tokens, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)
        torch.cuda.synchronize()  # 等待 GPU 操作完成
        end_time = time.time()
        
        # 打印结果
        print(f"Rank {rank}:")
        print(f"  - 输入 tokens: {rank_data.num_tokens}")
        print(f"  - 专家数量: {cfg.num_experts}")
        print(f"  - 每个 rank 的专家数: {ata.num_local_experts}")
        print(f"  - 每个专家接收的 tokens: {expert_num_tokens.tolist()}")
        print(f"  - expert_x 形状: {expert_x.shape}")
        print(f"  - expert_meta 形状: {expert_meta.shape}")
        print(f"  - Dispatch 耗时: {(end_time - start_time) * 1000:.2f} ms")
        
        # 验证结果
        total_received = expert_num_tokens.sum().item()
        print(f"  - 总共接收 tokens: {total_received}")
        
        # 简单验证：检查是否所有专家都收到了正确的数据
        for local_eid in range(ata.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())
            if cnt > 0:
                # 检查元数据中的专家ID是否正确
                for j in range(min(3, cnt)):  # 只检查前3个
                    meta = expert_meta[local_eid, j]
                    global_eid = meta[0].item()
                    expected_local = global_eid % ata.num_local_experts
                    assert expected_local == local_eid, f"专家ID不匹配: {expected_local} != {local_eid}"
        
        print(f"Rank {rank}: 测试通过!")
        
    except Exception as e:
        print(f"Rank {rank}: 错误 - {e}")
        raise
    finally:
        cleanup_distributed()

def main():
    """主函数"""
    # 测试配置
    cfg_params = {
        'num_experts': 8,
        'experts_per_token': 2,
        'hidden_dim': 6144,
        'max_num_tokens': 16
    }
    
    world_size = 2  # 使用 2 个进程进行测试
    
    print("=" * 60)
    print("AllToAll Dispatch 测试")
    print("=" * 60)
    print(f"配置参数: {cfg_params}")
    print(f"使用 {world_size} 个进程")
    print()
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持")
        return
    
    if torch.cuda.device_count() < world_size:
        print(f"警告: 只有 {torch.cuda.device_count()} 个 GPU，但需要 {world_size} 个")
        print("将尝试使用现有 GPU...")
        world_size = min(world_size, torch.cuda.device_count())
    
    # 使用多进程启动测试
    mp.set_start_method('spawn', force=True)
    
    processes = []
    try:
        for rank in range(world_size):
            p = mp.Process(
                target=run_dispatch_test,
                args=(rank, world_size, cfg_params)
            )
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"进程 {p.pid} 异常退出，代码: {p.exitcode}")
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("测试被用户中断")
        for p in processes:
            p.terminate()
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
