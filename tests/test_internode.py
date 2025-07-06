import argparse
import os
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
import test_low_latency


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int,
              local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    
    """
        refer: https://zhuanlan.zhihu.com/p/1890067712996270654
        num_sms(20)
        num_ranks(64) world_size
        num_nodes(8) 总结点数
        local_rank 当前Node上 gpu/rank idx
        rank = node*8 + local_rank， global rank idx 
    """

    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    """
        num_tokens 4096
        hidden 7168
    """
    num_topk_groups, num_topk, num_experts = args.num_topk_groups, args.num_topk, args.num_experts
    """
        num_topk_groups(4), per token发给4台机器， Deepseek V3技术文档中提到为了减小通信，token只会发送到最多4台机器上
        num_topk(8) 发送给4台机器中的top 8 专家
        num_experts(256) 
    """

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1) 
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    """
        scores  [ntokens, nexperts] 
        group_scores [ntokens, nnodes, nexperts//nnodes]， per node 上最大score  --> [ntokens, nnodes]
        group_idx， 从上述group_scores[ntokens, nnodes] nnodes 维度 选 num_topk_groups(4) -> [ntokens, num_token_groups]，即 per token 将router 去的 top4 节点

    """
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    """
        不在top4 group 中 score 被mask为0  
        masked_scores[ntokens, nexperts]，只是有些值是0
    """
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    """
        masked_scores[ntokens, nexperts] 在 nexperts 维度上取 top8 experts_idx， [ntokens, num_topk] 
        torch.topk() returns tuple (values, indices) --> topk_idx[ntokens, top8_expert_idx]
    """
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    """
        * num_experts//num_ranks : per rank 上 experts 数 (4)
        * topk_idx // 4， 即当前 expert_id 落在哪个 rank 上，亦即其 rank_idx
        * rank_idx [ntokens, num_ranks], e.g. ith token 的 top8_expert_idx[j] 落在哪个gpu 上[0, 64]
    """
    inplace_unique(rank_idx, num_ranks)
    """
        如果某个显卡有多个expert, 则rank会重复，inplace_unique是去重了rank, 剩余位置补 -1
    """
    rdma_rank_idx = rank_idx // num_local_ranks
    """
        rank 每8个一组为rdma_rank group。
        这里判断当前 rank_idx 落在哪个 rdma_rank_idx 
        rdma_rank_idx [ntokens, 8]，描述当前token 的 top8 experts 所在 rdma分组内的 idx
    """
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)
    """
        如果 top8 expert_idx 对应同一个node上的多个显卡上，则rdma_rank 会重复，inplace_unique去重,剩余位置补 -1
    """
    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    """
        * num_experts//num_nodes : per node 上 experts 数 (32)
        topk_idx[ntokens, top8_expert_ids] // 32 :: rdma_idx[ntokens, 8]
        即 当前 token 的 top8_experts 所在 rdma分组的idx

        rdma_idx 与 rdma_rank_idx 组成2D 的网格。
            * rdma_idx 与 node_id 对应
            * rdma_rank_idx 与 node 内 gpu_idx 对应
    """
    inplace_unique(rdma_idx, num_nodes)
    """
        如果落在同一个node上，去重。这里是粗粒度的，并没有关心node内具体哪个gpu上
    """
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()
    """
        rdma_idx[ntokens, 8]
        .ne(-1) : creates a boolean mask of same shape [ntokens, 8], any value!= -1 set as true
        .sum(): sum all true values and produce a scale (0-dim tensor)
        .item(): convert 0-dim tensor to integer 
    * num_rdma_token_sent，即在 rdma_level(节点层面)需要发送的tokens 总数
    """

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)
    """
        topk_idx[ntokens, top8_expert_idx] 统计各个expert_id 出现的次数，即 ntokens_per_expert
        `num_tokens_per_expert[i]` on any rank counts __local_token_selections__ for expert i
        所以，system-level 需要 allreduce 跟新全局 num_tokens_per_expert[i]
    """

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')
    num_tokens_per_rdma_rank = torch.empty((num_nodes, ), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    """
        * num_tokens_per_rank，即 per GPU 上 tokens 数
        * num_tokens_per_rdma_rank，即 per node 上 tokens 数
        token_idx_in_rank [num_ranks, num_tokens] 2D matrix，每行表示一个gpu 上 tokens idx
    """

    # token selection process 
    for i in range(num_ranks):
        """
            这个loop 是 per-rank(GPU) 上的执行，即 rank 视角，to builds a dispatch layout matrix
            每个rank上都维护相同的全局 token_idx_in_rank， 但per rank 仅仅负责填充其所拥有experts 要处理的 token_ids
            token_sel 是每个 rank_idx(gpu) 维护的一个本地数组，即每个gpu上都有这样一个数组于用于表示 token_ith 是否需要向该gpu 发送数据
        """
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        """
        * token_sel as temporary tensor created/destroyed each iteration， only persist during this iteration
            rank_idx[ntokens, 64]， 当前token 的top8 experts 所在 64x GPU 上的位置
            (rank_idx==i) :  [ntokens, 64] boolean tensor ，
            .max(dim=-1), max reduction along 64-dim : [ntokens, ][0] -> [ntokens]
            token_sel[ntokens]
        """
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        """
            * count: number of tokens with expert located in current rank(gpu)
            * .sort() 返回 (sorted_vals, sorted_idx), sorted in 降序，先大后小
                * 即将 token_sel 数组中 val=1 的值前置
            * [1] 返回 sorted_idx 维度 存入 tokens 数组
            * sort(tokens[:count])[0]， 对tokens中val=1 的 token_idx 做升序排列，且只返回 排序后的sorted_vals，即这里的 token_idx
        * 效果: 注意仍在 rank loop 中，tokens 存的是将发送到当前rank 的 token_idx，且由小到大排列
        """
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
        """
            * token_idx_in_rank [num_ranks, num_tokens]
            * tokens 数组 前 count 个元素，存的是将要发送给当前 gpu/rank_i 的 token_idx.  
            * 对于 token_idx_in_rank[i] 中这些token_idx 位置赋上 0, 1, 2, .. 
        * TODO: 
        """
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    """
        rdma_rank_idx[ntokens, 8] 在 ntokens 维度上 sum后，即当前rdma_rank node上的 tokens 数量
        * TODO: rdma_rank_idx vs rdma_idx ? 貌似应该是 rdma_idx ??
    """
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    """
        token_idx_in_rank.T ->   [num_ranks, num_tokens] ->  [num_tokens, num_ranks]，即从 gpu/rank视角 变成 以token 视角
        token_idx_in_rank 做 element-wise >0 ，即 判断在 token为中心的视角，该token是否在某个rank  
    """
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)
    """
        * num_tokens_per_rank[num_ranks, ]
        same as num_tokens_per_expert，per rank 上只有 local num_tokens_per_rank[i], 需要 allreduce sync-up
    """

    ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    """
        
    """
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (x_pure_rand, x, x_e4m3):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...', flush=True, end='')
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  'is_token_in_rank': is_token_in_rank,
                                     'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights})
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if current_x is not x_pure_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if current_x is not x_pure_rand:
                            check_data(recv_x, recv_gbl_rank_prefix_sum)

                    # Test combine
                    bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    combine_args = {'x': recv_x, 'bias': (bias_0, bias_1), 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    check_x = (combined_x.float() - bias_0.float() - bias_1.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(4, 33, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                tune_args = {'x': current_x, 'handle': handle, 'config': config}
                t = bench(lambda: buffer.dispatch(**tune_args))[0]
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size)
                if local_rank == 0:
                    print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: {rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ', flush=True)
        if local_rank == 0:
            print(f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
            print('', flush=True)

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor([best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device='cuda')
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2], rdma_buffer_size)

    dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                     'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                     'config': dispatch_config if dispatch_config is not None else config}
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 5, 1):
        for rdma_chunk_size in range(8, 33, 4):
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            tune_args = {'x': recv_x, 'handle': handle, 'config': config}
            t = bench(lambda: buffer.combine(**tune_args))[0]
            if local_rank == 0:
                print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: {combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ', flush=True)
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size)

    if local_rank == 0:
        print(f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
        print('', flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if args.test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    num_sms = 24
    num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)

    buffer = deep_ep.Buffer(group, int(1e9), int(1e9), low_latency_mode=args.test_ll_compatibility,
                            num_qps_per_rank=num_qps_per_rank)
    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    for i in (num_sms, ):
        test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)

    # Test compatibility with low latency functions
    if args.test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the communication group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test internode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=4096,
                       help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk-groups', type=int, default=None,
                       help='Number of top-k groups (default: `min(num_nodes, 4)`)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=256,
                       help='Number of experts (default: 256')
    parser.add_argument('--test-ll-compatibility', action='store_true',
                        help='whether to test compatibility with low-latency kernels')
    args = parser.parse_args()

    # Set default `num_topk_groups` if not provided
    if args.num_topk_groups is None:
        num_nodes = int(os.getenv('WORLD_SIZE', 1))
        args.num_topk_groups = min(num_nodes, 4)

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
