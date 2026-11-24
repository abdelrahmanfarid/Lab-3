// SPDX-License-Identifier: GPL-2.0
/*
 * xdp_filter.c - small XDP DDoS filter for Lab 3, Task 4
 *
 * For every received IPv4 packet:
 *   1. drop it if its source address is in the `blocklist` map
 *      (filled from user space, e.g. by an ML detector or with bpftool);
 *   2. otherwise drop it if its source sent more than RATE_LIMIT_PPS
 *      packets in the current one-second window (`src_rate` map);
 *   3. otherwise pass it to the normal network stack.
 * Non-IPv4 and truncated packets are passed. Every verdict is counted
 * in the per-CPU `xdp_stats` array.
 */
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

#define RATE_LIMIT_PPS 1000          /* packets per second per source IP */
#define NS_PER_SEC     1000000000ULL

enum { STAT_PASS, STAT_DROP_BLOCKLIST, STAT_DROP_RATE, STAT_MAX };

/* key: IPv4 source address (network byte order), value: dropped packets */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);
    __type(key, __u32);
    __type(value, __u64);
} blocklist SEC(".maps");

struct rate_state {
    __u64 window_start;              /* ns, from bpf_ktime_get_ns() */
    __u64 packets;                   /* packets seen in this window */
};

/* LRU: when full, the least recently seen sources are evicted */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);
    __type(value, struct rate_state);
} src_rate SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, STAT_MAX);
    __type(key, __u32);
    __type(value, __u64);
} xdp_stats SEC(".maps");

static __always_inline int verdict(__u32 stat, int action)
{
    __u64 *count = bpf_map_lookup_elem(&xdp_stats, &stat);

    if (count)
        *count += 1;                 /* per-CPU slot: no atomic needed */
    return action;
}

SEC("xdp")
int xdp_ddos_filter(struct xdp_md *ctx)
{
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    /* The verifier rejects any access that is not bounds-checked. */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return verdict(STAT_PASS, XDP_PASS);
    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return verdict(STAT_PASS, XDP_PASS);

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return verdict(STAT_PASS, XDP_PASS);

    __u32 src = ip->saddr;

    /* 1. Blocklist */
    __u64 *hits = bpf_map_lookup_elem(&blocklist, &src);
    if (hits) {
        __sync_fetch_and_add(hits, 1);
        return verdict(STAT_DROP_BLOCKLIST, XDP_DROP);
    }

    /* 2. Per-source rate limit (fixed one-second window) */
    __u64 now = bpf_ktime_get_ns();
    struct rate_state *st = bpf_map_lookup_elem(&src_rate, &src);
    if (!st) {
        struct rate_state fresh = { .window_start = now, .packets = 1 };
        bpf_map_update_elem(&src_rate, &src, &fresh, BPF_ANY);
        return verdict(STAT_PASS, XDP_PASS);
    }
    if (now - st->window_start >= NS_PER_SEC) {
        st->window_start = now;
        st->packets = 1;
        return verdict(STAT_PASS, XDP_PASS);
    }
    if (__sync_fetch_and_add(&st->packets, 1) >= RATE_LIMIT_PPS)
        return verdict(STAT_DROP_RATE, XDP_DROP);

    /* 3. Everything else */
    return verdict(STAT_PASS, XDP_PASS);
}

char LICENSE[] SEC("license") = "GPL";
