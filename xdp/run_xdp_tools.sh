#!/usr/bin/env bash
# Lab 3, Task 5: download, build and use xdp-tools (same three containers as Task 4).
#
#   xdp-victim   10.89.50.10  xdp-tools installed here; web server on port 80
#   xdp-attacker 10.89.50.20  ping / SYN flood
#   xdp-client   10.89.50.30  normal user
#
# Needs a rootful Podman machine (or Docker; use DOCKER=docker) and Internet
# access to clone xdp-tools from GitHub.
# Output: results/xdp_tools_output.txt (all steps) and
#         results/xdp_tools_build_log.txt (full configure/make/install log).
set -uo pipefail
cd "$(dirname "$0")"

D=${DOCKER:-podman}
NET=xdpnet
VICTIM=10.89.50.10
ATTACKER=10.89.50.20
CLIENT=10.89.50.30
XDP_TOOLS_VERSION=v1.6.3
OUT=results/xdp_tools_output.txt
BUILD_LOG=results/xdp_tools_build_log.txt
mkdir -p results
: > "$OUT"

step() { printf '\n===== %s =====\n' "$*" | tee -a "$OUT"; }
run()  { local c=$1; shift; printf '[%s]# %s\n' "${c#xdp-}" "$*" | tee -a "$OUT"
         $D exec "$c" sh -c "$*" 2>&1 | tee -a "$OUT"; }
curl_client() { run xdp-client "curl -s -m 3 -o /dev/null -w 'HTTP %{http_code}\n' http://$VICTIM/ || echo 'curl: no answer'"; }
cleanup() {
    $D rm -f xdp-victim xdp-attacker xdp-client >/dev/null 2>&1
    $D network rm $NET >/dev/null 2>&1
}

step "Setup: images, network and containers"
$D build -q -t xdp-lab . >/dev/null
$D build -q -t xdp-tools -f Dockerfile.xdptools . >/dev/null
cleanup
$D network create --subnet 10.89.50.0/24 $NET >/dev/null
$D run -dit --name xdp-victim   --hostname victim   --privileged --network $NET --ip $VICTIM   -v "$PWD:/xdp" xdp-tools bash >/dev/null
$D run -dit --name xdp-attacker --hostname attacker --privileged --network $NET --ip $ATTACKER xdp-lab bash >/dev/null
$D run -dit --name xdp-client   --hostname client   --privileged --network $NET --ip $CLIENT   xdp-lab bash >/dev/null
run xdp-victim "uname -r; . /etc/os-release; echo \$PRETTY_NAME; uname -m; nproc"
run xdp-victim "nohup python3 -m http.server 80 >/dev/null 2>&1 & sleep 1; echo web server started"

step "1. Download and build xdp-tools $XDP_TOOLS_VERSION"
run xdp-victim "git -c advice.detachedHead=false clone -q --depth 1 --branch $XDP_TOOLS_VERSION --recurse-submodules https://github.com/xdp-project/xdp-tools.git /opt/xdp-tools && cd /opt/xdp-tools && git log -1 --format='%h %cd'"
printf '[victim]# cd /opt/xdp-tools && ./configure && make -j4 && make install   (full log: %s)\n' "$BUILD_LOG" | tee -a "$OUT"
$D exec xdp-victim sh -c "cd /opt/xdp-tools && ./configure && make -j4 && make install && ldconfig" > "$BUILD_LOG" 2>&1
echo "exit code: $?" | tee -a "$OUT"
grep -E '^Found clang|^using .*bpftool|^libbpf support' "$BUILD_LOG" | tee -a "$OUT"
run xdp-victim "ls /usr/local/sbin; xdpdump --version"

step "2. BPF file system and xdp-loader status"
run xdp-victim "xdp-loader status 2>&1 | tail -2"
run xdp-victim "xdp-filter load -m skb eth0 2>&1; echo exit code \$?"
run xdp-victim "xdp-loader unload eth0 --all 2>/dev/null; mount -t bpf bpf /sys/fs/bpf && mount | grep ' /sys/fs/bpf '"

step "3. xdp-filter: block the attacker's source IP"
run xdp-victim "xdp-filter load -m skb eth0 && xdp-loader status"
run xdp-victim "xdp-filter ip $ATTACKER --mode src"
run xdp-attacker "ping -c 3 -W 1 $VICTIM | tail -2"
curl_client

step "4. xdp-filter: block destination port 80 (affects everyone)"
run xdp-victim "xdp-filter port 80"
curl_client
run xdp-victim "xdp-filter port 80 --remove"
curl_client
run xdp-victim "xdp-filter status"

step "5. xdpdump: capture at the XDP hook, with the verdict"
printf '[victim]# xdpdump -i eth0 --rx-capture entry,exit   (5 s, in background)\n' | tee -a "$OUT"
$D exec -d xdp-victim sh -c "timeout -s INT 5 xdpdump -i eth0 --rx-capture entry,exit > /tmp/xdpdump.txt 2>&1"
sleep 1
run xdp-attacker "ping -c 1 -W 1 $VICTIM > /dev/null; echo ping sent"
curl_client
sleep 5
run xdp-victim "cat /tmp/xdpdump.txt"

step "6. Load our Task 4 filter with xdp-loader, next to xdp-filter"
run xdp-victim "xdp-filter ip $ATTACKER --mode src --remove"
run xdp-victim "make -s -C /xdp && xdp-loader load -m skb eth0 /xdp/xdp_filter.o && xdp-loader status"
printf '[attacker]# timeout -s INT 5 hping3 -S -p 80 --flood %s   (in background)\n' "$VICTIM" | tee -a "$OUT"
$D exec -d xdp-attacker sh -c "timeout -s INT 5 hping3 -S -p 80 --flood $VICTIM > /tmp/hping.txt 2>&1"
sleep 2
curl_client
sleep 5
run xdp-attacker "grep transmitted /tmp/hping.txt"
run xdp-victim "python3 /xdp/stats.py"
run xdp-victim "xdp-filter status | sed -n '1,6p'"

step "7. Unload everything"
run xdp-victim "xdp-filter unload eth0 && xdp-loader unload eth0 --all && xdp-loader status"

step "8. xdp-bench drop during a SYN flood (generic mode)"
$D exec -d xdp-attacker sh -c "timeout -s INT 9 hping3 -S -p 80 --flood $VICTIM > /tmp/hping.txt 2>&1"
printf '(attacker floods for 9 s in background)\n' | tee -a "$OUT"
sleep 1
run xdp-victim "timeout -s INT 7 xdp-bench drop -m skb eth0"
sleep 2
run xdp-victim "xdp-loader status | tail -3"

[ "${KEEP:-0}" = 1 ] || cleanup
rm -f xdp_filter.o
echo; echo "Output saved to $OUT"
