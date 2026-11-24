#!/usr/bin/env bash
# Lab 3, Task 4: test the XDP filter with three containers on a private network.
#
#   xdp-victim   10.89.50.10  runs the XDP filter on eth0 and a web server on port 80
#   xdp-attacker 10.89.50.20  floods the victim, later gets blocklisted
#   xdp-client   10.89.50.30  normal user; its traffic must keep passing
#
# Needs a rootful Podman machine (or Docker); use DOCKER=docker to switch.
# All output is written to results/demo_output.txt.
set -uo pipefail
cd "$(dirname "$0")"

D=${DOCKER:-podman}
NET=xdpnet
VICTIM=10.89.50.10
ATTACKER=10.89.50.20
OUT=results/demo_output.txt
mkdir -p results
: > "$OUT"

step() { printf '\n===== %s =====\n' "$*" | tee -a "$OUT"; }
run()  { local c=$1; shift; printf '[%s]# %s\n' "${c#xdp-}" "$*" | tee -a "$OUT"
         $D exec "$c" sh -c "$*" 2>&1 | tee -a "$OUT"; }

cleanup() {
    $D rm -f xdp-victim xdp-attacker xdp-client >/dev/null 2>&1
    $D network rm $NET >/dev/null 2>&1
}

step "Setup: image, network and containers"
$D build -q -t xdp-lab . >/dev/null
cleanup
$D network create --subnet 10.89.50.0/24 $NET >/dev/null
$D run -dit --name xdp-victim   --privileged --network $NET --ip $VICTIM -v "$PWD:/xdp" xdp-lab bash >/dev/null
$D run -dit --name xdp-attacker --privileged --network $NET --ip $ATTACKER xdp-lab bash >/dev/null
$D run -dit --name xdp-client   --privileged --network $NET --ip 10.89.50.30 xdp-lab bash >/dev/null
run xdp-victim "uname -r"
run xdp-victim "mount | grep -q /sys/fs/bpf || mount -t bpf bpf /sys/fs/bpf; echo bpffs ok"
run xdp-victim "nohup python3 -m http.server 80 >/dev/null 2>&1 & sleep 1; echo web server started"

step "1. Build and attach the XDP program (generic mode)"
run xdp-victim "make clean >/dev/null; make"
run xdp-victim "ip link set dev eth0 xdpgeneric obj xdp_filter.o sec xdp"
run xdp-victim "ip -d link show dev eth0 | grep -E 'eth0|xdp'"
run xdp-victim "bpftool prog show name xdp_ddos_filter"
run xdp-victim "bpftool map show | grep -A1 -E 'blocklist|src_rate|xdp_stats'"

step "2. Normal traffic passes"
run xdp-attacker "ping -c 3 -i 0.2 $VICTIM"
run xdp-client   "curl -s -o /dev/null -w 'HTTP %{http_code} in %{time_total}s\n' http://$VICTIM/"
run xdp-victim   "python3 stats.py"

step "3. SYN flood to port 80: the rate limit drops the excess"
printf '[attacker]# timeout -s INT 5 hping3 -S -p 80 --flood %s   (in background)\n' "$VICTIM" | tee -a "$OUT"
$D exec -d xdp-attacker sh -c "timeout -s INT 5 hping3 -S -p 80 --flood $VICTIM > /tmp/hping.txt 2>&1"
sleep 2
printf '(during the flood)\n' | tee -a "$OUT"
run xdp-client   "curl -s -o /dev/null -w 'HTTP %{http_code} in %{time_total}s\n' http://$VICTIM/"
sleep 5
run xdp-attacker "tail -3 /tmp/hping.txt"
run xdp-victim   "python3 stats.py"

step "4. Blocklist the attacker (as an ML detector would)"
run xdp-victim   "bpftool map update name blocklist key ${ATTACKER//./ } value 0 0 0 0 0 0 0 0"
run xdp-victim   "bpftool map dump name blocklist"
run xdp-attacker "ping -c 3 -W 1 $VICTIM"
run xdp-attacker "curl -s -m 3 -o /dev/null -w 'HTTP %{http_code}\n' http://$VICTIM/ || echo 'curl: no answer'"
run xdp-client   "curl -s -o /dev/null -w 'HTTP %{http_code} in %{time_total}s\n' http://$VICTIM/"
run xdp-victim   "bpftool map dump name blocklist"
run xdp-victim   "python3 stats.py"

step "5. Remove the block and detach the program"
run xdp-victim   "bpftool map delete name blocklist key ${ATTACKER//./ }"
run xdp-attacker "ping -c 2 -i 0.2 $VICTIM | tail -2"
run xdp-victim   "ip link set dev eth0 xdpgeneric off"
run xdp-victim   "ip link show dev eth0 | grep -c xdp || echo 'no XDP program attached'"

[ "${KEEP:-0}" = 1 ] || cleanup
echo; echo "Output saved to $OUT"
