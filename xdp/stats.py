"""Print the XDP filter counters and the blocklist in readable form.

Run inside the victim container:  python3 stats.py
"""
import ipaddress
import json
import subprocess

NAMES = ["passed", "dropped (blocklist)", "dropped (rate limit)"]


def dump(map_name):
    out = subprocess.run(["bpftool", "-j", "map", "dump", "name", map_name],
                         capture_output=True, text=True, check=True).stdout
    return [entry["formatted"] for entry in json.loads(out)]


# xdp_stats is a per-CPU array: add up the value of every CPU
for entry in dump("xdp_stats"):
    total = sum(cpu["value"] for cpu in entry["values"])
    print(f"{NAMES[entry['key']]:<22} {total:>10,}")

# blocklist keys are IPv4 addresses in network byte order, read as a
# little-endian integer by bpftool
for entry in dump("blocklist"):
    ip = ipaddress.IPv4Address(entry["key"].to_bytes(4, "little"))
    print(f"blocklisted {ip}: {entry['value']:,} packets dropped")
