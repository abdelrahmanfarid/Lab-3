# Image for the Lab 3 Task 4 XDP test (victim, attacker and client containers)
FROM docker.io/library/debian:bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      clang llvm make libbpf-dev libelf-dev linux-libc-dev bpftool \
      iproute2 iputils-ping curl hping3 python3 procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /xdp
