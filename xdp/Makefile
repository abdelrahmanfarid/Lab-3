# Build the XDP filter (run inside the xdp-lab container, see run_demo.sh)
ARCH_INC := /usr/include/$(shell uname -m)-linux-gnu

xdp_filter.o: xdp_filter.c
	clang -O2 -g -Wall -target bpf -I$(ARCH_INC) -c $< -o $@

clean:
	rm -f xdp_filter.o

.PHONY: clean
