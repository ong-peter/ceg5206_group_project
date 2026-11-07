CXX = c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
PYTHON ?= python3
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
IS_ARM := $(filter arm64 aarch64,$(UNAME_M))
IS_CLANG := $(shell $(CXX) --version 2>/dev/null | grep -qi clang && echo 1)
# Keep FLAG only if the compiler accepts it.
probe = $(shell $(CXX) $(1) -Werror -x c++ -c /dev/null -o /dev/null >/dev/null 2>&1 && echo $(1))

# Target the build machine's full ISA (AVX2/AVX-512 on x86, NEON on Apple silicon).
# Applied to every kernel, so baseline and optimized builds share ISA settings.
# Override with ARCH_FLAGS= for a portable binary.
ARCH_FLAGS ?= $(if $(IS_ARM),$(call probe,-mcpu=native),$(call probe,-march=native))
# x86 only: preferred SIMD width in bits (256 or 512). Empty keeps the compiler
# default (GCC uses 256 on most AVX-512 CPUs). ARM NEON is fixed at 128 bits.
VEC_WIDTH ?=
ifneq ($(VEC_WIDTH),)
ifeq ($(IS_ARM),)
ARCH_FLAGS += -mprefer-vector-width=$(VEC_WIDTH)
endif
endif
# Honour "omp simd" loop hints (simd.h) without the OpenMP runtime.
SIMD_FLAGS = -fopenmp-simd
# Vectorization ablation: same flags and ISA, but no auto-vectorization or SIMD hints.
NOVEC_FLAGS = $(if $(IS_CLANG),-fno-vectorize -fno-slp-vectorize,-fno-tree-vectorize -fno-tree-slp-vectorize) -DATTN_NO_SIMD
VEC_REPORT_FLAGS = $(if $(IS_CLANG),-Rpass=loop-vectorize,-fopt-info-vec-optimized)

# OPENMP=1 enables query-block parallelism. Default builds stay portable.
# macOS: Apple Clang needs Homebrew libomp (brew install libomp).
ifeq ($(OPENMP),1)
ifeq ($(UNAME_S)$(IS_CLANG),Darwin1)
LIBOMP ?= $(shell brew --prefix libomp 2>/dev/null)
CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP)/include
LDFLAGS += -L$(LIBOMP)/lib -lomp
else
CXXFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif
endif

SOURCES = attention.cpp main.cpp $(wildcard blocked.cpp streaming.cpp)
HEADERS = attention.h simd.h
MODULE_FLAGS = $(if $(wildcard blocked.cpp),-DWITH_BLOCKED) $(if $(wildcard streaming.cpp),-DWITH_STREAMING)

COMPILE = $(CXX) $(CPPFLAGS) $(MODULE_FLAGS) $(CXXFLAGS) $(ARCH_FLAGS)
BUILD_FLAGS = $(COMPILE) $(SIMD_FLAGS) $(LDFLAGS)
NOVEC_BUILD_FLAGS = $(COMPILE) $(NOVEC_FLAGS) $(LDFLAGS)
# x86 SIMD-width comparison binaries (bench-width).
WIDTH_COMPILE = $(CXX) $(CPPFLAGS) $(MODULE_FLAGS) $(CXXFLAGS) $(call probe,-march=native) $(SIMD_FLAGS)

.PHONY: all attention-flags test test-threads full sanitize clean test-b bench bench-quick bench-scaling bench-width vec-report
all: attention

# Rebuild when flags change (e.g. OPENMP=1 after a default build).
.build-flags: attention-flags
	@echo '$(BUILD_FLAGS)' | cmp -s - $@ || echo '$(BUILD_FLAGS)' > $@

attention: $(SOURCES) $(HEADERS) Makefile .build-flags
	$(COMPILE) $(SIMD_FLAGS) -DATTN_BUILD_FLAGS='"$(BUILD_FLAGS)"' $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

attention-novec: $(SOURCES) $(HEADERS) Makefile .build-flags
	$(COMPILE) $(NOVEC_FLAGS) -DATTN_BUILD_FLAGS='"$(NOVEC_BUILD_FLAGS)"' $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

attention-w256 attention-w512: attention-w%: $(SOURCES) $(HEADERS) Makefile .build-flags
	@test -z "$(IS_ARM)" || { echo "SIMD width comparison is x86-only (NEON is 128-bit)"; exit 1; }
	$(WIDTH_COMPILE) -mprefer-vector-width=$* -DATTN_BUILD_FLAGS='"$(WIDTH_COMPILE) -mprefer-vector-width=$* $(LDFLAGS)"' $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test: attention
	$(PYTHON) check.py --algorithm all --tiles

# Thread correctness: outputs must be bitwise identical to one thread.
# Covers odd counts, more threads than query blocks, and oversubscription.
CORES := $(shell $(PYTHON) -c "from bench import physical_cores; print(physical_cores())" 2>/dev/null || echo 4)
LOGICAL := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo $(CORES))
# 1, 2, 4, ... below the physical core count, plus the core count itself.
SCALING_THREADS ?= $(shell $(PYTHON) -c "c=$(CORES); print(','.join(str(t) for t in sorted({c, *(2**i for i in range(c.bit_length()) if 2**i < c)})))")
test-threads: attention
	@./attention --build-info | grep -q openmp=1 || { echo "Rebuild with: make clean && make OPENMP=1"; exit 1; }
	@for t in 2 3 $(CORES) $(LOGICAL) 64; do \
		echo "== $$t threads"; $(PYTHON) check.py --algorithm all --tiles --threads $$t | tail -1 || exit 1; \
	done

# Correctness on all 12 core shapes plus edge cases; errors saved to results.csv.
full: attention
	$(PYTHON) check.py --algorithm all --full --tiles $(TILE_ARG) --csv results.csv

# Cumulative comparison: baseline -> blocked -> streaming -> threads, plus no-vec ablation.
# Use OPENMP=1 for the threaded stages.
# Tuned tiles from check.py --tune are used automatically when present.
TILE_ARG = $(if $(wildcard tuned-tiles.json),--tile-config tuned-tiles.json)
bench: attention attention-novec
	$(PYTHON) bench.py --full --novec-binary ./attention-novec $(TILE_ARG) --csv ladder.csv

bench-quick: attention attention-novec
	$(PYTHON) bench.py --novec-binary ./attention-novec $(TILE_ARG) --csv ladder.csv

# Thread scaling (Task 4): 1, 2, 4, ... up to physical cores, threads pinned to cores.
bench-scaling: attention
	@./attention --build-info | grep -q openmp=1 || { echo "Rebuild with: make clean && make OPENMP=1"; exit 1; }
	OMP_PLACES=cores OMP_PROC_BIND=close $(PYTHON) bench.py --threads $(SCALING_THREADS) \
		--shapes 512x64,4096x64 --rounds 3 $(TILE_ARG) --csv scaling.csv

# x86 AVX-512 CPUs: compare 256- and 512-bit vectors on the optimized kernels.
bench-width: attention-w256 attention-w512
	$(PYTHON) bench.py --binary ./attention-w256 --extra-binary w512=./attention-w512 $(TILE_ARG) --csv width.csv

# Which loops vectorized (GCC -fopt-info / Clang -Rpass remarks).
vec-report:
	$(COMPILE) $(SIMD_FLAGS) $(VEC_REPORT_FLAGS) -c $(wildcard blocked.cpp streaming.cpp) attention.cpp 2>&1 | \
		grep -E '\.cpp:[0-9]+' | sort -u | tee vectorization_report.txt
	@rm -f attention.o blocked.o streaming.o

sanitize:
	$(CXX) $(CPPFLAGS) $(MODULE_FLAGS) $(ARCH_FLAGS) $(SIMD_FLAGS) -O1 -g -std=c++17 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer $(SOURCES) $(LDFLAGS) $(LDLIBS) -o attention-sanitize
	$(PYTHON) check.py --binary ./attention-sanitize --algorithm all --tiles

clean:
	rm -f attention attention-novec attention-w256 attention-w512 attention-sanitize .build-flags
	rm -rf attention-sanitize.dSYM

# B-specific testing. Requires NumPy.
test-b: attention
	$(PYTHON) check.py --algorithm blocked --tiles
