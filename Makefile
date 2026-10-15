CXX = c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
PYTHON ?= python3
# Linux GCC: OPENMP=1 enables query-block parallelism. Default builds stay portable.
ifeq ($(OPENMP),1)
CXXFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif
SOURCES = attention.cpp main.cpp $(wildcard blocked.cpp streaming.cpp)
MODULE_FLAGS = $(if $(wildcard blocked.cpp),-DWITH_BLOCKED) $(if $(wildcard streaming.cpp),-DWITH_STREAMING)

.PHONY: all attention test full sanitize clean test-b bench-b
all: attention

attention: $(SOURCES) attention.h Makefile
	$(CXX) $(CPPFLAGS) $(MODULE_FLAGS) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test: attention
	$(PYTHON) check.py --algorithm all --tiles

full: attention
	$(PYTHON) check.py --algorithm all --full --tiles --bench --csv results.csv

sanitize:
	$(CXX) $(CPPFLAGS) $(MODULE_FLAGS) -O1 -g -std=c++17 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer $(SOURCES) $(LDFLAGS) $(LDLIBS) -o attention-sanitize
	$(PYTHON) check.py --binary ./attention-sanitize --algorithm all --tiles

clean:
	rm -f attention attention-sanitize
	rm -rf attention-sanitize.dSYM

# B-specific testing / benchmarking. Requires NumPy.
test-b: attention
	$(PYTHON) check.py --algorithm blocked --tiles

bench-b: attention
	$(PYTHON) benchmark_b.py --binary ./attention --output b_bench.csv
