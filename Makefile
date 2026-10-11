CXX = c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
PYTHON ?= python3

.PHONY: all test full sanitize clean
all: attention

attention: attention.cpp main.cpp attention.h
	$(CXX) $(CXXFLAGS) attention.cpp main.cpp -o $@

test: attention
	$(PYTHON) check.py

full: attention
	$(PYTHON) check.py --full --bench --csv results.csv

sanitize:
	$(CXX) -O1 -g -std=c++17 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer attention.cpp main.cpp -o attention-sanitize
	$(PYTHON) check.py --binary ./attention-sanitize

clean:
	rm -f attention attention-sanitize
	rm -rf attention-sanitize.dSYM
