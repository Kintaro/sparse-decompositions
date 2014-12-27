all: test

test: test.cc dense_tensor.h
	clang++ -std=c++14 -Wall -Werror -O2 -g3 -lglog -o test test.cc
