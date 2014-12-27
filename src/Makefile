all: playground test

playground: playground.cc dense_tensor.h
	clang++ -std=c++11 -Wall -Werror -O2 -c playground.cc
	clang++ -lglog -o playground playground.o

test: test.cc 
	clang++ -std=c++11 -Wall -Werror -O2 -c test.cc
	clang++ -o test test.o -lglog -lgtest -lgtest_main -lpthread
