all: playground run_tests

.PHONY: clean dense_tensor

clean:
	rm -f *.o playground run_tests

playground: playground.cc dense_tensor
	clang++ -std=c++11 -Wall -Werror -O2 -c $<
	clang++ -lglog -o playground playground.o

run_tests: dense_tensor_test.cc dense_tensor
	clang++ -std=c++11 -O2 -c $<
	clang++ -o run_tests dense_tensor_test.o -lglog -lgtest -lgtest_main -lpthread

dense_tensor: dense_tensor.h dense_tensor-inl.h