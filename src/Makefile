all: run_tests

.PHONY: clean dense_tensor run_tests

clean:
	find . -type f -iname "*.o" -exec rm -f {} \;
	rm -f run_tests

dense_tensor: tensor/dense_tensor.h tensor/dense_tensor-inl.h

run_tests:
	clang++ -std=c++11 -g3 -I./ -o run_tests test/*.cc -lglog -lgtest -lgtest_main -lpthread
