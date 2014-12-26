all: test

.PHONY: storage tensor

storage: storage.h dense_storage.h

tensor: tensor.h

test: test.cc storage tensor
	clang++ -std=c++11 -Wall -Werror -O2 -o test test.cc