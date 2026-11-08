CC ?= gcc
intref: intref.c
	$(CC) -O2 -Wall -Wextra -o intref intref.c
check: intref
	./intref ../vectors/t24 ../vectors
	./intref ../vectors/t128 ../vectors
	./intref ../vectors/t160 ../vectors
