intref_cuda: intref.cu
	nvcc -O2 -arch=native -o intref_cuda intref.cu
check: intref_cuda
	./intref_cuda ../vectors/t24 ../vectors
	./intref_cuda ../vectors/t128 ../vectors
	./intref_cuda ../vectors/t160 ../vectors
