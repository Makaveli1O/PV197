framework:
	nvcc -O3 -use_fast_math -o framework framework.cu
clean:
	-rm -f framework
run: framework
	./framework
