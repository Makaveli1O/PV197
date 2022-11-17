framework:
	nvcc -O3 -use_fast_math -o framework framework.cu
optimized:
	nvcc --expt-relaxed-constexpr -O3 -use_fast_math -g -std=c++17 framework.cu -o framework
clean:
	-rm -f framework
run: framework
	./framework
