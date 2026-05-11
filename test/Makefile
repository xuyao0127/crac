CC=gcc
CXX=g++

FILES=counter counter_mpi mpi_cuda

default: counter

mpi: counter_mpi mpi_cuda

counter: counter.cu
	nvcc -g -O0 -o $@ $^ -cudart=shared

counter_mpi: counter_mpi.cu
	nvcc -g -O0 -o $@ $^ -cudart=shared -I/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/include -L/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib -lmpi

mpi_cuda: mpi_cuda.cu
	nvcc -g -O0 -o $@ $^ -cudart=shared -I/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/include -L/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib -lmpi

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

clean: tidy
	rm -f $(FILES)

distclean: clean
	rm -f dmtcp_restart_script*.sh ckpt_*.dmtcp

.PHONY: default mpi tidy clean distclean
