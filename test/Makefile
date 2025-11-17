CC=gcc
CX=g++

FILES=counter mpi_cuda

default: ${FILES}

counter: counter.cu
	# nvcc -g -O0 -o $@ $^ -cudart=shared
	nvcc -g -O0 -o $@ $^ -cudart=shared -I/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/include -L/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib -lmpi

mpi_cuda: mpi_cuda.cu
	nvcc -g -O0 -o $@ $^ -cudart=shared -I/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/include -L/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib -lmpi

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

clean: tidy
	rm -f ${LIBOBJS} ${LIBNAME}.so

distclean: clean
	rm -f ${LIBNAME}.so *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

dist: distclean
	dir=`basename $$PWD`; cd ..; \
	  tar czvf $$dir.tar.gz --exclude-vcs ./$$dir
	dir=`basename $$PWD`; ls -l ../$$dir.tar.gz

.PHONY: default clean dist distclean
