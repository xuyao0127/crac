CC=gcc
CX=g++

# The name will be the same as the current directory name.
NAME=${shell basename $$PWD}

# By default, your resulting plugin library will have this name.
LIBNAME=libdmtcp_${NAME}

# As you add new files to your plugin library, add the object file names here.
LIBOBJS = ${NAME}.o

# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=../../
endif
DMTCP_INCLUDE=-I${DMTCP_ROOT}/include -I${DMTCP_ROOT}/jalib -I${DMTCP_ROOT}/src
CUDA_INCLUDE=-I/usr/local/cuda/include

override CFLAGS += -g3 -O0 -fPIC -I${DMTCP_INCLUDE} ${CUDA_INCLUDE}
override CXXFLAGS += -g3 -O0 -fPIC ${DMTCP_INCLUDE} ${CUDA_INCLUDE}
LINK = ${CC}

# if version.h not found:
ifeq (,$(wildcard ${DMTCP_INCLUDE}/dmtcp/version.h))
  override CFLAGS += -DDMTCP_PACKAGE_VERSION='"3.0.0"'
endif

default: ${LIBNAME}.so tests

check: ${LIBNAME}.so tests
	# Kill an old coordinator on this port if present, just in case.
	@ ${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Note that full path of plugin (using $$PWD in this case) is required.
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 5 \
	  --with-plugin $$PWD/${LIBNAME}.so ./test/counter

${LIBNAME}.so: ${LIBOBJS}
	${LINK} -shared -fPIC -o $@ $^ -lcuda -ldl

.c.o:
	${CC} ${CFLAGS} -c -o $@ $<
.cpp.o:
	${CXX} ${CXXFLAGS} -c -o $@ $<

tests:
	cd test && $(MAKE)

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

clean: tidy
	rm -f ${LIBOBJS} ${LIBNAME}.so
	cd test && $(MAKE) clean

distclean: clean
	rm -f ${LIBNAME}.so *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

dist: distclean
	dir=`basename $$PWD`; cd ..; \
	  tar czvf $$dir.tar.gz --exclude-vcs ./$$dir
	dir=`basename $$PWD`; ls -l ../$$dir.tar.gz

.PHONY: default clean dist distclean
