##===-- Makefile ----------------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

#------------------------------------------------------------------------------
# Define rules for making the Parallel STL library.
#------------------------------------------------------------------------------


proj_root ?= $(CURDIR)/..
bench_root ?= $(proj_root)/bench

include $(proj_root)/make/Makefile.common

.SECONDARY:
.PHONY: clean clean_all clean_pstl bench test

PSTL_MAKEFILE = $(proj_root)/make/Makefile.pstl
BENCH_MAKEFILE = $(proj_root)/make/Makefile.bench

recursive_wildcard = $(wildcard $(1)$(2)) $(foreach d,$(wildcard $(1)*),$(call recursive_wildcard,$(d)/,$(2)))

test_hdr = $(wildcard $(proj_root)/test/support/*.h)
test_src_all = $(call recursive_wildcard,$(proj_root)/test/,*.pass.cpp)
ifdef exclude_test_dirs
    ifdef include_test_dirs
        $(error Simultaneous use of 'exclude_test_dirs' 'include_test_dirs' options is not allowed)
    endif
    full_paths = $(addprefix $(proj_root)/test/, $(exclude_test_dirs))
    test_src = $(filter-out $(addsuffix /%, $(full_paths)), $(test_src_all))
endif
ifdef include_test_dirs
    full_paths = $(addprefix $(proj_root)/test/, $(include_test_dirs))
    test_src = $(filter $(addsuffix /%, $(full_paths)), $(test_src_all))
endif
test_src ?= $(test_src_all)
test_bin = $(notdir $(test_src:.cpp=.exe))
VPATH += $(sort $(dir $(test_src)))

all: $(test_bin)

$(PSTL_LIB_NAME):
	$(MAKE) -f $(PSTL_MAKEFILE) backend=$(backend) cfg=$(cfg)

ifneq (,$(filter $(backend), tbb omp serial))

ifeq (,$(filter $(compiler), dpcpp dpc++))
test_%.offload.exe: test_%.offload$(OBJ_SFX) exception_list.offload$(OBJ_SFX)
	$(CPLUS) $(CPLUS_FLAGS) $^ $(FKEY)o$@ $(LDFLAGS)

%.pass.exe: %.pass$(OBJ_SFX) $(PSTL_LIB_NAME)
	$(info LIBRARY_PATH=$(LIBRARY_PATH))
	$(LD) $< $(LD_OUT_KEY)$@ $(LDFLAGS) $(DYN_LDFLAGS) $(PSTL_LIB_LINK)

%.pass$(OBJ_SFX): %.pass.cpp $(test_hdr) $(proj_root)/make/Makefile
	$(CPLUS) $(CPLUS_FLAGS) $(KEY)c $< $(FKEY)o$@
else
%.pass.exe: %.pass.cpp $(PSTL_LIB_NAME) $(test_hdr) $(proj_root)/make/Makefile
	$(CPLUS) $(CPLUS_FLAGS) $(LDFLAGS) $(DYN_LDFLAGS) $(PSTL_LIB_LINK) $< $(FKEY)o$@
endif

else ifneq (,$(filter $(backend), sycl sycl_only))

ifeq ($(device_type), FPGA)
%.pass.exe: %.pass.cpp $(PSTL_LIB_NAME) $(test_hdr) $(proj_root)/make/Makefile
	$(CPLUS) $(CPLUS_FLAGS) $(DEVICE_COMPILE_FLAGS) $(KEY)c $< $(FKEY)o $@.o
	$(CPLUS) $(CPLUS_FLAGS) $(DEVICE_COMPILE_FLAGS) $(DYN_LDFLAGS) $@.o $(FKEY)o $@ $(DEVICE_LINK_FLAGS) $(PSTL_LIB_LINK)
else
%.pass.exe: %.pass.cpp $(PSTL_LIB_NAME) $(test_hdr) $(proj_root)/make/Makefile
ifeq ($(os_name), windows)
	$(CPLUS) $(CPLUS_FLAGS) $(DEVICE_COMPILE_FLAGS) $(LDFLAGS) $(DYN_LDFLAGS) $(DEVICE_LINK_FLAGS) $(PSTL_LIB_LINK) $<
else
	$(CPLUS) $(CPLUS_FLAGS) $(DEVICE_COMPILE_FLAGS) $(LDFLAGS) $(DYN_LDFLAGS) $(DEVICE_LINK_FLAGS) $(PSTL_LIB_LINK) $< $(FKEY)o$@
endif

endif # fpga device

endif # non-tbb backend

%.pass: %.pass.exe
	$(run_cmd) $(RUN_CMD)$*.pass.exe

# This definition intentionally consists of two blank lines
define eol


endef

test: $(test_bin)
	$(foreach test, $(test_bin), $(run_cmd) $(RUN_CMD)$(test) $(args) $(eol))

%.s: %.cpp $(proj_root)/make/Makefile
	$(CPLUS) $(CPLUS_FLAGS) -S $< $(FKEY)o$@

%.E: %.cpp
	$(CPLUS) $(CPLUS_FLAGS) -E $< >$@

algo_target ?= algorithm.json

TEMPLATE_FILES=$(wildcard $(bench_root)/*.*tmpl)
BENCH_COMMON_FILES=$(wildcard $(bench_root)/*.h) $(wildcard $(bench_root)/*.cpp)

$(BENCH_MAKEFILE): $(bench_root)/$(algo_target) $(bench_root)/gen.py  $(TEMPLATE_FILES)
	$(PYTHON) $(bench_root)/gen.py $(bench_root)/$(algo_target)

bench : $(BENCH_MAKEFILE) $(BENCH_COMMON_FILES) $(PSTL_LIB_NAME)
	$(MAKE) -f $(BENCH_MAKEFILE) bench_root=$(bench_root) proj_root=$(proj_root)

clean_bench:
	$(DEL_CMD) $(BENCH_MAKEFILE)
	$(DEL_CMD) batch.py
	$(DEL_CMD) $(proj_root)/make/bench/*.*

clean:
	$(DEL_CMD) *$(OBJ_SFX) *.exe *.E *.s *.asm *.d *.pdb *.pdb *.suo *.ilk *.o *.spv *.aocx

clean_pstl:
	$(MAKE) -f $(PSTL_MAKEFILE) clean

clean_all: clean clean_pstl clean_bench

info:
	@echo OS = $(os_name)
	@echo proj_root = "$(proj_root)"
	@echo $(CURDIR)
	@echo VPATH=$(VPATH)

-include *.d
