#!/bin/bash
##===----------------------------------------------------------------------===##
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

# Configures, builds and tests oneDPL on Linux.
# Requires BACKEND, DEVICE_TYPE, STD, BUILD_TYPE, CXX_COMPILER,
# BUILD_CONCURRENCY, TEST_TIMEOUT and LINUX_ONEAPI_PATH to be set in the environment.

set -exo pipefail

if [[ -f "${LINUX_ONEAPI_PATH}/setvars.sh" ]]; then
  source "${LINUX_ONEAPI_PATH}/setvars.sh"
fi

if [[ "${CXX_COMPILER}" == "acpp" ]]; then
  # Build the SYCL test subset excluding targets known to fail compilation with AdaptiveCpp:
  # - multiple_translation_units.pass includes DPC++-only kernel templates;
  # - test_sycl_iterator_operations.pass hits a hipSYCL sub-buffer constructor bug.
  make_targets="build-onedpl-sycl_iterator-tests build-onedpl-ranges-tests device_copyable.pass dpl_namespace.pass test_policies.pass lambda_naming.pass host_device_storage.pass"
elif [[ "${BACKEND}" == dpcpp* ]]; then
  # set targets for dpcpp tests
  make_targets="build-onedpl-general-tests build-onedpl-sycl_iterator-tests build-onedpl-implementation_details-tests build-onedpl-ranges-tests"
  tests_regex="(sycl_iterator_.*)|(device_copyable)|(dpl_namespace)|(test_policies)|(lambda_naming)|(host_device_storage)|(std_ranges_.*)"
  ctest_flags="-R ${tests_regex}\.pass"
else
  make_targets="build-onedpl-tests"
fi

cd "${GITHUB_WORKSPACE}"
mkdir build && cd build
lscpu

# TODO: fix or justify the excluded warnings
if [[ "${CXX_COMPILER}" == "acpp" ]]; then
  # AdaptiveCpp is built from develop and its headers/driver may emit warnings; keep warnings without -Werror.
  EXTRA_CXX_FLAGS="-Wall -Wextra-semi"
else
  EXTRA_CXX_FLAGS="-Wall -Wextra-semi -Werror -Wno-error=sign-compare"
fi
if [[ "${BACKEND}" == dpcpp* ]]; then
  # Reduce std_ranges view-type permutations in per-commit CI to keep
  # build time under the job limit. A couple of representative tests
  # force full coverage back on regardless of this flag.
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -DONEDPL_STD_RANGES_TEST_ALL_PERMUTATIONS=0"
fi
if [[ "${BACKEND}" == "dpcpp_only" ]]; then
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -DTEST_ONLY_HETERO_POLICIES"
fi
if [[ "${BACKEND}" == dpcpp* && "${CXX_COMPILER}" == "acpp" ]]; then
  # Explicitly enable DPC++ backend for AdaptiveCpp, as it cannot be detected automatically.
  # Disable predefined policies to avoid static sycl::queue instances hitting an AdaptiveCpp
  # shutdown-order bug in allocation_tracker::unregister_allocation.
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -DONEDPL_USE_DPCPP_BACKEND=1 -DONEDPL_USE_PREDEFINED_POLICIES=0"
fi
if [[ "${CXX_COMPILER}" != "g++" && "${CXX_COMPILER}" != "acpp" ]]; then
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -Wno-error=pass-failed"
fi
if [[ "${CXX_COMPILER}" == "icpx" ]]; then
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -Wno-error=recommended-option"
fi

cmake -DCMAKE_CXX_STANDARD="${STD}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" -DONEDPL_BACKEND="${BACKEND}" -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" ..
make VERBOSE=1 -j${BUILD_CONCURRENCY} ${make_targets} |& tee build.log
if [[ "${CXX_COMPILER}" == "acpp" ]]; then
  acpp-info
  # Skip running tests with AdaptiveCpp for now due to segmentation faults likely caused by sycl::buffer issues.
  python "${GITHUB_WORKSPACE}/.github/scripts/job_summary.py" --build-log build.log \
                                                              --output-file summary.md --compiler "${CXX_COMPILER}"
else
  export ONEAPI_DEVICE_SELECTOR=*:${DEVICE_TYPE}
  ctest --timeout "${TEST_TIMEOUT}" --output-on-failure ${ctest_flags} |& tee ctest.log

  python "${GITHUB_WORKSPACE}/.github/scripts/job_summary.py" --build-log build.log --ctest-log ctest.log \
                                                              --output-file summary.md --compiler "${CXX_COMPILER}"
fi
