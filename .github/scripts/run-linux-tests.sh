#!/bin/bash
# Configures, builds and tests oneDPL on Linux.
# Requires BACKEND, DEVICE_TYPE, STD, BUILD_TYPE, CXX_COMPILER,
# BUILD_CONCURRENCY, TEST_TIMEOUT and LINUX_ONEAPI_PATH to be set in the
# environment.

set -exo pipefail

if [[ -f "${LINUX_ONEAPI_PATH}/setvars.sh" ]]; then
  source "${LINUX_ONEAPI_PATH}/setvars.sh"
fi

if [[ "${BACKEND}" == "dpcpp" ]]; then
  # set targets for dpcpp tests
  make_targets="build-onedpl-general-tests build-onedpl-sycl_iterator-tests build-onedpl-implementation_details-tests"
  tests_regex="(sycl_iterator_.*)|(device_copyable)|(dpl_namespace)|(test_policies)|(lambda_naming)|(host_device_storage)"
  if [[ "${DEVICE_TYPE}" != "FPGA_EMU" ]]; then
    make_targets+=" build-onedpl-ranges-tests"
    tests_regex+="|(std_ranges_.*)"
  fi
  ctest_flags="-R ${tests_regex}\.pass"
else
  make_targets="build-onedpl-tests"
fi

cd "${GITHUB_WORKSPACE}"
mkdir build && cd build
lscpu

# TODO: fix or justify the excluded warnings
EXTRA_CXX_FLAGS="-Wall -Wextra-semi -Werror -Wno-error=sign-compare"
if [[ "${BACKEND}" == "dpcpp" ]]; then
  # Reduce std_ranges view-type permutations in per-commit CI to keep
  # build time under the job limit. A couple of representative tests
  # force full coverage back on regardless of this flag.
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -DONEDPL_STD_RANGES_TEST_ALL_PERMUTATIONS=0"
fi
if [[ "${CXX_COMPILER}" != "g++" ]]; then
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -Wno-error=pass-failed"
fi
if [[ "${CXX_COMPILER}" == "icpx" ]]; then
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -Wno-error=recommended-option"
fi

if [[ "${DEVICE_TYPE}" == "FPGA_EMU" ]]; then
  EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -Wno-error=deprecated-declarations -DONEDPL_FPGA_DEVICE -DONEDPL_FPGA_EMULATOR"
fi

cmake -DCMAKE_CXX_STANDARD="${STD}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" -DONEDPL_BACKEND="${BACKEND}" -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" ..
make VERBOSE=1 -j${BUILD_CONCURRENCY} ${make_targets} |& tee build.log
ONEAPI_DEVICE_SELECTOR=*:${DEVICE_TYPE}
ctest --timeout "${TEST_TIMEOUT}" --output-on-failure ${ctest_flags} |& tee ctest.log

"${GITHUB_WORKSPACE}/.github/scripts/generate-job-summary.sh" "${CXX_COMPILER}"
