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

# Build and run every example under $GITHUB_WORKSPACE/examples on Linux.
# Requires CXX_COMPILER, STD, BUILD_TYPE, BUILD_CONCURRENCY, TEST_TIMEOUT and
# LINUX_ONEAPI_PATH to be set in the environment.

set -exo pipefail

if [[ -f "${LINUX_ONEAPI_PATH}/setvars.sh" ]]; then
  source "${LINUX_ONEAPI_PATH}/setvars.sh"
fi

# Builds and runs the example in the current directory, in a subdirectory
# named $1, with $2 passed as extra CMAKE_CXX_FLAGS.
build_and_run() {
  mkdir "$1" && cd "$1"
  cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_CXX_STANDARD="${STD}" -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" -DCMAKE_CXX_FLAGS="$2" ..
  make VERBOSE=1 -j"${BUILD_CONCURRENCY}" |& tee build.log
  ctest --timeout "${TEST_TIMEOUT}" --output-on-failure |& tee ctest.log
  cd ..
}

for example_dir in "${GITHUB_WORKSPACE}"/examples/*; do
  if [[ -d "$example_dir" && -f "$example_dir/CMakeLists.txt" ]]; then
    cd "$example_dir"
    build_and_run build ""

    # gamma_correction supports an alternate, host-only code path selected by
    # BUILD_FOR_HOST; build and run both variants to cover it in CI.
    if [[ "$(basename "$example_dir")" == "gamma_correction" ]]; then
      build_and_run build_host "-DBUILD_FOR_HOST"
    fi
  fi
done
