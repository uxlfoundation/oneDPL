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

# Summarize build.log/ctest.log from the current directory into $GITHUB_STEP_SUMMARY.
# Requires CXX_COMPILER to be set in the environment.

set -euo pipefail

os_name=$(uname -a | head -n 1)
cmake_version=$(cmake --version | head -n 1)
compiler_version=$("${CXX_COMPILER}" --version | head -n 1)
if [[ "$(uname)" == "Darwin" ]]; then
  cpu_model=$(sysctl -n machdep.cpu.brand_string)
else
  cpu_model=$(lscpu | grep "Model name")
fi

python "${GITHUB_WORKSPACE}/.github/scripts/job_summary.py" --build-log build.log \
                                                              --ctest-log ctest.log \
                                                              --output-file summary.md \
                                                              --os "${os_name}" \
                                                              --cmake-version "${cmake_version}" \
                                                              --compiler-version "${compiler_version}" \
                                                              --cpu-model "${cpu_model}"
cat summary.md > "$GITHUB_STEP_SUMMARY"
