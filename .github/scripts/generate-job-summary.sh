#!/bin/bash
# Summarize build.log/ctest.log from the current directory into $GITHUB_STEP_SUMMARY.
# Usage: generate-job-summary.sh <cxx_compiler>

set -euo pipefail

compiler="$1"

os_name=$(uname -a | head -n 1)
cmake_version=$(cmake --version | head -n 1)
compiler_version=$("${compiler}" --version | head -n 1)
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
