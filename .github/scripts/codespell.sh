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

SCAN_TARGET=$1

SKIP_PATTERN='*/.github/*,*/examples/pSTL_offload/FileWordCount/*'

# Ignored cases
IGNORE_COMMAND="sed -e /.*windows.inc.*Od\\s*=.*/d \
-e /.*README.md.*varN\\s*=.*/d"

SCAN_RESULT=$(codespell -I ${GITHUB_WORKSPACE}/.github/scripts/allowed_words.txt --ignore-words=FILE --quiet-level=2 --skip "${SKIP_PATTERN}" ${SCAN_TARGET})
SCAN_RESULT=$(echo -e "${SCAN_RESULT}" | ${IGNORE_COMMAND})
echo "${SCAN_RESULT}"

if [[ ! -z ${SCAN_RESULT} ]]; then
    exit 1
fi
