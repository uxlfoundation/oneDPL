@echo off
REM
REM ===----------------------------------------------------------------------===
REM
REM Copyright (C) 2023 Intel Corporation
REM
REM SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
REM
REM This file incorporates work covered by the following copyright and permission
REM notice:
REM
REM Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
REM See https://llvm.org/LICENSE.txt for license information.
REM
REM ===----------------------------------------------------------------------===
REM

if not defined SETVARS_CALL (
    echo:
    echo :: ERROR: This script must be executed by oneapi-vars.bat.
    echo:   Try '[install-dir]\oneapi-vars.bat --help' for help.
    echo:
    exit /b 255
)

if not defined ONEAPI_ROOT (
    echo:
    echo :: ERROR: This script requires that the ONEAPI_ROOT env variable is set."
    echo:   Try '[install-dir]\oneapi-vars.bat --help' for help.
    echo:
    exit /b 254
)

rem ############################################################################

set "DPL_ROOT=%ONEAPI_ROOT%"
set "DPLROOT=%ONEAPI_ROOT%"
