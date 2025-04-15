@echo off
REM
REM ===----------------------------------------------------------------------===
REM
REM Copyright (C) Intel Corporation
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

set "VARSDIR=%~dp0"

if not defined DPL_ROOT for /f "delims=" %%F in ("%VARSDIR%..") do set "DPL_ROOT=%%~fF"

set "CPLUS_INCLUDE_PATH=%DPL_ROOT%\include;%CPLUS_INCLUDE_PATH%"
set "PKG_CONFIG_PATH=%DPL_ROOT%\lib\pkgconfig;%PKG_CONFIG_PATH%"

set "CMAKE_PREFIX_PATH=%DPL_ROOT%\lib\cmake\oneDPL;%CMAKE_PREFIX_PATH%"
