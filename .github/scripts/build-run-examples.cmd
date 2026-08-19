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

:: Builds and runs every example under %GITHUB_WORKSPACE%\examples on Windows.
:: Requires CXX_COMPILER, STD, BUILD_TYPE, BUILD_CONCURRENCY, TEST_TIMEOUT and
:: WINDOWS_ONEAPI_PATH to be set in the environment.

SETLOCAL ENABLEDELAYEDEXPANSION
call "%GITHUB_WORKSPACE%\.github\scripts\setup-windows-env.cmd"
if !errorlevel! neq 0 exit /b !errorlevel!

set exit_code=0
set BASE_DIR=%cd%
for /D %%i in (%GITHUB_WORKSPACE%\examples\*) do (
  if exist "%%i\CMakeLists.txt" (
    cd "%%i"
    mkdir build && cd build
    cmake -GNinja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_CXX_STANDARD=%STD% -DCMAKE_CXX_COMPILER=%CXX_COMPILER% ..
    if !errorlevel! neq 0 set exit_code=!errorlevel!
    ninja -j %BUILD_CONCURRENCY% -v > build.log 2>&1
    if !errorlevel! neq 0 set exit_code=!errorlevel!
    ctest --timeout %TEST_TIMEOUT% --output-on-failure > ctest.log 2>&1
    if !errorlevel! neq 0 set exit_code=!errorlevel!
    type ctest.log
    type build.log
    cd "%%i"

    :: gamma_correction supports an alternate, host-only code path selected by
    :: BUILD_FOR_HOST; build and run both variants to cover it in CI.
    if /I "%%~nxi" == "gamma_correction" (
      mkdir build_host && cd build_host
      cmake -GNinja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_CXX_STANDARD=%STD% -DCMAKE_CXX_COMPILER=%CXX_COMPILER% -DCMAKE_CXX_FLAGS=-DBUILD_FOR_HOST ..
      if !errorlevel! neq 0 set exit_code=!errorlevel!
      ninja -j %BUILD_CONCURRENCY% -v > build.log 2>&1
      if !errorlevel! neq 0 set exit_code=!errorlevel!
      ctest --timeout %TEST_TIMEOUT% --output-on-failure > ctest.log 2>&1
      if !errorlevel! neq 0 set exit_code=!errorlevel!
      type ctest.log
      type build.log
      cd "%%i"
    )

    if !exit_code! neq 0 exit /b !exit_code!
    cd %BASE_DIR%
  )
)
exit /b !exit_code!
