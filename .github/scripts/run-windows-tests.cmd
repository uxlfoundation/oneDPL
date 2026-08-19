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

:: Configures, builds and tests oneDPL on Windows.
:: Requires BACKEND, DEVICE_TYPE, STD, BUILD_TYPE, CXX_COMPILER,
:: BUILD_CONCURRENCY, TEST_TIMEOUT and WINDOWS_ONEAPI_PATH to be set in the
:: environment.

SETLOCAL ENABLEDELAYEDEXPANSION
call "%GITHUB_WORKSPACE%\.github\scripts\setup-windows-env.cmd"
if !errorlevel! neq 0 exit /b !errorlevel!

set exit_code=0
:: Preserve the code of an unsuccessful command if any.
:: By default, CMD shell only reports the error level of the final command.

:: cache the path to the current directory
set BASE_DIR=%cd%
if "%BACKEND%" == "dpcpp" (
  set ninja_targets=build-onedpl-general-tests build-onedpl-sycl_iterator-tests build-onedpl-implementation_details-tests build-onedpl-ranges-tests
  set ctest_flags=-R "(sycl_iterator_.*)|(std_ranges_.*)|(device_copyable)|(dpl_namespace)|(test_policies)|(lambda_naming)|(host_device_storage)\.pass"
) else (
  set ninja_targets=build-onedpl-tests
)

cd %BASE_DIR%
mkdir build && cd build

:: TODO: fix or justify the excluded warnings
if "%CXX_COMPILER%" == "cl" (
  set warning_flags=/W4 /WX /wd4018 /wd4100 /wd4146 /wd4244 /wd4245 /wd4267 /wd4310 /wd4389 /wd4805 /wd4996
) else (
  set warning_flags=-Wall -Werror -Wno-error=sign-compare -Wno-error=pass-failed
)

:: Reduce std_ranges view-type permutations in per-commit CI to keep
:: build time under the job limit. A couple of representative tests
:: force full coverage back on regardless of this flag.
if "%BACKEND%" == "dpcpp" (
  set warning_flags=!warning_flags! -DONEDPL_STD_RANGES_TEST_ALL_PERMUTATIONS=0
)

cmake -G "Ninja" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_CXX_STANDARD=%STD% -DCMAKE_CXX_COMPILER=%CXX_COMPILER% -DONEDPL_BACKEND=%BACKEND% -DCMAKE_CXX_FLAGS="%warning_flags%" ..
if !errorlevel! neq 0 set exit_code=!errorlevel!
for %%t in (%ninja_targets%) do (
  ninja -j %BUILD_CONCURRENCY% -v "%%t" >> build.log 2>&1
  if !errorlevel! neq 0 set exit_code=!errorlevel!
)
set ONEAPI_DEVICE_SELECTOR=*:%DEVICE_TYPE%
ctest --timeout %TEST_TIMEOUT% -C %BUILD_TYPE% --output-on-failure %ctest_flags% > ctest.log 2>&1
if !errorlevel! neq 0 set exit_code=!errorlevel!
type ctest.log

:: Display the results after executing all tests because "tee" alternative is unavailable in CMD.
type build.log

:: Generate a summary
powershell -command "(Get-CimInstance -ClassName Win32_OperatingSystem).Caption" > os_name.txt
set /p os_name=<os_name.txt
powershell -command "cmake --version | Select-Object -First 1" > cmake_version.txt
set /p cmake_version=<cmake_version.txt
:: cl writes the version into stderr
powershell -command "%CXX_COMPILER% --version | Select-Object -First 1" > compiler_version.txt 2>&1
set /p compiler_version=<compiler_version.txt
powershell -command "(Get-CimInstance -ClassName Win32_Processor).Name" > cpu_model.txt
set /p cpu_model=<cpu_model.txt
python %GITHUB_WORKSPACE%\.github\scripts\job_summary.py --build-log build.log ^
                                                         --ctest-log ctest.log ^
                                                         --output-file summary.md ^
                                                         --os "%os_name%" ^
                                                         --cmake-version "%cmake_version%" ^
                                                         --compiler-version "%compiler_version%" ^
                                                         --cpu-model "%cpu_model%"
type summary.md > "%GITHUB_STEP_SUMMARY%"
exit /b !exit_code!
