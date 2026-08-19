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

:: Sets up the build environment on a Windows runner, via oneAPI's setvars.bat
:: if available, otherwise falling back to MSVC's vcvarsall.bat.

if exist "%WINDOWS_ONEAPI_PATH%\setvars.bat" (
  call "%WINDOWS_ONEAPI_PATH%\setvars.bat"
  if !errorlevel! neq 0 exit /b !errorlevel!
) else (
  if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    echo "Error: vswhere.exe not found."
    exit /b 1
  )
  set "VS_PATH="
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%i"
  if defined VS_PATH (
    if not exist "!VS_PATH!\VC\Auxiliary\Build\vcvarsall.bat" (
      echo "Error: vcvarsall.bat not found under !VS_PATH!\VC\Auxiliary\Build\"
      exit /b 1
    )
    call "!VS_PATH!\VC\Auxiliary\Build\vcvarsall.bat" amd64
    if !errorlevel! neq 0 exit /b !errorlevel!
  ) else (
    echo "Error: Could not find oneAPI install nor Visual Studio installation with MSVC tools via vswhere.exe."
    exit /b 1
  )
)
