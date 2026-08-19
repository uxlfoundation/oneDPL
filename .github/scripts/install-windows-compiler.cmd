:: Installs the Intel oneAPI DPC++/C++ Compiler on a Windows runner.
:: Requires WINDOWS_ICPX_DOWNLOAD_LINK and WINDOWS_ONEAPI_PATH to be set in the environment.

curl %WINDOWS_ICPX_DOWNLOAD_LINK% --output icpx_install.exe
icpx_install.exe -s -a --silent --eula accept -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0
del icpx_install.exe
:: Avoid accidental use of a released version
rd /s /q "%WINDOWS_ONEAPI_PATH%\dpl"
