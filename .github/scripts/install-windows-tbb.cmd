:: Installs Intel oneAPI Threading Building Blocks on a Windows runner.
:: Requires WINDOWS_TBB_DOWNLOAD_LINK to be set in the environment.

curl %WINDOWS_TBB_DOWNLOAD_LINK% --output tbb_install.exe
tbb_install.exe -s -a --silent --eula accept -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0
del tbb_install.exe
