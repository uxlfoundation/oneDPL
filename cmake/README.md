# CMake system description

The project uses CMake for integration and testing purposes.

Configuration phase can be customized by passing additional variables: `cmake -D<var1>=<val1> -D<var2>=<val2> ... -D<varN>=<valN> <sources-dir>`

The following variables are provided for oneDPL configuration:

| Variable                     | Type   | Description                                                                                   | Default value |
|------------------------------|--------|-----------------------------------------------------------------------------------------------|---------------|
| ONEDPL_BACKEND               | STRING | Threading backend; supported values: tbb, dpcpp, dpcpp_only, tbb, omp, serial, ...; the default value is defined by compiler: dpcpp for DPC++ and tbb for others. These values may conflict with explicit set variables like `-DCMAKE_CXX_FLAGS="-DONEDPL_USE_DPCPP_BACKEND=0"` etc. | tbb/dpcpp |
| ONEDPL_DEVICE_TYPE           | STRING | Select device type for oneDPL test targets; affects only DPC++ backends; supported values: GPU, CPU, FPGA_HW, FPGA_EMU | GPU           |
| ONEDPL_DEVICE_BACKEND        | STRING | Select device backend type for oneDPL test targets; affects only oneDPL DPC++ backends; supported values: opencl, level_zero, cuda, hip or * (the best backend as per DPC++ runtime heuristics). | * |
| ONEDPL_USE_UNNAMED_LAMBDA    | BOOL   | Pass `-fsycl-unnamed-lambda`, `-fno-sycl-unnamed-lambda` compile options or nothing           |               |
| ONEDPL_FPGA_STATIC_REPORT    | BOOL   | Enable the static report generation for the FPGA_HW device type                               | OFF           |
| ONEDPL_USE_AOT_COMPILATION   | BOOL   | Enable ahead-of-time compilation for the GPU or CPU device types                              | OFF           |
| ONEDPL_ENABLE_SIMD           | BOOL   | Enable SIMD vectorization by passing an OpenMP SIMD flag to the compiler if supported; the flag is passed to user project compilation string if enabled | ON           |
| ONEDPL_AOT_ARCH              | STRING | Architecture options for ahead-of-time compilation, supported values can be found [here](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html)                                                                                            | "*" for GPU device and "avx" for CPU device |
| ONEDPL_TEST_EXPLICIT_KERNEL_NAMES   | STRING | Control kernel naming. Affects only oneDPL test targets. Supported values: AUTO, ALWAYS. AUTO: rely on the compiler if "Unnamed SYCL lambda kernels" feature is on, otherwise provide kernel names explicitly; ALWAYS: provide kernel names explicitly | AUTO          |
| ONEDPL_TEST_WIN_ICX_FIXES     | BOOL   | Affects only oneDPL test targets.  Enable icx, icx-cl workarounds to fix issues in CMake for Windows.                      | ON            |
| ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION | BOOL | Use as a workaround for incorrect results, which may be produced by reduction algorithms with 64-bit data types compiled by the Intel&reg; oneAPI DPC++/C++ Compiler and executed on GPU devices. |               |

Some useful CMake variables ([here](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html) you can find a full list of CMake variables for the latest version):

- [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) - C++ compiler used for build, e.g. `CMAKE_CXX_COMPILER=dpcpp`.
- [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html) - build type that affects optimization level and debug options, values: `RelWithDebInfo`, `Debug`, `Release`, ...; e.g. `CMAKE_BUILD_TYPE=RelWithDebInfo`.
- [`CMAKE_CXX_STANDARD`](https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html) - C++ standard, e.g. `CMAKE_CXX_STANDARD=17`.

## Testing

Steps:

1. Configure project using CMake.
2. Perform build [and run] using build system (e.g. `make build-onedpl-tests`).
3. (optional) Run tests using CTest.

**NOTE**: tests are not added to `all` target, so they are not built/run by default.
The following targets are available for build system after configuration:

- `<test-name>` - build specific test, e.g. `for_each.pass`;
- `run-<test-name>` - build and run specific test, e.g. `run-for_each.pass`;
- `build-onedpl-<tests-subdir>-tests` - build all tests from specific subdirectory under `<root>/test`, e.g. `build-onedpl-general-tests`;
- `run-onedpl-<tests-subdir>-tests` - build and run all tests from specific subdirectory under `<root>/test`, e.g. `run-onedpl-general-tests`;
- `build-onedpl-tests` - build all tests;
- `run-onedpl-tests` - build and run all tests.

Sudirectories are added as labels for each test and can be used with `ctest -L <label>`.
For example, `<root>/test/path/to/test.pass.cpp` will have `path` and `to` labels.

## How to use oneDPL from CMake
### Using oneDPL source files

Using oneDPL source files allows you to integrate oneDPL source code into your project with the [add_subdirectory](https://cmake.org/cmake/help/latest/command/add_subdirectory.html) command. `add_subdirectory(<oneDPL_root_dir> [<oneDPL_output_dir>])` adds oneDPL to the user project build.
* `<oneDPL_root_dir>` is a relative or absolute path to oneDPL root directory
* `<oneDPL_output_dir>` is a relative or absolute path to directory for holding output files for oneDPL
* If `<oneDPL_root_dir>` is the relative path, then `<oneDPL_output_dir>` is optional.

The variables from the table above can be specified before the `add_subdirectory` call to customize your oneDPL configuration and build.

For example:

```cmake
project(Foo)
add_executable(foo foo.cpp)

# Specify oneDPL backend
set(ONEDPL_BACKEND tbb)

# Add oneDPL to the build.
add_subdirectory(/path/to/oneDPL build_oneDPL)
```

oneDPL tests are not configured if oneDPL in such scenario. So they can't be built and run in this case.

### Using oneDPL package

oneDPLConfig.cmake and oneDPLConfigVersion.cmake are included into oneDPL distribution.

These files allow to integrate oneDPL into user project with the [find_package](https://cmake.org/cmake/help/latest/command/find_package.html) command. Successful invocation of `find_package(oneDPL <options>)` creates imported target `oneDPL` that can be passed to the [target_link_libraries](https://cmake.org/cmake/help/latest/command/target_link_libraries.html) command.

For example:

```cmake
project(Foo)
add_executable(foo foo.cpp)

# Search for oneDPL
find_package(oneDPL REQUIRED)

# Connect oneDPL to foo
target_link_libraries(foo oneDPL)
```

Use `ONEDPL_PAR_BACKEND` variable before the invocation of `find_package(oneDPL <options>)` to control standard host (par) backend:

- Supported values:
  - `tbb` for oneTBB backend;
  - `openmp` for OpenMP backend;
  - `serial` for serial backend.
- If this variable is not set then the first suitable backend is chosen among oneTBB, OpenMP and serial, they are considered in the order as specified.
- oneDPL is considered as not found (`oneDPL_FOUND=FALSE`) if `ONEDPL_PAR_BACKEND` is specified, but not found or not supported.
- Macro `ONEDPL_USE_OPENMP_BACKEND` is set to `0` if oneTBB backend is chosen.
- Macro `ONEDPL_USE_TBB_BACKEND` is set to `0` if OpenMP backend is chosen.
- Macro `ONEDPL_USE_TBB_BACKEND` is set to `0` and `ONEDPL_USE_OPENMP_BACKEND` is set to `0` if serial backend is chosen.

### Using oneDPL package on Windows
On Windows, CMake requires some workarounds to use icx[-cl] successfully.  A CMake package has been provided 'oneDPLWindowsIntelLLVM' to apply these required workarounds.
Some workarounds are provided for icpx, but it is not fully supported on Windows at this time.  We also recommend updating to the most recent version of CMake, as they are actively improving support for Intel compilers (https://gitlab.kitware.com/cmake/cmake/-/issues/24314).
To enable the workarounds, please add `find_package(oneDPLWindowsIntelLLVM)` to your cmake file before you call `project()`.  If using oneDPL from source files, you must add oneDPL's cmake directory to your `CMAKE_PREFIX_PATH` to allow CMake to find `oneDPLWindowsIntelLLVM`. 

For example:

```cmake
list(APPEND CMAKE_PREFIX_PATH /path/to/oneDPL/cmake)
find_package(oneDPLWindowsIntelLLVM)
project(Foo)
add_executable(foo foo.cpp)

# Specify oneDPL backend
set(ONEDPL_BACKEND tbb)

# Add oneDPL to the build.
add_subdirectory(/path/to/oneDPL build_oneDPL)
```

Finally, the supported generator in the Windows environment is Ninja, we recommend using `-GNinja` in your cmake configuration.

### oneDPLConfig files generation

This section is applicable for oneDPL packaging creation process, but not for usual development flow.

`cmake/scripts/generate_config.cmake` is provided to generate oneDPLConfig files for oneDPL package.

How to use:

`cmake [-DSKIP_HEADERS_SUBDIR=<TRUE|FALSE>] [-DOUTPUT_DIR=<output_dir>] -P cmake/scripts/generate_config.cmake`

When `SKIP_HEADERS_SUBDIR` is set to false or not defined (by default), the script adds the subdirectories:
- `windows` and `linux` for headers in `<prefix></subdirectory>/include` pattern.
- `pkgconfig-win` and `pkgconfig-lin` for pkg-config files in `<prefix></subdirectory>/dpl.pc` pattern.
