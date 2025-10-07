# How to Contribute

We welcome community contributions to oneAPI DPC++ Library (oneDPL). You can:

- Submit your changes directly with a [pull request](https://github.com/uxlfoundation/oneDPL/pulls).
- Log a bug or feedback with an [issue](https://github.com/uxlfoundation/oneDPL/issues).

# License

oneDPL is licensed under the terms in [LICENSE](https://github.com/uxlfoundation/oneDPL/blob/main/LICENSE.txt).
By contributing to the project, you agree to the license and copyright terms therein and
release your contribution under these terms.

# Pull Requests

This project follows the
[GitHub flow](https://guides.github.com/introduction/flow/index.html). To submit
your change directly to the repository:

- Make sure your code is in line with our
  [coding conventions](#coding-conventions).
- Test your code locally to identify and resolve simple issues by doing
  [validation testing](#validation-testing).
- Submit a
  [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) into the
  main branch. You may add a description of your contribution into [CREDITS.txt](https://github.com/uxlfoundation/oneDPL/blob/main/CREDITS.txt).
- Contributors that would like to open branches directly in the oneDPL repo instead of working via fork may request
  write access to the repository by contacting project maintainers on
  [UXL Foundation Slack](https://slack-invite.uxlfoundation.org/) using the
  [#onedpl](https://uxlfoundation.slack.com/channels/onedpl) channel.

# Coding Conventions

Running clang-format is required, except in the [test folder](https://github.com/uxlfoundation/oneDPL/tree/main/test).

# Validation Testing

The oneDPL test suite is organized in the test directory by the area of code tested.
* `general` - Basic tests of oneDPL policies and SYCL functionality needed
* `general/sycl_iterator` - Tests of oneDPL algorithms on SYCL buffers
* `kt` - Tests of the oneDPL experimental kernel template algorithms that require specific hardware support
* `parallel_api` - Tests of oneDPL Parallel Algorithms
* `pstl_offload` - Tests of support for offload using standard execution policies
* `xpu_api` - Tests of the use of C++ standard APIs in SYCL kernels

oneDPL validation tests can be configured through CMake to run on different devices including CPU and GPU by specifying the oneDPL backend to use and
the device type to be used if the DPC++ backend is being tested. The oneTBB and OpenMP backends for oneDPL can be used when the device type is CPU or HOST.

To run oneDPL validation tests with the oneDPL build system you will need CMake and CTest installed. The following example shows how to build and run tests
using the GPU on your local system using the Intel® oneAPI DPC++ Compiler.

1. Configure the build files by running the following command that creates a build directory named `build_gpu_tests`.
```
cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_STANDARD=17 -DONEDPL_BACKEND=dpcpp -DONEDPL_DEVICE_TYPE=gpu -DCMAKE_BUILD_TYPE=release -B build_gpu_tests
```

2. Build the tests
```
cmake --build build_gpu_tests --target build-onedpl-tests # specify a specific test name (e.g., reduce.pass) to build a single test
```

3. Run the tests from the `build_gpu_tests` directory
```
ctest --output-on-failure --timeout 1200 -R ^reduce.pass$ # Add -R testname (e.g., -R ^reduce.pass$) to run just one test.
```

Before submitting a PR to the oneDPL repository please run the tests that exercise the code you have updated. If you need help identifying those tests please
check with the maintainers on [UXL Foundation Slack](https://slack-invite.uxlfoundation.org/) using the [#onedpl](https://uxlfoundation.slack.com/channels/onedpl) channel
or ask a question through [GitHub issues](https://github.com/uxlfoundation/oneDPL/issues).

For more details on configurations available for oneDPL testing see the [CMake README](https://github.com/uxlfoundation/oneDPL/blob/main/cmake/README.md).
