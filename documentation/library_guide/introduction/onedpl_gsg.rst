Get Started with the |onedpl_long|
##################################

|onedpl_long| (|onedpl_short|) works with the |dpcpp_cpp_with_gsg_link|_
to provide high-productivity APIs to developers, which can minimize SYCL*
programming efforts across devices for high performance parallel applications.

|onedpl_short| consists of the following components:

* Parallel API
* API for SYCL Kernels
* Macros


For general information about |onedpl_short|, visit the `oneDPL GitHub* repository <https://github.com/uxlfoundation/oneDPL>`_,
or visit the |onedpl_library_guide|_ and the `Intel® oneAPI DPC++ Library main page
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-library.html>`_.

Quick Start
===========

Installation
------------

Visit the |onedpl_short| `Release Notes
<https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-dpcpp-library-release-notes.html>`_
page for:

* Where to Find the Release
* Overview
* New Features
* Fixed Issues
* Known Issues and Limitations

Install the `Intel® oneAPI Base Toolkit (Base Kit) <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html>`_
to use |onedpl_short|.

To use Parallel API, include the corresponding header files in your source code.

All |onedpl_short| header files are in the ``oneapi/dpl`` directory. Use ``#include <oneapi/dpl/…>`` to include them.
|onedpl_short| uses the namespace ``oneapi::dpl`` for most its classes and functions.

To use tested C++ standard APIs, you need to include the corresponding C++ standard header files
and use the ``std`` namespace.

CMake Support
-------------
`CMake <https://cmake.org/cmake/help/latest/index.html>`_ generates build scripts which can then be used
to build and link your application. |onedpl_short| can be added to your project via CMake. 

A simple example for Linux is provided below. For more detailed usage and options including details specific to Windows,
please look to the |dpcpp_cmake_support|_.

Simple Example CMake File
*************************
To use |onedpl_short| with CMake, create a CMakeLists.txt file for your project's base directory and use
`find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_
and `target_link_libraries <https://cmake.org/cmake/help/latest/command/target_link_libraries.html>`_ to add oneDPL.
For example:

.. code:: cpp

  project(Foo)
  add_executable(foo foo.cpp)
  
  # Search to find oneDPL
  find_package(oneDPL REQUIRED)
  
  # Connect oneDPL to foo
  target_link_libraries(foo oneDPL)

Simple Example CMake Invocation
*******************************
The following is an example CMake invocation which generates build scripts for the project in the parent directory: 

.. code:: cpp

  mkdir build && cd build
  cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=release ..

Example Build Command
*********************
Once build scripts have been generated for your desired configuration following the instruction above, a `build command
<https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project>`_ can be issued to build your project:

.. code:: cpp

  cmake --build .

pkg-config Support
------------------

The pkg-config program is used to retrieve information about your installed libraries, and
to compile and link against one or more libraries.

Use pkg-config with |onedpl_short|
**********************************

Use pkg-config with the ``--cflags`` flag to get the include path to the oneDPL directory:

.. code:: cpp

  icpx -fsycl foo.cpp $(pkg-config --cflags dpl)
  
The ``--msvc-syntax`` flag is required when you use a Microsoft Visual C++* compiler.
This flag converts your compiling and linking flags to the appropriate form:

.. code:: cpp

  icpx -fsycl foo.cpp $(pkg-config --msvc-syntax --cflags dpl)

.. note::
  Use the pkg-config tool to get rid of large hard-coded paths and make compilation more portable.


Usage Examples
--------------

|onedpl_short| sample code is available from the
`oneAPI GitHub samples repository <https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDPL>`_.
Each sample includes a readme with build instructions.

\<oneapi/dpl/random\> Header Usage Example
******************************************

This example illustrates |onedpl_short| random number generator usage.
The sample below shows you how to create an random number generator engine object (the source of pseudo-randomness),
a distribution object (specifying the desired probability distribution), and how to generate
the random numbers themselves. Random number generation is performed in a vectorized manner
to improve the speed of your computations.

This example performs its computations on your default SYCL device. You can set the
``SYCL_DEVICE_TYPE`` environment variable to CPU or GPU.

.. code:: cpp

    template<int VecSize>
    void random_fill(float* usmptr, std::size_t n) {
        auto zero = oneapi::dpl::counting_iterator<std::size_t>(0);

        std::for_each(oneapi::dpl::execution::dpcpp_default,
            zero, zero + n/VecSize,
            [usmptr](std::size_t i) {
                auto offset = i * VecSize;

                oneapi::dpl::minstd_rand_vec<VecSize> engine(seed, offset);
                oneapi::dpl::uniform_real_distribution<sycl::vec<float, VecSize>> distr;

                auto res = distr(engine);
                res.store(i, sycl::global_ptr<float>(usmptr));
            });
    }

Pi Benchmark Usage Example
**************************

This example uses a Monte Carlo method to estimate the value of π.
The basic idea is to generate random points within a square, and to check what
fraction of these random points lie in a quarter-circle inscribed within that square.
The expected value is the ratio of the areas of the quarter-circle and the square (π/4).
You can take the observed fraction of points in the quarter-circle as an estimate of π/4.

This example shows you how to create an random number generator engine object (the source of pseudo-randomness),
a distribution object (specifying the desired probability distribution), generate the
random numbers themselves, and then perform a reduction to count quantity of points that
fit into the square *S*. Random number generation is performed in scalar manner to simplify your code.


.. figure:: images/pi_benchmark.png
   :alt: An image of pi chart.

.. code:: cpp

    float estimated_pi;
    {
        sycl::queue q(sycl::gpu_selector_v);
        auto policy = oneapi::dpl::execution::make_device_policy(q);

        float sum = std::transform_reduce( policy,
                                          oneapi::dpl::counting_iterator<int>(0),
                                          oneapi::dpl::counting_iterator<int>(N),
                                          0.0f,
                                          std::plus<float>{},
                                          [=](int n){
                                              float local_sum = 0.0f;
                                              oneapi::dpl::minstd_rand engine(SEED, n * ITER * 2);
                                              oneapi::dpl::uniform_real_distribution<float> distr;
                                              for(int i = 0; i < ITER; ++i) {
                                                  float x = distr(engine);
                                                  float y = distr(engine);
                                                  if (x * x + y * y <= 1.0)
                                                      local_sum += 1.0;
                                              }
                                              return local_sum / (float)ITER;
                                          }
        );
        estimated_pi = 4.0f * (float)sum / N;
    }


Find More
=========

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Resource Link
     - Description
   * - |onedpl_library_guide|_
     - Refer to the |onedpl_short| guide for  more in depth information.
   * - `System Requirements <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_
     - Check system requirements before you install |onedpl_short|.
   * - `Intel® oneAPI DPC++ Library Release Notes
       <https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-dpcpp-library-release-notes.html>`_
     - Check the release notes to learn about updates in the latest release.
   * - `oneDPL Samples <https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDPL>`_
     - Learn how to use |onedpl_short| with samples.
   * - `oneAPI Samples Catalog <https://oneapi-src.github.io/oneAPI-samples/>`_
     - Explore the complete list of oneAPI code samples in the oneAPI Samples Catalog (GitHub*).
       These samples were designed to help you develop, offload, and optimize multiarchitecture applications targeting CPUs, GPUs, and FPGAs.
