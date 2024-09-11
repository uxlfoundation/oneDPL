CMake Support
#############

General Usage
=============
`CMake <https://cmake.org/cmake/help/latest/index.html>`_ is a cross-platform build system generator. It can be used to generate build scripts which can then be used to build and link your application.

``oneDPLConfig.cmake`` and ``oneDPLConfigVersion.cmake`` are distributed with |onedpl_short|. These files allow integration of |onedpl_short| into user projects with the `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_ command. Successful invocation of ``find_package(oneDPL <options>)`` creates imported target `oneDPL` that can be passed to the `target_link_libraries <https://cmake.org/cmake/help/latest/command/target_link_libraries.html>`_ command.

Some useful CMake variables (`here <https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html>`_ you can find a full list of CMake variables for the latest version):

- `CMAKE_CXX_COMPILER <https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html>`_ - C++ compiler used for build: ``CMAKE_CXX_COMPILER=dpcpp``.
- `CMAKE_BUILD_TYPE <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_ - build type that affects optimization level and debug options; For example: ``CMAKE_BUILD_TYPE=Release``.
- `CMAKE_CXX_STANDARD <https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html>`_ - C++ standard: ``CMAKE_CXX_STANDARD=17``.

Requirements
============
The minimal supported CMake version for |onedpl_short| is 3.11 on Linux and 3.20 on Windows.

The supported `CMake Generator <https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#ninja-generators>`_
for Linux is `Unix Makefiles <https://cmake.org/cmake/help/latest/generator/Unix%20Makefiles.html>`_ (default).
In the Windows environment, the supported generator is `Ninja <https://cmake.org/cmake/help/latest/generator/Ninja.html>`_
which may be specified via ``-GNinja`` as described in the |dpcpp_cmake_support|_.

|onedpl_short| Backend Options
==============================

Backend for Parallel Execution Policies (par and par_unseq)
-----------------------------------------------------------
The |onedpl_short| backend for parallel execution policies controls how algorithms with parallel execution policies (``par`` or ``par_unseq``) are implemented. This option is controlled via the ``ONEDPL_PAR_BACKEND`` setting.

+--------------------+--------+--------+--------+
| ONEDPL_PAR_BACKEND | oneTBB | OpenMP | Serial |
+====================+========+========+========+
| [not set]          |     oneDPL heuristics    |
+--------------------+--------+--------+--------+
| tbb                |   X    |        |        |
+--------------------+--------+--------+--------+
| openmp             |        |    X   |        |
+--------------------+--------+--------+--------+
| serial             |        |        |    X   |
+--------------------+--------+--------+--------+

The |onedpl_short| heuristics are the following: the first suitable backend is chosen among ``oneTBB``, ``OpenMP`` and ``Serial``, in that order. If ``ONEDPL_PAR_BACKEND`` is specified, but the selected backend is not found or unsupported, |onedpl_short| is considered not found (``oneDPL_FOUND=False``).

Backend for Device Execution Policies
-----------------------------------------------------------
The |onedpl_short| backend for device execution policies controls if device policies are enabled.

+-------------------+
|       DPCPP       |
+===================+
| oneDPL heuristics |
+-------------------+

The heuristics are the following: ``DPCPP`` backend is enabled if the compiler supports ``-fsycl`` option and SYCL headers are available.

For more details on |onedpl_short| backends, see :doc:`Execution Policies <parallel_api/execution_policies>`.

Example CMake File
==================
To use |onedpl_short| with CMake, you must create a ``CMakeLists.txt`` file for your project and add |onedpl_short|. This file should be placed in the project's base directory. Below is an example ``CMakeLists.txt`` file:

.. code:: cpp

  if (CMAKE_HOST_WIN32)
    find_package(oneDPLWindowsIntelLLVM)
  endif()

  project(Foo)
  add_executable(foo foo.cpp)
  
  # Search to find oneDPL
  find_package(oneDPL REQUIRED)
  
  # Connect oneDPL to foo
  target_link_libraries(foo oneDPL)

.. note::
  On Windows, some workarounds may be required to use ``icx[-cl]`` successfully with |onedpl_short|. We recommend updating to the most recent version of CMake to minimize the workarounds required for successful use. A CMake package has been provided, ``oneDPLWindowsIntelLLVM``, to provide the necessary workarounds to enable support for ``icx[-cl]`` on Windows with CMake versions 3.20 and greater. Some workarounds are provided for ``icpx``, but it is not fully supported on Windows at this time. To use this package, please add ``find_package(oneDPLWindowsIntelLLVM)`` to your CMake file *before* you call ``project()``.


Example CMake Invocation
========================
After creating a ``CMakeLists.txt`` file for your project, you may use a command line CMake invocation to generate build scripts.

Below is an example ``Linux`` CMake invocation which generates Unix makefiles for the project with the ``icpx`` compiler, ``oneTBB`` backend and ``Release`` build type:

.. code:: cpp

  mkdir build && cd build
  cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=release -DONEDPL_PAR_BACKEND=tbb ..

Below is an example ``Windows`` CMake invocation which generates ``Ninja`` build scripts (see the Requirements Section) for the project in the parent directory with the ``icx`` compiler, ``OpenMP`` backend and ``debug`` build type:

.. code:: cpp

  mkdir build && cd build
  cmake -GNinja -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=debug -DONEDPL_PAR_BACKEND=openmp ..

Both of these examples assume the starting working directory is the project's base directory which contains ``CMakeLists.txt``. The build scripts are generated in a newly created ``build`` directory.


Example Build Command
=====================
Once build scripts have been generated for your desired configuration following the instruction above, a `build command <https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project>`_ can be issued to build your project:

.. code:: cpp

  cmake --build .

This example assumes the starting working directory is in the directory which contains the CMake generated build scripts, ``build``, if following the instructions above.
