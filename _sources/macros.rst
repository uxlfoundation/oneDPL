Macros
######

Version Macros
==============
Use these macros to get the current version of |onedpl_long| (|onedpl_short|).

================================= ==============================
Macro                             Description
================================= ==============================
``ONEDPL_SPEC_VERSION``           The version of the `specification
                                  <https://uxlfoundation.github.io/oneAPI-spec/spec/elements/oneDPL/source/>`_
                                  that the implementation is compliant with.
--------------------------------- ------------------------------
``ONEDPL_VERSION_MAJOR``          A decimal number for the major version of the library.
--------------------------------- ------------------------------
``ONEDPL_VERSION_MINOR``          A decimal number for the minor version.
--------------------------------- ------------------------------
``ONEDPL_VERSION_PATCH``          A decimal number for the patch.
--------------------------------- ------------------------------
``_PSTL_VERSION``                 The version of LLVM PSTL code used in |onedpl_short|.

                                  The value is a decimal numeral of the form ``xxyyz``
                                  where ``xx`` is the major version number, ``yy`` is the
                                  minor version number and ``z`` is the patch number.
--------------------------------- ------------------------------
``_PSTL_VERSION_MAJOR``           ``_PSTL_VERSION/1000``: The major version number.
--------------------------------- ------------------------------
``_PSTL_VERSION_MINOR``           ``(_PSTL_VERSION % 1000) / 10``: The minor version number.
--------------------------------- ------------------------------
``_PSTL_VERSION_PATCH``           ``_PSTL_VERSION % 10``: The patch number.
================================= ==============================

.. _feature-macros:

Feature Macros
==============
Use these macros to test presence of specific |onedpl_short| functionality.

================================== ===============================================
Macro                              Macro values and the functionality
================================== ===============================================
``ONEDPL_HAS_RANDOM_NUMBERS``      Pseudo-random number generators and distributions.

                                   * ``202409L`` - added support of comparison and I/O stream operators and an experimental Philox engine
---------------------------------- -----------------------------------------------
``ONEDPL_HAS_RANGE_ALGORITHMS``    Parallel range algorithms.

                                   * ``202409L`` - see :ref:`available algorithms <range-algorithms-202409L>`.
                                   * ``202505L`` - see :ref:`available algorithms <range-algorithms-202505L>`.
================================== ===============================================

Additional Macros
==================
Use these macros to control aspects of |onedpl_short| usage. You can set them in your program code
before including |onedpl_short| headers.

================================== ==============================
Macro                              Description
================================== ==============================
``PSTL_USE_NONTEMPORAL_STORES``    This macro enables the use of ``#pragma vector nontemporal``
                                   for write-only data when algorithms such as ``std::copy``, ``std::fill``, etc.,
                                   are executed with unsequenced policies.
                                   For further details about the pragma, see the |vector_pragma|_.
                                   If the macro evaluates to a non-zero value,
                                   the use of ``#pragma vector nontemporal`` is enabled.
                                   By default, the macro is not defined.

                                   Using this macro may have the same effect on the implementation of parallel
                                   algorithms in the C++ standard libraries of GCC and LLVM.
---------------------------------- ------------------------------
``PSTL_USAGE_WARNINGS``            This macro enables Parallel STL to
                                   emit compile-time messages, such as warnings
                                   about an algorithm not supporting a certain execution policy.
                                   When set to 1, the macro allows the implementation to emit
                                   usage warnings. When the macro is not defined (by default)
                                   or evaluates to zero, usage warnings are disabled.

                                   Using this macro may have the same effect on the implementation of parallel
                                   algorithms in the C++ standard libraries of GCC and LLVM.
---------------------------------- ------------------------------
``ONEDPL_USE_TBB_BACKEND``         This macro controls the use of |onetbb_long| or |tbb_long| for parallel
                                   execution policies (``par`` and ``par_unseq``).

                                   When the macro evaluates to a non-zero value, or when it is not defined (by default)
                                   and no other parallel backends are explicitly chosen, algorithms with parallel policies
                                   are executed using the |onetbb_short| or |tbb_short| library.
                                   Setting the macro to 0 disables use of TBB API for parallel execution and is recommended
                                   for code that should not depend on the presence of the |onetbb_short| or |tbb_short| library.

                                   If all parallel backends are disabled by setting respective macros to 0, algorithms
                                   with parallel policies are executed sequentially by the calling thread.
---------------------------------- ------------------------------
``ONEDPL_USE_OPENMP_BACKEND``      This macro controls the use of OpenMP* for parallel execution policies (``par`` and ``par_unseq``).

                                   When the macro evaluates to a non-zero value, algorithms with parallel policies are executed
                                   using OpenMP unless the TBB backend is explicitly enabled (that is, the TBB backend takes
                                   precedence over the OpenMP backend).
                                   When the macro is not defined (by default) and no other parallel backends are chosen,
                                   a dedicated compiler option to enable OpenMP (such as ``-fopenmp``) also enables its use
                                   for algorithms with parallel policies.
                                   Setting the macro to 0 disables use of OpenMP for parallel execution.

                                   If all parallel backends are disabled by setting respective macros to 0, algorithms
                                   with parallel policies are executed sequentially by the calling thread.
---------------------------------- ------------------------------
``ONEDPL_USE_DPCPP_BACKEND``       This macro enables the use of device execution policies.

                                   When the macro is not defined (default),
                                   device policies are enabled only if SYCL support can be detected;
                                   otherwise, they are disabled.
                                   If the macro is set to a non-zero value, device policies are enabled unconditionally.
                                   Setting the macro to 0 disables device policies.

                                   When device policies are disabled, no SYCL dependency is introduced,
                                   and their usage will lead to compilation errors.
---------------------------------- ------------------------------
``ONEDPL_USE_PREDEFINED_POLICIES`` This macro enables the use of predefined device policy objects,
                                   such as ``dpcpp_default`` and ``dpcpp_fpga``. When the macro is not defined (by default)
                                   or evaluates to non-zero, predefined policies objects can be used.
                                   When the macro is set to 0, predefined policies objects and make functions
                                   without arguments (``make_device_policy()`` and ``make_fpga_policy()``) are not available.
---------------------------------- ------------------------------
``ONEDPL_ALLOW_DEFERRED_WAITING``  This macro allows waiting for completion of certain algorithms executed with
                                   device policies to be deferred. (Disabled by default.)

                                   When the macro evaluates to non-zero, a call to a oneDPL algorithm with
                                   a device policy might return before the computation completes on the device.

                                   .. Warning:: Before accessing data produced or modified by the call, waiting
                                      for completion of all tasks in the corresponding SYCL queue is required;
                                      otherwise, the program behavior is undefined.
---------------------------------- ------------------------------
``ONEDPL_FPGA_DEVICE``             Use this macro to build your code containing |onedpl_short| parallel
                                   algorithms for FPGA devices. (Disabled by default.)
---------------------------------- ------------------------------
``ONEDPL_FPGA_EMULATOR``           Use this macro to build your code containing Parallel STL
                                   algorithms for FPGA emulation device. (Disabled by default.)

                                   .. Note:: Define ``ONEDPL_FPGA_DEVICE`` and ``ONEDPL_FPGA_EMULATOR`` macros in the same
                                      application to run on a FPGA emulation device.
                                      Define only the ``ONEDPL_FPGA_DEVICE`` macro to run on a FPGA hardware device.
================================== ==============================
