.. SPDX-FileCopyrightText: 2019-2022 Intel Corporation
.. SPDX-FileCopyrightText: Contributors to the oneAPI Specification project.
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _oneDPL-section:

======
oneDPL
======

This is the specification of |dpl_full_name|, part of the oneAPI Specification version |dpl_version|.

The |dpl_full_name| (oneDPL) provides the functionality
specified in the `C++ standard`_, with extensions to support data parallelism
and offloading to devices, and with extensions to simplify its usage for 
implementing data parallel algorithms.

.. note::
    Unless specified otherwise, in this document the `C++ standard`_ refers to
    ISO/IEC 14882:2017 Programming languages - C++, commonly known as C++17. 

The library is comprised of the following components:

- :doc:`Parallel API <parallel_api>`:

  * Parallel algorithms, complemented with execution
    policies and companion APIs for running on oneAPI devices.

  * An additional set of library classes and functions that
    are known to be useful in practice but are not yet included into
    C++ or SYCL specifications. 

- :doc:`SYCL Kernels API <sycl_kernels_api>`:

  * A subset of the C++ standard library which can be used with
    buffers and data parallel kernels.

  * Support of random number generation including engines and distributions.

  * Various utilities in addition to the C++ standard functionality.

.. toctree::
   :titlesonly:

   common.rst
   parallel_api.rst
   sycl_kernels_api.rst

.. _`C++ Standard`: https://isocpp.org/std/the-standard
