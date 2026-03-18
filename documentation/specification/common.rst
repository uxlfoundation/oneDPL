.. SPDX-FileCopyrightText: 2019-2022 Intel Corporation
.. SPDX-FileCopyrightText: Contributors to the oneAPI Specification project.
..
.. SPDX-License-Identifier: CC-BY-4.0

General Information
-------------------

This section describes the most general functionality of |dpl_full_name| (oneDPL)
such as namespaces, versioning, etc.

Namespaces
++++++++++

oneDPL uses ``namespace oneapi::dpl`` and a shorter variant ``namespace dpl`` for all
functionality including parallel algorithms, oneDPL execution policies, etc.
For the subset of the standard C++ library for kernels, the standard class
and function names are also aliased in ``namespace oneapi::dpl``.

oneDPL uses nested namespaces for the functionality aligned with the C++ standard.
The names of those namespaces are the same as for their analogues in ``namespace std``.
[*Example*: oneDPL execution policies are provided in ``namespace oneapi::dpl::execution``.  -- *end example*]

Header Files
++++++++++++

The oneDPL header files are provided under the ``oneapi/dpl/`` include path,
which must explicitly precede oneDPL header file names in ``#include`` directives.

Following the naming convention for the C++ standard library header files,
the oneDPL header files have no extension suffix in their names.

A oneDPL header file automatically includes the C++ standard library header file
which name matches that of the oneDPL header file without the include path prefix.
[*Example*: ``#include <oneapi/dpl/version>`` automatically includes ``<version>`` 
if that is available. -- *end example*]

Version Information
+++++++++++++++++++

A oneDPL implementation must define the following preprocessor macro representing
the version of the oneDPL specification that the implementation is compliant with.

.. code:: cpp

  // Defined in <oneapi/dpl/version>

  #define ONEDPL_SPEC_VERSION /*implementation-defined*/

The ``ONEDPL_SPEC_VERSION`` macro must be defined to the decimal literal whose value equals to
*major-spec-version* * 100 + *minor-spec-version*, where *major-spec-version* and *minor-spec-version*
are the major and the minor versions of the latest fully supported specification.
[*Example*: ``#define ONEDPL_SPEC_VERSION 104`` for the oneDPL specification 1.4.  -- *end example*]
