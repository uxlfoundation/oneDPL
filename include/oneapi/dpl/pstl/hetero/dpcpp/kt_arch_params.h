// -*- C++ -*-
//===-- kt_arch_params.h --------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_ARCH_PARAMS_H
#define _ONEDPL_KT_ARCH_PARAMS_H

#include "sycl_defs.h"

#if defined(SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE)
#    define _ONEDPL_SYCL_DEVICE_ARCHITECTURE_PRESENT 1
#endif

#if _ONEDPL_SYCL_DEVICE_ARCHITECTURE_PRESENT

#    include <cstddef>
#    include <utility>

#    include "../../../experimental/kt/kernel_param.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

namespace __syclex = sycl::ext::oneapi::experimental;

// Kernel templates are tuned per architecture, but a kernel_param is a compile-time entity while the
// architecture of the device a call runs on is only known at run time. The types below express the
// association as a compile-time table which is walked at run time: every mainline API backed by a
// kernel template declares one table listing the architectures it has tuned parameters for, and the
// table instantiates the kernel once per distinct parameter set.
//
//   using __my_params = __arch_param_table<
//       __arch_params<kernel_param<28, 512>, __syclex::architecture::intel_gpu_pvc,
//                                            __syclex::architecture::intel_gpu_pvc_vg>,
//       __arch_params<kernel_param<10, 512>, __syclex::architecture::intel_gpu_bmg_g21>>;
//
// Architectures are named with the SYCL device architecture extension's enumerators directly, so no
// oneDPL-side architecture list has to be kept in sync with the hardware the extension knows about.

// A set of architectures sharing one tuned kernel_param.
template <typename _KernelParam, __syclex::architecture... _Archs>
struct __arch_params
{
    static_assert(sizeof...(_Archs) > 0, "an __arch_params entry must name at least one architecture");

    using __param = _KernelParam;
    static constexpr bool __is_catch_all = false;

    static bool
    __matches(__syclex::architecture __a)
    {
        return ((__a == _Archs) || ...);
    }
};

// A fallback kernel_param used for any architecture not named by a preceding entry. Tables which omit
// it reject unrecognized architectures instead, leaving the caller to use its non-KT implementation.
template <typename _KernelParam>
struct __default_arch_params
{
    using __param = _KernelParam;
    static constexpr bool __is_catch_all = true;

    static bool
    __matches(__syclex::architecture)
    {
        return true;
    }
};

template <typename... _Entries>
constexpr bool
__catch_all_entry_is_last()
{
    // The trailing false keeps the array non-empty for a table without entries.
    constexpr bool __flags[] = {_Entries::__is_catch_all..., false};
    for (std::size_t __i = 0; __i + 1 < sizeof...(_Entries); ++__i)
        if (__flags[__i])
            return false;
    return true;
}

// An ordered list of __arch_params entries, optionally terminated by a __default_arch_params entry.
template <typename... _Entries>
struct __arch_param_table
{
    static_assert(__catch_all_entry_is_last<_Entries...>(),
                  "a __default_arch_params entry must be last: entries after it are unreachable");

    // Invokes __f with a default-constructed kernel_param of the first entry matching __a and returns
    // true. Returns false without invoking __f if no entry matches.
    template <typename _F>
    static bool
    __try_dispatch(__syclex::architecture __a, _F&& __f)
    {
        return (__try_entry<_Entries>(__a, __f) || ...);
    }

    // Convenience overload for the common case of dispatching on the architecture of a device.
    template <typename _F>
    static bool
    __try_dispatch(const sycl::device& __device, _F&& __f)
    {
        return __try_dispatch(__device.get_info<__syclex::info::device::architecture>(), std::forward<_F>(__f));
    }

  private:
    template <typename _Entry, typename _F>
    static bool
    __try_entry(__syclex::architecture __a, _F& __f)
    {
        if (!_Entry::__matches(__a))
            return false;
        __f(typename _Entry::__param{});
        return true;
    }
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SYCL_DEVICE_ARCHITECTURE_PRESENT

#endif // _ONEDPL_KT_ARCH_PARAMS_H
