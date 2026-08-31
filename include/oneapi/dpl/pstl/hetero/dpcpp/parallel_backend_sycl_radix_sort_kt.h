// -*- C++ -*-
//===-- parallel_backend_sycl_radix_sort_kt.h -----------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_RADIX_SORT_KT_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_RADIX_SORT_KT_H

#include "../../../experimental/kt/internal/kt_defs.h"
#include "kt_arch_params.h"

#if !defined(_ONEDPL_ENABLE_SYCL_RADIX_SORT_KT) || !_ONEDPL_ENABLE_SYCL_RADIX_SORT_KT ||                               \
    !defined(_ONEDPL_SYCL_DEVICE_ARCHITECTURE_PRESENT)
// KT radix sort unavailable (no cooperative kernels, no sub_group_mask, or no device architecture
// query to select tuned kernel parameters with); disable dispatch.
#    define _ONEDPL_KT_RADIX_SORT_IN_SORT_ACTIVE 0
#else
#    define _ONEDPL_KT_RADIX_SORT_IN_SORT_ACTIVE 1

#    include <cstdint>
#    include <new>
#    include <utility>
#    include <type_traits>

#    include "sycl_defs.h"
#    include "sycl_forward_progress.h"
#    include "utils_ranges_sycl.h"
#    include "parallel_backend_sycl_utils.h"
#    include "../../utils_ranges.h"
#    include "../../../experimental/kt/kernel_param.h"
#    include "../../../experimental/kt/internal/radix_sort_dispatchers.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{
namespace __kt_radix
{

namespace __syclex = sycl::ext::oneapi::experimental;
namespace __kt_impl = oneapi::dpl::experimental::kt::gpu::__impl;
namespace __dpl_ranges = oneapi::dpl::__ranges;

// Minimum input size below which the legacy radix sort is preferred; the crossover has not been
// benchmarked per architecture yet.
inline constexpr std::size_t __min_size = 1 << 18;

// KT supports inputs strictly smaller than 2^30 (hard limit from radix_sort_utils.h:63).
inline constexpr std::size_t __max_size = std::size_t(1) << 30;

// Architectures with tuned kernel parameters. Data from
// documentation/library_guide/kernel_templates/sycl/radix_sort.rst:298. There is deliberately no
// __default_arch_params entry: an untuned architecture is served by the legacy radix sort instead of by
// a parameter set which has never been measured on it.
using __radix_kt_params = __arch_param_table<
    __arch_params<oneapi::dpl::experimental::kt::kernel_param<28, 512>, __syclex::architecture::intel_gpu_pvc,
                  __syclex::architecture::intel_gpu_pvc_vg>,
    __arch_params<oneapi::dpl::experimental::kt::kernel_param<10, 512>, __syclex::architecture::intel_gpu_bmg_g21>>;

// Size- and device-based eligibility, checked before the architecture lookup. Size bounds come first to
// avoid device queries for ineligible inputs.
inline bool
__is_eligible(const sycl::queue& __q, std::size_t __n)
{
    if (__n < __min_size || __n >= __max_size)
        return false;

    const sycl::device __device = __q.get_device();
    // Correctness gate: the KT cooperative kernels require concurrent root-group forward-progress; without it the
    // KT path throws rather than degrading gracefully.
    return __device.is_gpu() && __supports_concurrent_root_group_progress(__device);
}

enum class __kt_sort_shape
{
    __keys_only,
    __by_key,
    __none
};

// Shape detection: relies on the projection type to distinguish keys-only (identity) from by-key (__pattern_sort_by_key_fn).
template <typename _Range, typename _Proj>
struct __kt_radix_sort_shape_impl
{
    static constexpr __kt_sort_shape value = __kt_sort_shape::__none;
};

// Keys-only: projection is identity and range is a compatible view.
template <typename _Range>
struct __kt_radix_sort_shape_impl<_Range, oneapi::dpl::identity>
{
    static constexpr __kt_sort_shape value =
        __dpl_ranges::__is_contiguous_backed_view<_Range> ? __kt_sort_shape::__keys_only : __kt_sort_shape::__none;
};

// By-key: projection is __pattern_sort_by_key_fn, range is a zip_view of two compatible views.
template <typename _V1, typename _V2>
struct __kt_radix_sort_shape_impl<__dpl_ranges::zip_view<_V1, _V2>, oneapi::dpl::__internal::__pattern_sort_by_key_fn>
{
    static constexpr __kt_sort_shape value =
        (__dpl_ranges::__is_contiguous_backed_view<_V1> && __dpl_ranges::__is_contiguous_backed_view<_V2>)
            ? __kt_sort_shape::__by_key
            : __kt_sort_shape::__none;
};

template <typename _Range, typename _Proj>
inline constexpr __kt_sort_shape __kt_radix_sort_shape = __kt_radix_sort_shape_impl<_Range, _Proj>::value;

// Runs the KT radix sort with the kernel parameters tuned for the queue's device, storing the resulting
// event in __event. Returns false without submitting anything if the input, the device, or its
// architecture is not served by the KT path, or if KT could not allocate its temporary storage; the
// caller then uses the legacy radix sort.
template <bool __is_ascending, __kt_sort_shape __shape, typename _Range>
bool
__try_parallel_kt_radix_sort(sycl::queue __q, _Range&& __rng, sycl::event& __event)
{
    static_assert(__shape != __kt_sort_shape::__none);

    if (!__is_eligible(__q, __rng.size()))
        return false;

    auto __sort = [&](auto __param) {
        auto __pack = [&]() {
            if constexpr (__shape == __kt_sort_shape::__keys_only)
            {
                return __kt_impl::__range_pack{__dpl_ranges::__normalize_contiguous_view(__rng)};
            }
            else // __by_key: KT consumes keys and values as separate ranges, so decompose the zip_view.
            {
                auto __base = __rng.base();
                return __kt_impl::__range_pack{__dpl_ranges::__normalize_contiguous_view(std::get<0>(__base)),
                                               __dpl_ranges::__normalize_contiguous_view(std::get<1>(__base))};
            }
        }();
        __event = __kt_impl::__radix_sort<__is_ascending, /*__radix_bits=*/8, /*__in_place=*/true>(
            __kt_impl::__sycl_tag{}, __q, __pack, __pack, __param);
    };

    try
    {
        return __radix_kt_params::__try_dispatch(__q.get_device(), __sort);
    }
    catch (const std::bad_alloc&)
    {
        // KT could not allocate its temporary storage.
        return false;
    }
}

} // namespace __kt_radix
} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_RADIX_SORT_KT_H
