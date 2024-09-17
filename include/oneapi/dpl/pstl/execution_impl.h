// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_EXECUTION_IMPL_H
#define _ONEDPL_EXECUTION_IMPL_H

#include <iterator>
#include <type_traits>

#include "parallel_backend.h"
#include "execution_defs.h"
#include "iterator_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// backend selector with tags
//------------------------------------------------------------------------

#if _ONEDPL_PAR_BACKEND_TBB
using __par_backend_tag = __tbb_backend_tag;
#elif _ONEDPL_PAR_BACKEND_OPENMP
using __par_backend_tag = __omp_backend_tag;
#elif _ONEDPL_PAR_BACKEND_SERIAL
using __par_backend_tag = __serial_backend_tag;
#else
#    error "Parallel backend was not specified"
#endif

template <class _IsVector>
struct __serial_tag
{
    using __is_vector = _IsVector;
};

template <class _IsVector>
struct __parallel_tag
{
    using __is_vector = _IsVector;
    using __backend_tag = __par_backend_tag;
};

struct __parallel_forward_tag
{
    using __is_vector = ::std::false_type;
    using __backend_tag = __par_backend_tag;
};

//----------------------------------------------------------
// __select_backend (for the host policies)
//----------------------------------------------------------

template <class _IsVector, class... _IteratorTypes>
using __parallel_policy_tag_selector_t = ::std::conditional_t<
    __internal::__is_random_access_iterator_v<_IteratorTypes...>, __parallel_tag<_IsVector>,
    ::std::conditional_t<__is_forward_iterator_v<_IteratorTypes...>, __parallel_forward_tag, __serial_tag<_IsVector>>>;

template <class... _IteratorTypes>
__serial_tag<std::false_type>
__select_backend(oneapi::dpl::execution::sequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__serial_tag<__internal::__is_random_access_iterator<_IteratorTypes...>>
__select_backend(oneapi::dpl::execution::unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__parallel_policy_tag_selector_t<std::false_type, _IteratorTypes...>
__select_backend(oneapi::dpl::execution::parallel_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__parallel_policy_tag_selector_t<__internal::__is_random_access_iterator<_IteratorTypes...>, _IteratorTypes...>
__select_backend(oneapi::dpl::execution::parallel_unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

} //__internal

#if _ONEDPL_CPP20_RANGES_PRESENT
namespace __ranges
{

inline ::oneapi::dpl::__internal::__serial_tag<std::false_type>
__select_backend(oneapi::dpl::execution::sequenced_policy)
{
    return {};
}

inline ::oneapi::dpl::__internal::__serial_tag<std::true_type> //vectorization allowed
__select_backend(oneapi::dpl::execution::unsequenced_policy)
{
    return {};
}

inline ::oneapi::dpl::__internal::__parallel_tag<std::false_type>
__select_backend(oneapi::dpl::execution::parallel_policy)
{
    return {};
}

inline ::oneapi::dpl::__internal::__parallel_tag<std::true_type> //vectorization allowed
__select_backend(oneapi::dpl::execution::parallel_unsequenced_policy)
{
    return {};
}

} //__ranges

#endif //_ONEDPL_CPP20_RANGES_PRESENT

namespace __internal
{
//----------------------------------------------------------
// __is_serial_tag, __is_serial_tag_v
//----------------------------------------------------------

template <class _Tag>
struct __is_serial_tag : ::std::false_type
{
};

template <class _IsVector>
struct __is_serial_tag<__serial_tag<_IsVector>> : ::std::true_type
{
};

template <class _Tag>
inline constexpr bool __is_serial_tag_v = __is_serial_tag<_Tag>::value;

//----------------------------------------------------------
// __is_parallel_forward_tag, __is_parallel_forward_tag_v
//----------------------------------------------------------

template <class _Tag>
struct __is_parallel_forward_tag : ::std::false_type
{
};

template <>
struct __is_parallel_forward_tag<__parallel_forward_tag> : ::std::true_type
{
};

template <class _Tag>
inline constexpr bool __is_parallel_forward_tag_v = __is_parallel_forward_tag<_Tag>::value;

//----------------------------------------------------------
// __is_parallel_tag, __is_parallel_tag_v
//----------------------------------------------------------

template <class _Tag>
struct __is_parallel_tag : ::std::false_type
{
};

template <class _IsVector>
struct __is_parallel_tag<__parallel_tag<_IsVector>> : ::std::true_type
{
};

template <class _Tag>
inline constexpr bool __is_parallel_tag_v = __is_parallel_tag<_Tag>::value;

//----------------------------------------------------------
// __is_host_dispatch_tag_v
//----------------------------------------------------------

template <class _Tag>
inline constexpr bool __is_host_dispatch_tag_v =
    __is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag> || __is_parallel_tag_v<_Tag>;

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXECUTION_IMPL_H
