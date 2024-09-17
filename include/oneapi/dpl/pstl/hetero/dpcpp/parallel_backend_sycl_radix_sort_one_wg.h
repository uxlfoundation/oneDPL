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

#ifndef _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H
#define _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

//The file is an internal file and the code of that file is included by a major file into the following namespaces:
//namespace oneapi
//{
//namespace dpl
//{
//namespace __par_backend_hetero
//{

template <typename... _Name>
class __radix_sort_one_wg_kernel;

template <typename _KernelNameBase, uint16_t __wg_size = 256 /*work group size*/, uint16_t __block_size = 16,
          std::uint32_t __radix = 4, bool __is_asc = true>
struct __subgroup_radix_sort
{
    template <typename _RangeIn, typename _Proj>
    auto
    operator()(sycl::queue __q, _RangeIn&& __src, _Proj __proj)
    {
        using __wg_size_t = ::std::integral_constant<::std::uint16_t, __wg_size>;
        using __block_size_t = ::std::integral_constant<::std::uint16_t, __block_size>;
        using __call_0_t = ::std::integral_constant<::std::uint16_t, 0>;
        using __call_1_t = ::std::integral_constant<::std::uint16_t, 1>;
        using __call_2_t = ::std::integral_constant<::std::uint16_t, 2>;

        using _SortKernelLoc = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __radix_sort_one_wg_kernel<_KernelNameBase, __wg_size_t, __block_size_t, __call_0_t>>;
        using _SortKernelPartGlob = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __radix_sort_one_wg_kernel<_KernelNameBase, __wg_size_t, __block_size_t, __call_1_t>>;
        using _SortKernelGlob = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __radix_sort_one_wg_kernel<_KernelNameBase, __wg_size_t, __block_size_t, __call_2_t>>;

        using _KeyT = oneapi::dpl::__internal::__value_t<_RangeIn>;
        //check SLM size
        const auto __SLM_available = __check_slm_size<_KeyT>(__q, __src.size());
        if (__SLM_available.first && __SLM_available.second)
            return __one_group_submitter<_SortKernelLoc>()(__q, ::std::forward<_RangeIn>(__src), __proj,
                                                           ::std::true_type{} /*SLM*/, ::std::true_type{} /*SLM*/);
        if (__SLM_available.second)
            return __one_group_submitter<_SortKernelPartGlob>()(__q, ::std::forward<_RangeIn>(__src), __proj,
                                                                ::std::false_type{} /*No SLM*/,
                                                                ::std::true_type{} /*SLM*/);
        return __one_group_submitter<_SortKernelGlob>()(__q, ::std::forward<_RangeIn>(__src), __proj,
                                                            ::std::false_type{} /*No SLM*/, ::std::false_type{} /*No SLM*/);
    }

  private:
    template <typename _KeyT, typename>
    class _TempBuf;

    template <typename _KeyT>
    class _TempBuf<_KeyT, ::std::true_type /*shared local memory buffer*/>
    {
        uint16_t __buf_size;

      public:
        _TempBuf(uint16_t __n) : __buf_size(__n) {}
        auto
        get_acc(sycl::handler& __cgh)
        {
            return __dpl_sycl::__local_accessor<_KeyT>(__buf_size, __cgh);
        }
    };

    template <typename _KeyT>
    class _TempBuf<_KeyT, ::std::false_type /*global memory buffer*/>
    {
        sycl::buffer<_KeyT> __buf;

      public:
        _TempBuf(uint16_t __n) : __buf(__n) {}
        auto
        get_acc(sycl::handler& __cgh)
        {
            return sycl::accessor(__buf, __cgh, sycl::read_write, __dpl_sycl::__no_init{});
        }
    };

    template <typename _ValueT, typename _Wi, typename _Src, typename _Values>
    static void
    __block_load(const _Wi __wi, const _Src& __src, _Values& __values, const uint32_t __n)
    {
        _ONEDPL_PRAGMA_UNROLL
        for (uint16_t __i = 0; __i < __block_size; ++__i)
        {
            const uint16_t __idx = __wi * __block_size + __i;
            if (__idx < __n)
                new (&__values[__i]) _ValueT(::std::move(__src[__idx]));
        }
    }

    static_assert(__wg_size <= 1024);
    static constexpr uint16_t __bin_count = 1 << __radix;
    static constexpr uint16_t __counter_buf_sz = __wg_size * __bin_count + 1; //+1(init value) for exclusive scan result

    template <typename _T, typename _Size>
    auto
    __check_slm_size(sycl::queue __q, _Size __n)
    {
        assert(__n <= 1 << 16); //the kernel is designed for data size <= 64K

        const auto __req_slm_size_counters = __counter_buf_sz * sizeof(uint32_t);

        // Pessimistically only use half of the memory to take into account
        // a SYCL group algorithm might use a portion of SLM
        const ::std::size_t __max_slm_size =
            __q.get_device().template get_info<sycl::info::device::local_mem_size>() / 2;

        const auto __n_uniform = 1 << (::std::uint32_t(log2(__n - 1)) + 1);
        const auto __req_slm_size_val = sizeof(_T) * __n_uniform;

        if (__req_slm_size_val + __req_slm_size_counters <= __max_slm_size)
            return ::std::make_pair(true, true);  //the values and the counters are placed in SLM
        if (__req_slm_size_counters <= __max_slm_size)
            return ::std::make_pair(false, true); //the counters are placed in SLM, the values - in the global memory
        return ::std::make_pair(false, false);    //the values and the counters are placed in the global memory
    }

    template <typename _KernelName>
    struct __one_group_submitter;

    template <typename... _Name>
    struct __one_group_submitter<__internal::__optional_kernel_name<_Name...>>
    {
        template <typename _RangeIn, typename _Proj, typename _SLM_tag_val, typename _SLM_counter>
        auto
        operator()(sycl::queue __q, _RangeIn&& __src, _Proj __proj, _SLM_tag_val, _SLM_counter)
        {
            uint16_t __n = __src.size();
            assert(__n <= __block_size * __wg_size);

            using _ValT = oneapi::dpl::__internal::__value_t<_RangeIn>;
            using _KeyT = oneapi::dpl::__internal::__key_t<_Proj, _RangeIn>;

            _TempBuf<_ValT, _SLM_tag_val> __buf_val(__block_size * __wg_size);
            _TempBuf<uint32_t, _SLM_counter> __buf_count(__counter_buf_sz);

            sycl::nd_range __range{sycl::range{__wg_size}, sycl::range{__wg_size}};
            return __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __src);

                auto __exchange_lacc = __buf_val.get_acc(__cgh);
                auto __counter_lacc = __buf_count.get_acc(__cgh);

                __cgh.parallel_for<_Name...>(
                    __range, ([=](sycl::nd_item<1> __it) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(16)]] {
                        union __storage
                        {
                            _ValT __v[__block_size];
                            __storage() {}
                        } __values;
                        uint16_t __wi = __it.get_local_linear_id();
                        uint16_t __begin_bit = 0;
                        constexpr uint16_t __end_bit = sizeof(_KeyT) * ::std::numeric_limits<unsigned char>::digits;

                        //copy(move) values construction
                        __block_load<_ValT>(__wi, __src, __values.__v, __n);

                        __dpl_sycl::__group_barrier(__it);
                        while (true)
                        {
                            uint16_t __indices[__block_size]; //indices for indirect access in the "re-order" phase
                            {
                                //pointers(by performance reasons) to bucket's counters
                                uint32_t* __counters[__block_size];

                                //1. "counting" phase
                                //counter initialization
                                auto __pcounter = __dpl_sycl::__get_accessor_ptr(__counter_lacc) + __wi;

                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 0; __i < __bin_count; ++__i)
                                    __pcounter[__i * __wg_size] = 0;

                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 0; __i < __block_size; ++__i)
                                {
                                    const uint16_t __idx = __wi * __block_size + __i;
                                    const uint16_t __bin = __idx < __n ? __get_bucket</*mask*/ __bin_count - 1>(
                                        __order_preserving_cast<__is_asc>(__proj(__values.__v[__i])), __begin_bit)
                                        : __bin_count - 1/*default bin for out of range elements (when idx >= n)*/;

                                    //"counting" and local offset calculation
                                    __counters[__i] = &__pcounter[__bin * __wg_size];
                                    __indices[__i] = *__counters[__i];
                                    *__counters[__i] = __indices[__i] + 1;
                                }
                                __dpl_sycl::__group_barrier(__it);

                                //2. scan phase
                                {
                                    //TODO: probably can be further optimized

                                    //scan contiguous numbers
                                    uint16_t __bin_sum[__bin_count];
                                    __bin_sum[0] = __counter_lacc[__wi * __bin_count];

                                    _ONEDPL_PRAGMA_UNROLL
                                    for (uint16_t __i = 1; __i < __bin_count; ++__i)
                                        __bin_sum[__i] = __bin_sum[__i - 1] + __counter_lacc[__wi * __bin_count + __i];

                                    __dpl_sycl::__group_barrier(__it);
                                    //exclusive scan local sum
                                    uint16_t __sum_scan = __dpl_sycl::__exclusive_scan_over_group(
                                        __it.get_group(), __bin_sum[__bin_count - 1], __dpl_sycl::__plus<uint16_t>());
                                    //add to local sum, generate exclusive scan result
                                    _ONEDPL_PRAGMA_UNROLL
                                    for (uint16_t __i = 0; __i < __bin_count; ++__i)
                                        __counter_lacc[__wi * __bin_count + __i + 1] = __sum_scan + __bin_sum[__i];

                                    if (__wi == 0)
                                        __counter_lacc[0] = 0;
                                    __dpl_sycl::__group_barrier(__it);
                                }

                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 0; __i < __block_size; ++__i)
                                {
                                    // a global index is a local offset plus a global base index
                                    __indices[__i] += *__counters[__i];
                                }
                            }

                            __begin_bit += __radix;

                            //3. "re-order" phase
                            __dpl_sycl::__group_barrier(__it);
                            if (__begin_bit >= __end_bit)
                            {
                                // the last iteration - writing out the result
                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 0; __i < __block_size; ++__i)
                                {
                                    const uint16_t __r = __indices[__i];
                                    if (__r < __n)
                                    {
                                        //move the values to source range and destroy the values
                                        __src[__r] = ::std::move(__values.__v[__i]);
                                        __values.__v[__i].~_ValT();
                                    }
                                }

                                //destroy values in exchange buffer
                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 0; __i < __block_size; ++__i)
                                {
                                    const uint16_t __idx = __wi * __block_size + __i;
                                    if (__idx < __n)
                                        __exchange_lacc[__idx].~_ValT();
                                }

                                return;
                            }

                            //3.1 data exchange
                            if (__begin_bit == __radix) //the first sort iteration
                            {
                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 0; __i < __block_size; ++__i)
                                {
                                    const uint16_t __r = __indices[__i];
                                    if (__r < __n)
                                        new (&__exchange_lacc[__r]) _ValT(::std::move(__values.__v[__i]));
                                }
                            }
                            else
                            {
                                _ONEDPL_PRAGMA_UNROLL
                                for (uint16_t __i = 0; __i < __block_size; ++__i)
                                {
                                    const uint16_t __r = __indices[__i];
                                    if (__r < __n)
                                        __exchange_lacc[__r] = ::std::move(__values.__v[__i]);
                                }
                            }

                            __dpl_sycl::__group_barrier(__it);

                            _ONEDPL_PRAGMA_UNROLL
                            for (uint16_t __i = 0; __i < __block_size; ++__i)
                            {
                                const uint16_t __idx = __wi * __block_size + __i;
                                if (__idx < __n)
                                    __values.__v[__i] = ::std::move(__exchange_lacc[__idx]);
                            }

                            __dpl_sycl::__group_barrier(__it);
                        }
                    }));
            });
        }
    };
};

//} // namespace __par_backend_hetero
//} // namespace dpl
//} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H */
