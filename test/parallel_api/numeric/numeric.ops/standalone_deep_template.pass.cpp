// -*- C++ -*-
//===-- standalone_deep_template.pass.cpp ---------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Standalone SYCL test: Mimics the deep template nesting and *this capture
// pattern of the RTS kernels. A templated submitter struct with multiple
// members is captured via [=, *this] in a parallel_for, and the kernel body
// calls through 4+ levels of templated helper functions.
// Tests whether this pattern itself triggers stack issues under /RTC1.

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>

template <bool __use_subgroup_ops, typename _ValueType>
_ValueType
__helper_level4(sycl::sub_group __sg, _ValueType __v, _ValueType* __slm)
{
    std::uint32_t __lid = __sg.get_local_linear_id();
    if (!__use_subgroup_ops)
    {
        std::uint32_t __base = __sg.get_group_linear_id() * __sg.get_max_local_range()[0];
        __slm[__base + __lid] = __v;
        sycl::group_barrier(__sg);
        _ValueType __shifted = __slm[__base + (__lid > 0 ? __lid - 1 : __lid)];
        return __shifted;
    }
    else
    {
        return sycl::shift_group_right(__sg, __v, 1);
    }
}

template <bool __use_subgroup_ops, typename _ValueType, typename _BinaryOp>
_ValueType
__helper_level3(sycl::sub_group __sg, _ValueType __v, _BinaryOp __op, _ValueType* __slm)
{
    _ValueType __shifted = __helper_level4<__use_subgroup_ops>(__sg, __v, __slm);
    if (__sg.get_local_linear_id() > 0)
        __v = __op(__shifted, __v);
    return __v;
}

template <bool __use_subgroup_ops, typename _ValueType, typename _BinaryOp>
_ValueType
__helper_level2(sycl::sub_group __sg, _ValueType __v, _BinaryOp __op, _ValueType& __carry, _ValueType* __slm)
{
    __v = __helper_level3<__use_subgroup_ops>(__sg, __v, __op, __slm);
    __carry = sycl::group_broadcast(__sg, __v, __sg.get_max_local_range()[0] - 1);
    return __v;
}

template <bool __use_subgroup_ops, std::uint16_t __max_iters, typename _ValueType, typename _BinaryOp,
          typename _GenInput>
void
__helper_level1(sycl::sub_group __sg, _GenInput __gen, _BinaryOp __op, _ValueType& __carry,
                const _ValueType* __in, _ValueType* __out, std::size_t __start, std::size_t __n,
                std::uint32_t __iters, _ValueType* __slm)
{
    std::uint32_t __sg_size = __sg.get_max_local_range()[0];
    _ValueType __v = __gen(__in, __start);
    __v = __helper_level2<__use_subgroup_ops>(__sg, __v, __op, __carry, __slm);
    __out[__start] = __v;

    for (std::uint32_t __j = 1; __j < __iters; ++__j)
    {
        std::size_t __id = __start + __j * __sg_size;
        if (__id < __n)
        {
            __v = __gen(__in, __id);
            __v = __op(__carry, __v);
            __v = __helper_level2<__use_subgroup_ops>(__sg, __v, __op, __carry, __slm);
            __out[__id] = __v;
        }
    }
}

template <std::uint16_t __max_inputs_per_item, typename _ValueType, typename _BinaryOp, typename _GenInput>
struct deep_template_submitter
{
    sycl::event
    operator()(sycl::queue& __q, std::size_t __n, const _ValueType* __in_usm, _ValueType* __out_usm) const
    {
        constexpr std::uint32_t __wg_size = 256;
        constexpr std::uint32_t __num_wg = 8;
        std::uint32_t __global = __wg_size * __num_wg;

        return __q.submit([&, *this](sycl::handler& __cgh) {
            sycl::local_accessor<_ValueType, 1> __slm(sycl::range<1>(__wg_size), __cgh);
            sycl::local_accessor<_ValueType, 1> __sg_partials(sycl::range<1>(__wg_size / 32), __cgh);

            __cgh.parallel_for<class deep_template_kernel>(
                sycl::nd_range<1>(sycl::range<1>(__global), sycl::range<1>(__wg_size)),
                [=, *this](sycl::nd_item<1> __ndi) {
                    auto __sg = __ndi.get_sub_group();
                    std::uint32_t __sg_size = __sg.get_max_local_range()[0];
                    std::uint32_t __sg_id = __sg.get_group_linear_id();
                    std::uint32_t __sg_lid = __sg.get_local_linear_id();
                    std::uint32_t __gid = __ndi.get_group(0);

                    std::size_t __group_start = __gid * __wg_size * __iters_per_item;
                    std::size_t __start = __group_start + __sg_id * __iters_per_item * __sg_size + __sg_lid;

                    _ValueType __carry{};
                    _ValueType* __slm_ptr = &__slm[0];

                    if (__start < __n)
                    {
                        __helper_level1<false, __max_inputs_per_item>(
                            __sg, __gen_input, __binary_op, __carry, __in_usm, __out_usm,
                            __start, __n, __iters_per_item, __slm_ptr);
                    }

                    // Write sub-group partial
                    if (__sg_lid == 0)
                        __sg_partials[__sg_id] = __carry;

                    sycl::group_barrier(__ndi.get_group());

                    // Sub-group 0 scans the partials
                    if (__sg_id == 0 && __sg_lid < __wg_size / __sg_size)
                    {
                        _ValueType __p = __sg_partials[__sg_lid];
                        __p = __helper_level3<false>(__sg, __p, __binary_op, __slm_ptr);
                        __sg_partials[__sg_lid] = __p;
                    }
                });
        });
    }

    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::uint32_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::size_t __n;
    const bool __use_slm_for_comm;
    const std::uint32_t __iters_per_item;

    const _GenInput __gen_input;
    const _BinaryOp __binary_op;
    const _ValueType __init;
};

struct simple_gen
{
    template <typename T>
    T operator()(const T* __in, std::size_t __id) const { return __in[__id]; }
};

int
run_deep_template()
{
    sycl::queue q;
    constexpr std::size_t n = 20000;
    std::vector<std::int32_t> input(n), output(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        input[i] = static_cast<std::int32_t>(i % 100);

    auto* in_usm = sycl::malloc_device<std::int32_t>(n, q);
    auto* out_usm = sycl::malloc_device<std::int32_t>(n, q);
    q.memcpy(in_usm, input.data(), n * sizeof(std::int32_t)).wait();

    deep_template_submitter<128, std::int32_t, std::plus<std::int32_t>, simple_gen> submitter{
        8, 256, 256 * 128 * 8, 8, n, true, 2, simple_gen{}, std::plus<std::int32_t>{}, 0};

    auto ev = submitter(q, n, in_usm, out_usm);
    ev.wait();

    q.memcpy(output.data(), out_usm, n * sizeof(std::int32_t)).wait();
    sycl::free(in_usm, q);
    sycl::free(out_usm, q);

    bool ok = false;
    for (std::size_t i = 0; i < n; ++i)
        if (output[i] != 0) { ok = true; break; }

    if (!ok)
    {
        std::printf("FAIL: standalone_deep_template output is all zeros\n");
        return 1;
    }
    return 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_deep_template();
    if (result != 0)
        return result;
#endif
    return TestUtils::done();
}
