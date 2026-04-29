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

template <bool use_subgroup_ops, typename ValueType>
ValueType
helper_level4(sycl::sub_group sg, ValueType v, ValueType* slm)
{
    std::uint32_t lid = sg.get_local_linear_id();
    if (!use_subgroup_ops)
    {
        std::uint32_t base = sg.get_group_linear_id() * sg.get_max_local_range()[0];
        slm[base + lid] = v;
        sycl::group_barrier(sg);
        ValueType shifted = slm[base + (lid > 0 ? lid - 1 : lid)];
        return shifted;
    }
    else
    {
        return sycl::shift_group_right(sg, v, 1);
    }
}

template <bool use_subgroup_ops, typename ValueType, typename BinaryOp>
ValueType
helper_level3(sycl::sub_group sg, ValueType v, BinaryOp op, ValueType* slm)
{
    ValueType shifted = helper_level4<use_subgroup_ops>(sg, v, slm);
    if (sg.get_local_linear_id() > 0)
        v = op(shifted, v);
    return v;
}

template <bool use_subgroup_ops, typename ValueType, typename BinaryOp>
ValueType
helper_level2(sycl::sub_group sg, ValueType v, BinaryOp op, ValueType& carry, ValueType* slm)
{
    v = helper_level3<use_subgroup_ops>(sg, v, op, slm);
    carry = sycl::group_broadcast(sg, v, sg.get_max_local_range()[0] - 1);
    return v;
}

template <bool use_subgroup_ops, std::uint16_t max_iters, typename ValueType, typename BinaryOp, typename GenInput>
void
helper_level1(sycl::sub_group sg, GenInput gen, BinaryOp op, ValueType& carry, const ValueType* in_ptr,
              ValueType* out_ptr, std::size_t begin_idx, std::size_t n, std::uint32_t iters, ValueType* slm)
{
    std::uint32_t sg_size = sg.get_max_local_range()[0];
    ValueType v = gen(in_ptr, begin_idx);
    v = helper_level2<use_subgroup_ops>(sg, v, op, carry, slm);
    out_ptr[begin_idx] = v;

    for (std::uint32_t j = 1; j < iters; ++j)
    {
        std::size_t id = begin_idx + j * sg_size;
        if (id < n)
        {
            v = gen(in_ptr, id);
            v = op(carry, v);
            v = helper_level2<use_subgroup_ops>(sg, v, op, carry, slm);
            out_ptr[id] = v;
        }
    }
}

template <std::uint16_t max_inputs_per_item, typename ValueType, typename BinaryOp, typename GenInput>
struct deep_template_submitter
{
    sycl::event
    operator()(sycl::queue& q, std::size_t n, const ValueType* in_usm, ValueType* out_usm) const
    {
        constexpr std::uint32_t wg_size = 256;
        constexpr std::uint32_t num_wg = 8;
        std::uint32_t global = wg_size * num_wg;

        return q.submit([&, *this](sycl::handler& cgh) {
            sycl::local_accessor<ValueType, 1> slm(sycl::range<1>(wg_size), cgh);
            sycl::local_accessor<ValueType, 1> sg_partials(sycl::range<1>(wg_size / 32), cgh);

            cgh.parallel_for<class deep_template_kernel>(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg_size)), [=, *this](sycl::nd_item<1> ndi) {
                    auto sg = ndi.get_sub_group();
                    std::uint32_t sg_size = sg.get_max_local_range()[0];
                    std::uint32_t sg_id = sg.get_group_linear_id();
                    std::uint32_t sg_lid = sg.get_local_linear_id();
                    std::uint32_t gid = ndi.get_group(0);

                    std::size_t group_begin = gid * wg_size * iters_per_item;
                    std::size_t begin_idx = group_begin + sg_id * iters_per_item * sg_size + sg_lid;

                    ValueType carry{};
                    ValueType* slm_ptr = &slm[0];

                    if (begin_idx < n)
                    {
                        helper_level1<false, max_inputs_per_item>(sg, gen_input, binary_op, carry, in_usm, out_usm,
                                                                  begin_idx, n, iters_per_item, slm_ptr);
                    }

                    if (sg_lid == 0)
                        sg_partials[sg_id] = carry;

                    sycl::group_barrier(ndi.get_group());

                    if (sg_id == 0 && sg_lid < wg_size / sg_size)
                    {
                        ValueType p = sg_partials[sg_lid];
                        p = helper_level3<false>(sg, p, binary_op, slm_ptr);
                        sg_partials[sg_lid] = p;
                    }
                });
        });
    }

    const std::uint32_t max_num_work_groups;
    const std::uint32_t work_group_size;
    const std::uint32_t max_block_size;
    const std::uint32_t max_num_sub_groups_local;
    const std::size_t n;
    const bool use_slm_for_comm;
    const std::uint32_t iters_per_item;

    const GenInput gen_input;
    const BinaryOp binary_op;
    const ValueType init;
};

struct simple_gen
{
    template <typename T>
    T
    operator()(const T* data, std::size_t id) const
    {
        return data[id];
    }
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
