// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// merge_wg_resident_tag with keys and values of a device copyable, non-arithmetic type

#include "../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include "../support/utils.h"
#include "../support/utils_sycl.h"

namespace kt = oneapi::dpl::experimental::kt;

// A struct with padding; w duplicates the low bits of v to check that whole objects are moved
struct Elem
{
    std::uint32_t v;
    std::uint16_t w;
};
static_assert(sycl::is_device_copyable_v<Elem>);

Elem
make(std::uint32_t x)
{
    return Elem{x, std::uint16_t(x)};
}

std::uint32_t
id(const Elem& e)
{
    return e.w == std::uint16_t(e.v) ? e.v : ~0u;
}

// Orders keys by their value modulo 7 only, so stability is visible in the keys
struct Mod7Less
{
    bool
    operator()(const Elem& a, const Elem& b) const
    {
        return a.v % 7 < b.v % 7;
    }
};

// Sorts segments of keys, with their input indices as values, and checks against std::stable_sort per segment
template <typename KernelParam>
void
test_case(sycl::queue q, std::size_t segment_size, std::size_t segment_count)
{
    const std::size_t n = segment_size * segment_count;
    std::vector<Elem> keys(n), vals(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        keys[i] = make(std::uint32_t((i * 2654435761u) >> 7) % 50);
        vals[i] = make(std::uint32_t(i));
    }

    Elem* d_keys = sycl::malloc_device<Elem>(n, q);
    Elem* d_vals = sycl::malloc_device<Elem>(n, q);
    q.copy(keys.data(), d_keys, n).wait();
    q.copy(vals.data(), d_vals, n).wait();
    kt::gpu::batched_stable_sort_by_key<kt::gpu::merge_wg_resident_tag<Mod7Less>>(q, d_keys, d_keys + n, d_vals,
                                                                                  segment_size, KernelParam{})
        .wait();
    std::vector<Elem> keys_out(n), vals_out(n);
    q.copy(d_keys, keys_out.data(), n).wait();
    q.copy(d_vals, vals_out.data(), n).wait();
    sycl::free(d_keys, q);
    sycl::free(d_vals, q);

    std::vector<std::uint32_t> order(n);
    for (std::size_t i = 0; i < n; ++i)
        order[i] = std::uint32_t(i);
    for (std::size_t begin = 0; begin < n; begin += segment_size)
        std::stable_sort(order.begin() + begin, order.begin() + begin + segment_size,
                         [&](std::uint32_t a, std::uint32_t b) { return Mod7Less{}(keys[a], keys[b]); });

    bool ok = true;
    for (std::size_t i = 0; i < n && ok; ++i)
        ok = id(keys_out[i]) == id(keys[order[i]]) && id(vals_out[i]) == order[i];
    EXPECT_TRUE(
        ok, ("segment size: " + std::to_string(segment_size) + ", count: " + std::to_string(segment_count)).c_str());
}

int
main()
{
    auto q = TestUtils::get_test_queue();
    try
    {
        for (std::size_t segment_size : {7, 317, 1024})
            for (std::size_t segment_count : {3, 64})
                test_case<kt::kernel_param<4, 256>>(q, segment_size, segment_count);
    }
    catch (const std::exception& exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
    return TestUtils::done();
}
