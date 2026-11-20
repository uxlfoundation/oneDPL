// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include <cstddef> // std::size_t
#include <cstdint> // std::uint8_t
#include <vector>

namespace hetero = oneapi::dpl::__par_backend_hetero;

namespace Test
{

template <typename _T>
struct range_of
{
    using value_type = _T;
};

// Adds 1 to the _Params::__vector_size adjacent elements starting at its index, as the vectorized bricks do
struct counting_body
{
    std::size_t n = 0;

    template <typename _IsFull, typename _Params, typename _Range>
    void
    operator()(_IsFull, const std::size_t idx, _Params, _Range&& rng) const
    {
        for (std::size_t k = 0; k < _Params::__vector_size; ++k)
        {
            if (idx + k < n)
                rng[idx + k] = static_cast<std::uint8_t>(rng[idx + k] + 1);
        }
    }
};

// A brick without a limit of its vector size
struct unlimited_brick : counting_body
{
};

// A brick which limits its vector size
template <std::uint8_t Limit>
struct limited_brick : counting_body
{
    static constexpr std::uint8_t __max_vector_size = Limit;
};

// The parameters given to a brick
static_assert(hetero::__pfor_params<unlimited_brick, range_of<std::uint8_t>>::__vector_size == 4);
static_assert(hetero::__pfor_params<unlimited_brick, range_of<std::uint8_t>>::__can_vectorize);
static_assert(hetero::__pfor_params<unlimited_brick, range_of<std::uint8_t>>::__iters_per_item == 4);
static_assert(hetero::__pfor_params<unlimited_brick, range_of<std::uint16_t>>::__vector_size == 2);
static_assert(hetero::__pfor_params<unlimited_brick, range_of<float>>::__vector_size == 1);
static_assert(!hetero::__pfor_params<unlimited_brick, range_of<float>>::__can_vectorize);

static_assert(hetero::__pfor_params<limited_brick<1>, range_of<std::uint8_t>>::__vector_size == 1);
static_assert(!hetero::__pfor_params<limited_brick<1>, range_of<std::uint8_t>>::__can_vectorize);
static_assert(hetero::__pfor_params<limited_brick<1>, range_of<std::uint8_t>>::__iters_per_item == 16);
static_assert(hetero::__pfor_params<limited_brick<2>, range_of<std::uint8_t>>::__vector_size == 2);
static_assert(hetero::__pfor_params<limited_brick<2>, range_of<std::uint8_t>>::__can_vectorize);
static_assert(hetero::__pfor_params<limited_brick<2>, range_of<std::uint8_t>>::__iters_per_item == 8);
static_assert(hetero::__pfor_params<limited_brick<8>, range_of<std::uint8_t>>::__vector_size == 4);
static_assert(hetero::__iterations_per_item_v<limited_brick<2>, range_of<std::uint8_t>> == 8);

// __dual_brick takes the smaller limit of its bricks
static_assert(hetero::__pfor_params<hetero::__dual_brick<unlimited_brick, unlimited_brick>, range_of<std::uint8_t>>::__vector_size == 4);
static_assert(hetero::__pfor_params<hetero::__dual_brick<limited_brick<1>, unlimited_brick>, range_of<std::uint8_t>>::__vector_size == 1);
static_assert(hetero::__pfor_params<hetero::__dual_brick<limited_brick<8>, limited_brick<2>>, range_of<std::uint8_t>>::__vector_size == 2);

// Every element is processed exactly once by __parallel_for with the brick, small and large kernels alike
template <typename _KernelName, typename _Brick>
void
test_each_element_once(sycl::queue& q, const std::size_t n)
{
    std::vector<std::uint8_t> data(n, 0);
    {
        sycl::buffer<std::uint8_t> buf(data.data(), n);

        auto keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
        auto rng = keep(oneapi::dpl::begin(buf), oneapi::dpl::end(buf));
        auto policy = oneapi::dpl::execution::make_device_policy<_KernelName>(q);

        _Brick brick;
        brick.n = n;
        hetero::__parallel_for(oneapi::dpl::__internal::__device_backend_tag{}, policy, brick, n, rng.all_view()).wait();
    }

    // Compared as int, so that the error messages show numbers rather than characters
    const std::vector<int> expected(n, 1);
    const std::vector<int> actual(data.begin(), data.end());
    EXPECT_EQ_N(expected.begin(), actual.begin(), n, "elements not processed exactly once by __parallel_for");
}

template <typename _KernelName, typename _Brick>
void
test_brick(sycl::queue& q)
{
    // Not multiples of the vector size; the largest one selects the large kernel on every device
    for (std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{1003}, (std::size_t{1} << 24) + 3})
        test_each_element_once<_KernelName, _Brick>(q, n);
}

} // namespace Test

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q = TestUtils::get_test_queue();

    Test::test_brick<class unlimited_kernel, Test::unlimited_brick>(q);
    Test::test_brick<class limited_1_kernel, Test::limited_brick<1>>(q);
    Test::test_brick<class limited_2_kernel, Test::limited_brick<2>>(q);
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
