// -*- C++ -*-
//===-- reduce_by_segment_offsets.pass.cpp --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// reduce_by_segment with variable-length segments defined by offsets,
// see rfcs/proposed/segmented_reduce/README.md

#include "support/test_config.h"

#include "oneapi/dpl/execution"
#include "oneapi/dpl/functional"
#include "oneapi/dpl/iterator"
#include "oneapi/dpl/numeric"

#include "support/reduce_serial_impl.h"
#include "support/utils.h"
#include "support/utils_device_copyable.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/sycl_alloc_utils.h"
#    include "support/utils_sycl.h"
#endif

#include <algorithm>   // std::copy_n, std::fill, std::min, std::max
#include <cstddef>     // std::size_t, std::nullptr_t
#include <cstdint>     // std::int16_t, std::int64_t, std::uint32_t, std::uint64_t
#include <functional>  // std::plus, std::multiplies, std::bit_xor, std::logical_and
#include <iterator>    // std::back_inserter, std::distance, std::iterator_traits, std::random_access_iterator_tag
#include <limits>      // std::numeric_limits
#include <numeric>     // std::iota
#include <random>      // std::mt19937, std::uniform_int_distribution
#include <type_traits> // std::enable_if_t, std::is_signed_v
#include <vector>      // std::vector

static_assert(ONEDPL_HAS_SEGMENTED_ALGORITHMS >= 202611L);

using namespace TestUtils;

// The ways to split the input [0, n) into segments
enum class segments
{
    one,           // a single segment over the whole input
    unit,          // every element is a segment
    equal,         // segments of the same length, the last one crosses the end of the input
    mixed,         // segments of random lengths, an empty one, one of one element and a long one
    all_empty,     // only empty segments
    inner,         // the segments cover only the middle of the input
    out_of_bounds, // the segments cross the ends of the input or lie beyond them
};

// Returns the num_segments + 1 non-decreasing offsets of the segments
template <typename Offset>
std::vector<Offset>
make_offsets(segments kind, std::int64_t n, std::mt19937& gen)
{
    std::vector<std::int64_t> bounds{0};
    auto add = [&bounds](std::int64_t length) { bounds.push_back(bounds.back() + length); };
    switch (kind)
    {
    case segments::one:
        add(n);
        break;
    case segments::unit:
        for (std::int64_t i = 0; i < n; ++i)
            add(1);
        break;
    case segments::equal:
        while (bounds.back() < n)
            add(1 + n / 10);
        break;
    case segments::mixed:
    {
        // The long segment has three quarters of the input: every operation reduces a long segment
        std::uniform_int_distribution<std::int64_t> length(0, 2 + n / 32);
        auto add_random = [&](std::int64_t end) {
            while (bounds.back() < end)
                add(std::min(length(gen), end - bounds.back()));
        };
        add(0);
        add(std::min<std::int64_t>(1, n));
        add_random(n / 8);
        add(std::min(n * 3 / 4, n - bounds.back()));
        add_random(n);
        add(0);
        break;
    }
    case segments::all_empty:
        bounds.assign(5, n / 2);
        break;
    case segments::inner:
        bounds.assign(1, n / 4);
        while (bounds.back() < n - n / 4)
            add(std::min(1 + n / 8, n - n / 4 - bounds.back()));
        break;
    case segments::out_of_bounds:
        // A negative first offset, if representable; the last offset is the largest representable one
        bounds = {std::is_signed_v<Offset> ? -3 : n / 4, n / 3, n / 2, n + 2, n + 5};
        break;
    }
    std::vector<Offset> offsets(bounds.begin(), bounds.end());
    if (kind == segments::out_of_bounds)
        offsets.push_back(std::numeric_limits<Offset>::max());
    return offsets;
}

// The algorithm requires random access iterators, as the RFC does: the forward and bidirectional iterators that the
// test framework makes for the host policies are skipped
struct test_reduce_by_segment_offsets
{
    template <typename Policy, typename InputIt, typename... Args>
    std::enable_if_t<!is_base_of_iterator_category_v<std::random_access_iterator_tag, InputIt>>
    operator()(Policy&&, InputIt, Args&&...)
    {
    }

    template <typename Policy, typename InputIt, typename OutputIt, typename OffsetIt, typename SegmentNum,
              typename Init, typename BinaryOp, typename Out>
    std::enable_if_t<is_base_of_iterator_category_v<std::random_access_iterator_tag, InputIt>>
    operator()(Policy&& exec, InputIt first, InputIt last, OutputIt out_first, OutputIt out_last, OffsetIt offsets,
               SegmentNum num_segments, Init init, BinaryOp op, Out unwritten)
    {
        // The reference runs on copies of the input and the offsets
        const std::vector<typename std::iterator_traits<InputIt>::value_type> input(first, last);
        std::vector<typename std::iterator_traits<OffsetIt>::value_type> offsets_copy;
        if (num_segments > 0)
            std::copy_n(offsets, num_segments + 1, std::back_inserter(offsets_copy));
        std::vector<Out> expected(std::distance(out_first, out_last), unwritten);
        reduce_by_segment_offsets_serial(input.begin(), input.end(), offsets_copy.begin(), num_segments, init, op,
                                         expected.begin());

        auto res = oneapi::dpl::reduce_by_segment(std::forward<Policy>(exec), first, last, out_first, num_segments,
                                                  offsets, init, op);
        EXPECT_EQ_N(expected.begin(), out_first, expected.size(), "wrong result from reduce_by_segment with offsets");
        // The end of the results, which a non-positive number of segments leaves at the start
        EXPECT_EQ(num_segments > 0 ? out_first + num_segments : out_first, res,
                  "wrong iterator returned by reduce_by_segment with offsets");

        std::fill(out_first, out_last, unwritten);
    }
};

// Offsets which are not sorted give negatively sized segments: the results are unspecified, but the input
// must not be accessed out of its bounds
struct test_unordered_offsets
{
    template <typename Policy, typename InputIt, typename... Args>
    std::enable_if_t<!is_base_of_iterator_category_v<std::random_access_iterator_tag, InputIt>>
    operator()(Policy&&, InputIt, Args&&...)
    {
    }

    template <typename Policy, typename InputIt, typename OutputIt, typename OffsetIt, typename SegmentNum,
              typename Init, typename BinaryOp>
    std::enable_if_t<is_base_of_iterator_category_v<std::random_access_iterator_tag, InputIt>>
    operator()(Policy&& exec, InputIt first, InputIt last, OutputIt out_first, OutputIt /*out_last*/,
               OffsetIt offsets, SegmentNum num_segments, Init init, BinaryOp op)
    {
        auto res = oneapi::dpl::reduce_by_segment(std::forward<Policy>(exec), first, last, out_first, num_segments,
                                                  offsets, init, op);
        EXPECT_EQ(num_segments > 0 ? out_first + num_segments : out_first, res,
                  "wrong iterator returned by reduce_by_segment with unordered offsets");
    }
};

// The offsets of fixed-length segments computed on the fly, which emulates fixed-length segments as the RFC suggests
struct segment_begin
{
    std::int64_t length;

    std::int64_t
    operator()(std::int64_t segment) const
    {
        return segment * length;
    }
};

// An accumulator which only the operation makes from the input elements: the RFC requires the operation for every
// combination of the accumulator, the initial value and the input, but no conversion of an element
struct sum_acc
{
    std::int64_t value;

    sum_acc() = default;
    sum_acc(std::int64_t v, std::nullptr_t) : value(v) {}
    operator std::int64_t() const { return value; }
};

struct sum_acc_plus
{
    sum_acc
    operator()(sum_acc a, int b) const
    {
        return sum_acc(a.value + b, nullptr);
    }
    sum_acc
    operator()(int a, sum_acc b) const
    {
        return sum_acc(a + b.value, nullptr);
    }
    sum_acc
    operator()(int a, int b) const
    {
        return sum_acc(std::int64_t(a) + b, nullptr);
    }
    sum_acc
    operator()(sum_acc a, sum_acc b) const
    {
        return sum_acc(a.value + b.value, nullptr);
    }
};

// An accumulator which can be made from an input element, but with another meaning: bit_set(b) holds the bit pattern
// b, while the operation adds an element i as the bit i. As in std::reduce, a partial result must not start from an
// element converted to the accumulator type
struct bit_set
{
    std::uint64_t bits;

    bit_set() = default;
    bit_set(std::uint64_t b) : bits(b) {}
    operator std::uint64_t() const { return bits; }
};

struct bit_set_union
{
    static std::uint64_t
    bit(int i)
    {
        return std::uint64_t{1} << i;
    }
    bit_set
    operator()(bit_set a, int b) const
    {
        return bit_set(a.bits | bit(b));
    }
    bit_set
    operator()(int a, bit_set b) const
    {
        return bit_set(bit(a) | b.bits);
    }
    bit_set
    operator()(int a, int b) const
    {
        return bit_set(bit(a) | bit(b));
    }
    bit_set
    operator()(bit_set a, bit_set b) const
    {
        return bit_set(a.bits | b.bits);
    }
};

// An integer accumulator made from a floating-point initial value: an empty segment gets the initial value converted
// to the accumulator, so without its fraction
struct int_sum
{
    std::int64_t value;

    int_sum() = default;
    explicit int_sum(std::int64_t v) : value(v) {}
    int_sum(float f) : value(std::int64_t(f)) {}
    operator float() const { return float(value); }
};

struct int_sum_plus
{
    static std::int64_t
    get(int_sum a)
    {
        return a.value;
    }
    static std::int64_t
    get(float a)
    {
        return std::int64_t(a);
    }
    static std::int64_t
    get(int a)
    {
        return a;
    }
    template <typename A, typename B>
    int_sum
    operator()(A a, B b) const
    {
        return int_sum(get(a) + get(b));
    }
};

template <std::size_t CallNumber, typename T, typename Out, typename Offset, typename SegmentNum, typename Init,
          typename BinaryOp, typename Generator>
void
test_segments(Init init, BinaryOp op, Generator generator, Out unwritten)
{
    std::mt19937 gen(42);
    for (std::size_t n = 0; n <= max_n; n = n <= 16 ? n + 1 : std::size_t(3.1415 * n))
    {
        Sequence<T> in(n, generator);
        const std::vector<Offset> offsets = make_offsets<Offset>(segments::mixed, n, gen);
        const SegmentNum num_segments = SegmentNum(offsets.size() - 1);
        Sequence<Out> out(num_segments, [unwritten](std::size_t) { return unwritten; });
        invoke_on_all_policies<CallNumber * 2>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                                 out.end(), offsets.begin(), num_segments, init, op, unwritten);

        // A non-positive number of segments: the output is not written
        invoke_on_all_policies<CallNumber * 2>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                                 out.end(), offsets.begin(), SegmentNum(0), init, op, unwritten);
        if constexpr (std::is_signed_v<SegmentNum>)
        {
            invoke_on_all_policies<CallNumber * 2>()(test_reduce_by_segment_offsets(), in.begin(), in.end(),
                                                     out.begin(), out.end(), offsets.begin(), SegmentNum(-1), init,
                                                     op, unwritten);
        }
    }

    // Read-only input
    const std::size_t n = 1000;
    Sequence<T> in(n, generator);
    const std::vector<Offset> offsets = make_offsets<Offset>(segments::mixed, n, gen);
    const SegmentNum num_segments = SegmentNum(offsets.size() - 1);
    Sequence<Out> out(num_segments, [unwritten](std::size_t) { return unwritten; });
    invoke_on_all_policies<CallNumber * 2 + 1>()(test_reduce_by_segment_offsets(), in.cbegin(), in.cend(),
                                                 out.begin(), out.end(), offsets.begin(), num_segments, init, op,
                                                 unwritten);
}

// The segment distributions at the edges, with one operation and one type
template <std::size_t CallNumber>
void
test_edge_segment_distributions()
{
    std::mt19937 gen(42);
    for (std::size_t n = 0; n <= max_n; n = n <= 16 ? n + 1 : std::size_t(3.1415 * n))
    {
        Sequence<int> in(n, [](std::size_t i) { return int(i % 101) - 50; });
        for (segments kind : {segments::one, segments::unit, segments::equal, segments::all_empty})
        {
            const std::vector<int> offsets = make_offsets<int>(kind, n, gen);
            const int num_segments = int(offsets.size() - 1);
            Sequence<int> out(std::max(num_segments, 1), [](std::size_t) { return -1; });
            invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                                 out.end(), offsets.begin(), num_segments, 5, std::plus<int>{}, -1);
        }
    }
}

// Fixed-length segments, by offsets computed on the fly, which users can write themselves; the last two segments
// cross or lie beyond the end of the input
template <std::size_t CallNumber>
void
test_fixed_length_segments()
{
    for (std::size_t n = 0; n <= max_n; n = n <= 16 ? n + 1 : std::size_t(3.1415 * n))
    {
        Sequence<int> in(n, [](std::size_t i) { return int(i % 101) - 50; });
        for (std::int64_t length : {1, 7, 1000})
        {
            auto offsets = oneapi::dpl::make_transform_iterator(oneapi::dpl::counting_iterator<std::int64_t>(0),
                                                                segment_begin{length});
            const std::int64_t num_segments = std::int64_t(n / length + 2);
            Sequence<int> out(num_segments, [](std::size_t) { return -1; });
            invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                                 out.end(), offsets, num_segments, 5, std::plus<int>{}, -1);
        }
    }
}

template <std::size_t CallNumber>
void
test_unordered()
{
    const std::size_t n = 100;
    Sequence<int> in(n, [](std::size_t i) { return int(i % 7); });
    const std::vector<int> offsets{50, 0, 200, -5, 30, 110, 1, 1, 99};
    Sequence<int> out(offsets.size() - 1);
    invoke_on_all_policies<CallNumber>()(test_unordered_offsets(), in.begin(), in.end(), out.begin(), out.end(),
                                         offsets.begin(), int(offsets.size() - 1), 0, std::plus<int>{});
}

// The segment length boundaries are based on the thresholds when the implementation selects different paths
template <std::size_t CallNumber>
void
test_boundaries()
{
    const std::int64_t lengths[] = {511, 512, 513, 4096, 4097, 16384, 16385, 32767, 32768, 32769, 49152, 49153};
    for (std::int64_t short_ones : {0, 42})
    {
        std::vector<std::int64_t> offsets{0};
        for (std::int64_t length : lengths)
        {
            offsets.push_back(offsets.back() + length);
            for (std::int64_t i = 0; i < short_ones; ++i)
                offsets.push_back(offsets.back() + 1);
        }
        const std::int64_t num_segments = std::int64_t(offsets.size() - 1);
        Sequence<int> in(offsets.back(), [](std::size_t i) { return int(i % 101) - 50; });
        Sequence<std::int64_t> out(num_segments, [](std::size_t) { return -1; });
        invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                             out.end(), offsets.begin(), num_segments, std::int64_t{5},
                                             std::plus<std::int64_t>{}, std::int64_t{-1});
    }
}

// The input is the middle of a larger sequence, whose other elements are poison: a read out of the bounds of the input
// changes a result
constexpr std::size_t guard_length = 1000;
constexpr int poison = 1000000;

template <std::size_t CallNumber>
void
test_input_bounds()
{
    std::mt19937 gen(42);
    for (std::size_t n : {std::size_t(1), std::size_t(1000), std::size_t(50000)})
    {
        Sequence<int> guarded(n + 2 * guard_length, [n](std::size_t i) {
            return i < guard_length || i >= guard_length + n ? poison : int(i % 101) - 50;
        });
        for (segments kind : {segments::mixed, segments::inner, segments::out_of_bounds})
        {
            const std::vector<std::int64_t> offsets = make_offsets<std::int64_t>(kind, n, gen);
            const std::int64_t num_segments = std::int64_t(offsets.size() - 1);
            Sequence<std::int64_t> out(std::max<std::size_t>(num_segments, 1), [](std::size_t) { return -1; });
            invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), guarded.begin() + guard_length,
                                                 guarded.begin() + guard_length + n, out.begin(), out.end(),
                                                 offsets.begin(), num_segments, std::int64_t{5},
                                                 std::plus<std::int64_t>{}, std::int64_t{-1});
        }
    }
}

#if TEST_DPCPP_BACKEND_PRESENT
// USM pointers and sycl::buffer iterators: the tests with host iterators above do not reach them
template <sycl::usm::alloc alloc_type, typename KernelName>
void
test_usm()
{
    using NewKernelName = TestUtils::unique_kernel_name<KernelName, TestUtils::uniq_kernel_index<alloc_type>()>;

    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();
    const std::size_t n = 20000; // some segments are long, beyond the chunk length of the device implementation
    std::mt19937 gen(42);
    // The input is guarded by poison on both sides, as in test_input_bounds
    std::vector<int> in(n + 2 * guard_length, poison);
    std::iota(in.begin() + guard_length, in.begin() + guard_length + n, -500);
    for (segments kind : {segments::mixed, segments::all_empty})
    {
        std::vector<int> offsets = make_offsets<int>(kind, n, gen);
        const int num_segments = int(offsets.size() - 1);
        std::vector<int> expected(num_segments), result(num_segments);
        reduce_by_segment_offsets_serial(in.begin() + guard_length, in.begin() + guard_length + n, offsets.begin(),
                                         num_segments, 5, std::plus<int>{}, expected.begin());

        TestUtils::usm_data_transfer<alloc_type, int> in_usm(q, in.begin(), in.end());
        TestUtils::usm_data_transfer<alloc_type, int> offsets_usm(q, offsets.begin(), offsets.end());
        TestUtils::usm_data_transfer<alloc_type, int> out_usm(q, num_segments);
        auto res = oneapi::dpl::reduce_by_segment(CLONE_TEST_POLICY_NAME(policy, NewKernelName),
                                                  in_usm.get_data() + guard_length,
                                                  in_usm.get_data() + guard_length + n, out_usm.get_data(),
                                                  num_segments, offsets_usm.get_data(), 5, std::plus<int>{});
        EXPECT_EQ(out_usm.get_data() + num_segments, res, "wrong iterator returned by reduce_by_segment with USM");
        out_usm.retrieve_data(result.begin());
        EXPECT_EQ_N(expected.begin(), result.begin(), num_segments, "wrong result from reduce_by_segment with USM");
    }
}

template <typename KernelName>
void
test_buffer()
{
    auto policy = TestUtils::get_dpcpp_test_policy();
    const std::size_t n = 20000; // some segments are long, beyond the chunk length of the device implementation
    std::mt19937 gen(42);
    // The input is guarded by poison on both sides, as in test_input_bounds
    std::vector<int> in(n + 2 * guard_length, poison);
    std::iota(in.begin() + guard_length, in.begin() + guard_length + n, -500);
    for (segments kind : {segments::mixed, segments::all_empty})
    {
        std::vector<int> offsets = make_offsets<int>(kind, n, gen);
        const int num_segments = int(offsets.size() - 1);
        std::vector<int> expected(num_segments), result(num_segments, -1);
        reduce_by_segment_offsets_serial(in.begin() + guard_length, in.begin() + guard_length + n, offsets.begin(),
                                         num_segments, 5, std::plus<int>{}, expected.begin());
        {
            sycl::buffer<int> in_buf(in.data(), in.size());
            sycl::buffer<int> offsets_buf(offsets.data(), offsets.size());
            sycl::buffer<int> out_buf(result.data(), result.size());
            auto res = oneapi::dpl::reduce_by_segment(CLONE_TEST_POLICY_NAME(policy, KernelName),
                                                      oneapi::dpl::begin(in_buf) + guard_length,
                                                      oneapi::dpl::begin(in_buf) + guard_length + n,
                                                      oneapi::dpl::begin(out_buf), num_segments,
                                                      oneapi::dpl::begin(offsets_buf), 5, std::plus<int>{});
            EXPECT_EQ(oneapi::dpl::begin(out_buf) + num_segments, res,
                      "wrong iterator returned by reduce_by_segment with sycl::buffer");
        }
        EXPECT_EQ_N(expected.begin(), result.begin(), num_segments,
                    "wrong result from reduce_by_segment with sycl::buffer");
    }
}

class KernelNameUSM;
class KernelNameBuffer;
class KernelNameCounting;
class KernelNameNarrowCounting;
#endif // TEST_DPCPP_BACKEND_PRESENT

// Counting iterators help cover input with specific difference types.
void
test_counting_input()
{
    std::vector<int> offsets{0};
    for (int length : {100000, 70000})
    {
        offsets.push_back(offsets.back() + length);
        for (int i = 0; i < 500; ++i)
            offsets.push_back(offsets.back() + 1 + i % 3);
    }
    const int num_segments = int(offsets.size() - 1);
    const oneapi::dpl::counting_iterator<int> first(0);
    std::vector<std::int64_t> out(num_segments, -1);
    test_reduce_by_segment_offsets()(oneapi::dpl::execution::par_unseq, first, first + offsets.back(), out.begin(),
                                     out.end(), offsets.begin(), num_segments, std::int64_t{5},
                                     std::plus<std::int64_t>{}, std::int64_t{-1});
#if TEST_DPCPP_BACKEND_PRESENT
    std::fill(out.begin(), out.end(), -1);
    auto policy = TestUtils::get_dpcpp_test_policy();
    test_reduce_by_segment_offsets()(CLONE_TEST_POLICY_NAME(policy, KernelNameCounting),
                                     first, first + offsets.back(), out.begin(), out.end(), offsets.begin(),
                                     num_segments, std::int64_t{5}, std::plus<std::int64_t>{}, std::int64_t{-1});
#endif
}

// More segments than a 16-bit difference type can count
void
test_narrow_counting_input()
{
    const int n = 1000, num_segments = 40000;
    std::vector<int> offsets(num_segments + 1, n);
    offsets[0] = 0;
    const oneapi::dpl::counting_iterator<std::int16_t> first(0);
    std::vector<std::int64_t> out(num_segments, -1);
    test_reduce_by_segment_offsets()(oneapi::dpl::execution::par_unseq, first, first + n, out.begin(), out.end(),
                                     offsets.begin(), num_segments, std::int64_t{5}, std::plus<std::int64_t>{},
                                     std::int64_t{-1});
#if TEST_DPCPP_BACKEND_PRESENT
    std::fill(out.begin(), out.end(), -1);
    auto policy = TestUtils::get_dpcpp_test_policy();
    test_reduce_by_segment_offsets()(CLONE_TEST_POLICY_NAME(policy, KernelNameNarrowCounting), first, first + n,
                                     out.begin(), out.end(), offsets.begin(), num_segments, std::int64_t{5},
                                     std::plus<std::int64_t>{}, std::int64_t{-1});
#endif
}

int
main()
{
    // The operations: the ones of the RFC (sum, min, max, product, logical), the accumulator types of its
    // examples, and the ones of the other reduction tests (a user-defined type, an accumulator which is not
    // default-constructible, bitwise operations)

    // Sum; the initial value is not the identity
    test_segments<0, int, int, int, int>(5, std::plus<int>{}, [](std::size_t i) { return int(i % 101) - 50; }, -1);

    // Sum of 32-bit values with the 64-bit accumulator deduced from the initial value. A partial result may start from
    // the operation on two elements, which adds them in 32 bits as in std::reduce, so two elements fit in 32 bits
    test_segments<1, std::uint32_t, std::uint64_t, std::int64_t, std::size_t>(
        std::uint64_t{1}, std::plus<>{}, [](std::size_t i) { return std::uint32_t(2'000'000'000u + i % 1000); },
        std::uint64_t{0});

    // Sum of a user-defined type
    test_segments<2, MatrixPoint<float>, MatrixPoint<float>, std::uint32_t, std::int64_t>(
        MatrixPoint<float>(1, 2), std::plus<>{},
        [](std::size_t i) { return MatrixPoint<float>(float(i % 7), float(i % 3)); }, MatrixPoint<float>(-1, -1));

    // Sum with an accumulator which is not default-constructible
    test_segments<3, int, int, int, std::int64_t>(
        NoDefaultCtorWrapper<int>{5}, std::plus<NoDefaultCtorWrapper<int>>{},
        [](std::size_t i) { return int(i % 101) - 50; }, -1);

    // Minimum; the initial value is the identity for the input (not the infinity, which EXPECT_EQ_N cannot compare)
    test_segments<4, float, float, std::uint64_t, int>(
        std::numeric_limits<float>::max(), oneapi::dpl::minimum<float>{},
        [](std::size_t i) { return float((i * 7919) % 1000) - 500.f; }, -1.f);

    // Maximum with a lambda, as in the example of the RFC
    test_segments<5, int, int, std::int64_t, std::uint32_t>(
        std::numeric_limits<int>::lowest(), [](int a, int b) { return std::max(a, b); },
        [](std::size_t i) { return int((i * 7919) % 1000) - 500; }, 0);

    // Product, wrapping around in 32 bits
    test_segments<6, std::uint32_t, std::uint32_t, int, std::int64_t>(
        std::uint32_t{1}, std::multiplies<>{}, [](std::size_t i) { return std::uint32_t(i % 13 + 1); },
        std::uint32_t{0});

    // Logical and with the bool accumulator
    test_segments<7, int, int, int, int>(
        true, std::logical_and<>{}, [](std::size_t i) { return int(i % 97 != 0); }, -1);

    // Bitwise exclusive or
    test_segments<8, std::uint64_t, std::uint64_t, std::uint32_t, std::size_t>(
        std::uint64_t{0}, std::bit_xor<std::uint64_t>{},
        [](std::size_t i) { return std::uint64_t(i) * 0x9E3779B97F4A7C15ull; }, std::uint64_t{1});

    // Sum into an accumulator which cannot be made from an input element
    test_segments<9, int, std::int64_t, std::int64_t, int>(
        sum_acc(5, nullptr), sum_acc_plus{}, [](std::size_t i) { return int(i % 101) - 50; }, std::int64_t{-1});

    // Set union into an accumulator which can be made from an input element, but with another meaning
    test_segments<10, int, std::uint64_t, std::uint32_t, std::int64_t>(
        bit_set(1), bit_set_union{}, [](std::size_t i) { return int(32 + (i * 7) % 32); }, std::uint64_t{0});

    // An accumulator which cannot keep the fraction of the initial value: an empty segment gets the initial value
    // converted to the accumulator type (2), not the initial value itself (2.5)
    test_segments<12, int, float, std::int64_t, int>(
        2.5f, int_sum_plus{}, [](std::size_t i) { return int(i % 101) - 50; }, -1.f);

    test_unordered<39>();
    test_boundaries<42>();
    test_input_bounds<43>();
    test_edge_segment_distributions<44>();
    test_fixed_length_segments<45>();
    test_counting_input();
    test_narrow_counting_input();

#if TEST_DPCPP_BACKEND_PRESENT
    test_usm<sycl::usm::alloc::shared, KernelNameUSM>();
    test_usm<sycl::usm::alloc::device, KernelNameUSM>();
    test_buffer<KernelNameBuffer>();
#endif

    return done();
}
