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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(ONEDPL_HAS_SEGMENTED_ALGORITHMS >= 202611L);

using namespace TestUtils;

// The ways to split the input [0, n) into segments
enum class segments
{
    one,           // a single segment over the whole input
    unit,          // every element is a segment
    equal,         // segments of the same length, the last one crosses the end of the input
    growing,       // segments of the lengths 1, 2, 3, ..., the last one is cut at the end of the input
    random,        // segments of random lengths, including empty ones
    all_empty,     // only empty segments
    skewed,        // one long segment between short ones
    inner,         // the segments cover only the middle of the input
    out_of_bounds, // the segments cross the ends of the input or lie beyond them
};

constexpr segments all_segments[] = {segments::one,       segments::unit,   segments::equal,
                                     segments::growing,   segments::random, segments::all_empty,
                                     segments::skewed,    segments::inner,  segments::out_of_bounds};

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
    case segments::growing:
        for (std::int64_t length = 1; bounds.back() < n; ++length)
            add(std::min(length, n - bounds.back()));
        break;
    case segments::random:
    {
        std::uniform_int_distribution<std::int64_t> length(0, 2 + n / 32);
        while (bounds.back() < n)
            add(std::min(length(gen), n - bounds.back()));
        add(0);
        break;
    }
    case segments::all_empty:
        bounds.assign(5, n / 2);
        break;
    case segments::skewed:
    {
        const std::int64_t short_ones = std::min<std::int64_t>(3, n / 2);
        for (std::int64_t i = 0; i < short_ones; ++i)
            add(1);
        add(n - 2 * short_ones);
        for (std::int64_t i = 0; i < short_ones; ++i)
            add(1);
        break;
    }
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
               SegmentNum num_segments, Init init, BinaryOp op, Out trash)
    {
        // The reference runs on copies of the input and the offsets
        const std::vector<typename std::iterator_traits<InputIt>::value_type> input(first, last);
        std::vector<typename std::iterator_traits<OffsetIt>::value_type> offsets_copy;
        if (num_segments > 0)
            std::copy_n(offsets, num_segments + 1, std::back_inserter(offsets_copy));
        std::vector<Out> expected(std::distance(out_first, out_last), trash);
        reduce_by_segment_offsets_serial(input.begin(), input.end(), offsets_copy.begin(), num_segments, init, op,
                                         expected.begin());

        oneapi::dpl::reduce_by_segment(std::forward<Policy>(exec), first, last, out_first, num_segments, offsets,
                                       init, op);
        EXPECT_EQ_N(expected.begin(), out_first, expected.size(), "wrong result from reduce_by_segment with offsets");

        std::fill(out_first, out_last, trash);
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
        oneapi::dpl::reduce_by_segment(std::forward<Policy>(exec), first, last, out_first, num_segments, offsets,
                                       init, op);
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

// The same set union into an accumulator of an arithmetic type, the bit pattern itself, which the reduction of the
// chunks by sub-groups on GPUs also takes
struct bit_mask_union
{
    static std::uint64_t
    bit(int i)
    {
        return std::uint64_t{1} << i;
    }
    std::uint64_t
    operator()(std::uint64_t a, int b) const
    {
        return a | bit(b);
    }
    std::uint64_t
    operator()(int a, std::uint64_t b) const
    {
        return bit(a) | b;
    }
    std::uint64_t
    operator()(int a, int b) const
    {
        return bit(a) | bit(b);
    }
    std::uint64_t
    operator()(std::uint64_t a, std::uint64_t b) const
    {
        return a | b;
    }
};

// A sum whose call operator is not const
struct non_const_plus
{
    std::int64_t
    operator()(std::int64_t a, std::int64_t b)
    {
        return a + b;
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
test_segments(Init init, BinaryOp op, Generator generator, Out trash)
{
    std::mt19937 gen(42);
    for (std::size_t n = 0; n <= max_n; n = n <= 16 ? n + 1 : std::size_t(3.1415 * n))
    {
        Sequence<T> in(n, generator);
        for (segments kind : all_segments)
        {
            const std::vector<Offset> offsets = make_offsets<Offset>(kind, n, gen);
            const SegmentNum num_segments = SegmentNum(offsets.size() - 1);
            Sequence<Out> out(std::max<std::size_t>(offsets.size() - 1, 1), [trash](std::size_t) { return trash; });
            invoke_on_all_policies<CallNumber * 3>()(test_reduce_by_segment_offsets(), in.begin(), in.end(),
                                                     out.begin(), out.end(), offsets.begin(), num_segments, init, op,
                                                     trash);
        }

        // A non-positive number of segments: the output is not written
        const std::vector<Offset> offsets{0, Offset(n)};
        Sequence<Out> out(1, [trash](std::size_t) { return trash; });
        invoke_on_all_policies<CallNumber * 3>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                                 out.end(), offsets.begin(), SegmentNum(0), init, op, trash);
        if constexpr (std::is_signed_v<SegmentNum>)
        {
            invoke_on_all_policies<CallNumber * 3>()(test_reduce_by_segment_offsets(), in.begin(), in.end(),
                                                     out.begin(), out.end(), offsets.begin(), SegmentNum(-1), init,
                                                     op, trash);
        }

        // Fixed-length segments; the last two ones cross or lie beyond the end of the input
        for (std::int64_t length : {1, 7, 1000})
        {
            auto fixed_offsets = oneapi::dpl::make_transform_iterator(
                oneapi::dpl::counting_iterator<std::int64_t>(0), segment_begin{length});
            const SegmentNum num_segments = SegmentNum(n / length + 2);
            Sequence<Out> fixed_out(num_segments, [trash](std::size_t) { return trash; });
            invoke_on_all_policies<CallNumber * 3 + 1>()(test_reduce_by_segment_offsets(), in.begin(), in.end(),
                                                         fixed_out.begin(), fixed_out.end(), fixed_offsets,
                                                         num_segments, init, op, trash);
        }

        // Read-only input
        if (n <= 1000)
        {
            const std::vector<Offset> random_offsets = make_offsets<Offset>(segments::random, n, gen);
            const SegmentNum num_segments = SegmentNum(random_offsets.size() - 1);
            Sequence<Out> random_out(num_segments, [trash](std::size_t) { return trash; });
            invoke_on_all_policies<CallNumber * 3 + 2>()(test_reduce_by_segment_offsets(), in.cbegin(), in.cend(),
                                                         random_out.begin(), random_out.end(),
                                                         random_offsets.begin(), num_segments, init, op, trash);
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

// The example of the RFC: predefined results for a small input, which also check the reference
template <std::size_t CallNumber>
void
test_rfc_example()
{
    const std::vector<int> input{10, 20, 5, 7, 4};
    const std::vector<int> offsets{0, 2, 2, 4, 5};
    const std::vector<int> expected{30, 0, 12, 4};

    std::vector<int> reference(expected.size());
    reduce_by_segment_offsets_serial(input.begin(), input.end(), offsets.begin(), 4, 0, std::plus<int>{},
                                     reference.begin());
    EXPECT_EQ_N(expected.begin(), reference.begin(), expected.size(), "wrong reference for the example of the RFC");

    Sequence<int> in(input.size(), [&input](std::size_t i) { return input[i]; });
    Sequence<int> out(expected.size(), [](std::size_t) { return -1; });
    invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                         out.end(), offsets.begin(), 4, 0, std::plus<int>{}, -1);
}

// Segments of the lengths at the boundaries of the implementations: the longest short segment and the shortest long one
// (512 on GPUs, 32768 on CPU devices and on the host), and long segments whose first chunk has one element or the whole
// chunk length (on GPUs the least power of two whose square is not less than the length of the segment, at least 64,
// 4096 on CPU devices, 16384 on the host); alone, and among enough short segments for an average length below 256,
// which takes another path on GPUs
template <std::size_t CallNumber>
void
test_boundaries()
{
    const std::int64_t lengths[] = {511, 512, 513, 4096, 4097, 16384, 16385, 32767, 32768, 32769, 49152, 49153};
    for (std::int64_t short_ones : {0, 100})
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

// Long segments only, which on GPUs take the reduction of their chunks by sub-groups, with operations whose identity is
// not a zero; the first chunks of the segments keep from none to all of the lanes of a sub-group busy
template <std::size_t CallNumber>
void
test_long_segments()
{
    const std::int64_t lengths[] = {1064, 255, 2053, 300, 4159, 30000, 65537, 70001, 200000};
    std::vector<std::int64_t> offsets{0};
    for (std::int64_t length : lengths)
        offsets.push_back(offsets.back() + length);
    const std::int64_t num_segments = std::int64_t(offsets.size() - 1);
    Sequence<int> positive(offsets.back(), [](std::size_t i) { return int((i * 7919) % 1000) + 1; });
    Sequence<int> negative(offsets.back(), [](std::size_t i) { return -int((i * 7919) % 1000) - 1; });
    Sequence<int> out(num_segments, [](std::size_t) { return 0; });
    invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), positive.begin(), positive.end(),
                                         out.begin(), out.end(), offsets.begin(), num_segments,
                                         std::numeric_limits<int>::max(), oneapi::dpl::minimum<int>{}, 0);
    invoke_on_all_policies<CallNumber + 1>()(test_reduce_by_segment_offsets(), negative.begin(), negative.end(),
                                             out.begin(), out.end(), offsets.begin(), num_segments,
                                             std::numeric_limits<int>::lowest(), oneapi::dpl::maximum<int>{}, 0);
}

// Segments ending one element past a power of two, which on GPUs leaves the next tile of the reduction by tiles a piece
// of one element of a segment begun before it, and empty segments at the end, which the last tile takes
template <std::size_t CallNumber>
void
test_one_past_powers_of_two(std::int64_t max_power)
{
    std::vector<std::int64_t> offsets{0};
    for (std::int64_t power = 4096; power <= max_power; power *= 2)
        offsets.push_back(power + 1);
    // ... and two empty segments at and past the end of the input
    const std::int64_t n = offsets.back();
    offsets.push_back(n);
    offsets.push_back(n + 5);
    const std::int64_t num_segments = std::int64_t(offsets.size() - 1);
    Sequence<int> in(n, [](std::size_t i) { return int(i % 7) + 1; });
    Sequence<std::int64_t> out(num_segments, [](std::size_t) { return -1; });
    invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                         out.end(), offsets.begin(), num_segments, std::int64_t{5},
                                         std::plus<std::int64_t>{}, std::int64_t{-1});
}

// As many segments as a work-group of the reduction of the chunks computes the ends of their chunks for in its local
// memory on GPUs, and one more, which takes a scan before it instead: with average lengths that take the reduction of
// the chunks by work-items, up to 2048 segments, and by sub-groups, up to 1024
template <std::size_t CallNumber>
void
test_many_segments()
{
    for (std::int64_t base_length : {257, 1024})
    {
        const std::int64_t local_ends = base_length < 1024 ? 2048 : 1024;
        for (std::int64_t num_segments : {local_ends, local_ends + 1})
        {
            std::vector<std::int64_t> offsets{0};
            for (std::int64_t i = 0; i < num_segments; ++i)
                offsets.push_back(offsets.back() + base_length + (i * 37) % 200);
            Sequence<int> in(offsets.back(), [](std::size_t i) { return int(i % 101) - 50; });
            Sequence<std::int64_t> out(num_segments, [](std::size_t) { return -1; });
            invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                                 out.end(), offsets.begin(), num_segments, std::int64_t{5},
                                                 std::plus<std::int64_t>{}, std::int64_t{-1});
        }
    }
}

// Larger inputs reduced one work-group per segment on GPUs: just up to the size reduced by one kernel, with as many
// segments as the path needs and one fewer, which take the chunks instead; past it, with the least average segment of
// the second kernel and one element less, and with that average but one segment fewer than the path needs; then
// segments longer than a work-group reduces alone among shorter ones, for the second kernel with the tiles of 8192
// elements of a GPU of at least 32 compute units: beginning at the start of a tile and inside one, one ending where
// the next begins inside a tile, and some leaving a last piece of one element
template <std::size_t CallNumber>
void
test_large_inputs_by_work_groups()
{
    const std::pair<std::int64_t, std::int64_t> sizes[] = {
        {65536 - 7, 100}, {65536 - 7, 99}, {2048 * 100, 100}, {2048 * 100 - 1, 100}, {2048 * 99, 99}};
    for (const auto& [n, num_segments] : sizes)
    {
        std::vector<std::int64_t> offsets{0};
        for (std::int64_t i = 1; i < num_segments; ++i)
            offsets.push_back(n * i / num_segments + (i * 37) % 11);
        offsets.push_back(n);
        Sequence<int> in(n, [](std::size_t i) { return int(i % 101) - 50; });
        Sequence<std::int64_t> out(num_segments, [](std::size_t) { return -1; });
        invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                             out.end(), offsets.begin(), num_segments, std::int64_t{5},
                                             std::plus<std::int64_t>{}, std::int64_t{-1});
    }

    const std::int64_t tile = 8192;
    std::vector<std::int64_t> offsets{0};
    auto add = [&offsets](std::int64_t length, std::int64_t count) {
        for (std::int64_t i = 0; i < count; ++i)
            offsets.push_back(offsets.back() + length);
    };
    // From the start of a tile, with a last piece of one element
    add(10 * tile + 1, 1);
    add(1000, 50);
    // From inside a tile, and the next one from inside the tile where the first one ends
    add(70001, 1);
    add(9 * tile + 5000, 1);
    add(300, 49);
    // Up to the start of a tile, then from there the shortest long segment, with a last piece of one element
    add(tile - offsets.back() % tile, 1);
    add(8 * tile + 1, 1);
    // Not a long segment: a last element past a tile
    add(3 * tile + 1, 1);
    add(17, 3);
    add(0, 1);
    const std::int64_t num_segments = std::int64_t(offsets.size() - 1);
    Sequence<int> in(offsets.back(), [](std::size_t i) { return int(i % 101) - 50; });
    Sequence<std::int64_t> out(num_segments, [](std::size_t) { return -1; });
    invoke_on_all_policies<CallNumber + 1>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                             out.end(), offsets.begin(), num_segments, std::int64_t{5},
                                             std::plus<std::int64_t>{}, std::int64_t{-1});
}

// The input is the middle of a larger sequence, whose other elements are poison: a read out of the bounds of the input
// changes a result
constexpr std::size_t guard_length = 1000;
constexpr int poison = 1000000;

// A few segments of many tiles each, whose carries are combined one sub-group per tile on GPUs, for both tile lengths:
// from the start of a tile with a last piece of one element, from inside a tile, up to the end of a tile, from the start
// of a tile to the end of the input, and two empty segments at and past the end of the input
template <std::size_t CallNumber>
void
test_long_chains()
{
    for (std::int64_t tile : {std::int64_t(8192), std::int64_t(16384)})
    {
        std::vector<std::int64_t> offsets{0};
        for (std::int64_t end : {20 * tile + 1, 50 * tile + 101, 100 * tile, 128 * tile})
            offsets.push_back(end);
        const std::int64_t n = offsets.back();
        offsets.push_back(n);
        offsets.push_back(n + 3);
        const std::int64_t num_segments = std::int64_t(offsets.size() - 1);
        Sequence<int> in(n, [](std::size_t i) { return int(i % 13) - 6; });
        Sequence<std::int64_t> out(num_segments, [](std::size_t) { return -1; });
        if (tile == 8192)
            invoke_on_all_policies<CallNumber>()(test_reduce_by_segment_offsets(), in.begin(), in.end(), out.begin(),
                                                 out.end(), offsets.begin(), num_segments, std::int64_t{5},
                                                 std::plus<std::int64_t>{}, std::int64_t{-1});
        else
            invoke_on_all_policies<CallNumber + 1>()(test_reduce_by_segment_offsets(), in.begin(), in.end(),
                                                     out.begin(), out.end(), offsets.begin(), num_segments,
                                                     std::int64_t{5}, std::plus<std::int64_t>{}, std::int64_t{-1});
    }
}

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
        for (segments kind : all_segments)
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

// ArgMin of the RFC: the zip of global indices and values is reduced, and the output turns the global index of
// the minimum into the index within its segment; ties are resolved to the smallest index
using argmin_idx_t = std::uint32_t;
using argmin_t = std::tuple<argmin_idx_t, float>;

struct argmin
{
    template <typename A, typename B>
    argmin_t
    operator()(const A& a, const B& b) const
    {
        const auto ia = std::get<0>(a);
        const auto ib = std::get<0>(b);
        const auto va = std::get<1>(a);
        const auto vb = std::get<1>(b);
        const bool take_b = vb < va || (!(va < vb) && ib < ia);
        return take_b ? argmin_t{ib, vb} : argmin_t{ia, va};
    }
};

struct global_to_local
{
    argmin_idx_t* out;
    argmin_idx_t begin;

    const global_to_local&
    operator=(argmin_idx_t global) const
    {
        *out = global - begin;
        return *this;
    }
};

struct make_global_to_local
{
    argmin_idx_t* idx_out;
    const argmin_idx_t* offsets;

    global_to_local
    operator()(argmin_idx_t segment) const
    {
        return global_to_local{idx_out + segment, offsets[segment]};
    }
};

// values, offsets, idx_out and val_out must be accessible with the policy
template <typename Policy>
void
test_argmin(Policy&& exec, float* values, argmin_idx_t n, argmin_idx_t* offsets, argmin_idx_t num_segments,
            argmin_idx_t* idx_out, float* val_out)
{
    std::mt19937 gen(7);
    std::uniform_int_distribution<int> value(0, 9); // many ties
    std::uniform_int_distribution<argmin_idx_t> length(0, 2 * n / num_segments);
    for (argmin_idx_t i = 0; i < n; ++i)
        values[i] = float(value(gen));
    offsets[0] = 0;
    for (argmin_idx_t s = 1; s <= num_segments; ++s)
        offsets[s] = std::min(n, offsets[s - 1] + length(gen));

    // Not the infinity of the RFC, which EXPECT_EQ cannot compare
    const argmin_t init{std::numeric_limits<argmin_idx_t>::max(), std::numeric_limits<float>::max()};
    auto in = oneapi::dpl::make_zip_iterator(oneapi::dpl::counting_iterator<argmin_idx_t>(0), values);
    std::vector<argmin_t> expected(num_segments);
    reduce_by_segment_offsets_serial(in, in + n, offsets, num_segments, init, argmin{}, expected.begin());

    auto local_idx_out = oneapi::dpl::make_transform_iterator(oneapi::dpl::counting_iterator<argmin_idx_t>(0),
                                                              make_global_to_local{idx_out, offsets});
    auto check = [&](bool check_values) {
        for (argmin_idx_t s = 0; s < num_segments; ++s)
        {
            // The index of an empty segment comes from the initial value and is meaningless
            if (offsets[s] != offsets[s + 1])
                EXPECT_EQ(std::get<0>(expected[s]) - offsets[s], idx_out[s], "wrong index from ArgMin");
            if (check_values)
                EXPECT_EQ(std::get<1>(expected[s]), val_out[s], "wrong value from ArgMin");
        }
    };

    // Index-value pairs
    oneapi::dpl::reduce_by_segment(CLONE_TEST_POLICY_IDX(exec, 0), in, in + n,
                                   oneapi::dpl::make_zip_iterator(local_idx_out, val_out), num_segments, offsets,
                                   init, argmin{});
    check(true);

    // Only the index
    std::fill(idx_out, idx_out + num_segments, argmin_idx_t(0));
    oneapi::dpl::reduce_by_segment(CLONE_TEST_POLICY_IDX(exec, 1), in, in + n,
                                   oneapi::dpl::make_zip_iterator(local_idx_out, oneapi::dpl::discard_iterator()),
                                   num_segments, offsets, init, argmin{});
    check(false);
}

void
test_argmin_on_host()
{
    const argmin_idx_t n = 10000, num_segments = 100;
    std::vector<float> values(n), val_out(num_segments);
    std::vector<argmin_idx_t> offsets(num_segments + 1), idx_out(num_segments);
    test_argmin(oneapi::dpl::execution::seq, values.data(), n, offsets.data(), num_segments, idx_out.data(),
                val_out.data());
    test_argmin(oneapi::dpl::execution::par_unseq, values.data(), n, offsets.data(), num_segments, idx_out.data(),
                val_out.data());
}

#if TEST_DPCPP_BACKEND_PRESENT
void
test_argmin_on_device()
{
    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();
    const argmin_idx_t n = 10000, num_segments = 100;
    float* values = sycl::malloc_shared<float>(n, q);
    float* val_out = sycl::malloc_shared<float>(num_segments, q);
    argmin_idx_t* offsets = sycl::malloc_shared<argmin_idx_t>(num_segments + 1, q);
    argmin_idx_t* idx_out = sycl::malloc_shared<argmin_idx_t>(num_segments, q);
    test_argmin(policy, values, n, offsets, num_segments, idx_out, val_out);
    sycl::free(values, q);
    sycl::free(val_out, q);
    sycl::free(offsets, q);
    sycl::free(idx_out, q);
}

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
    for (segments kind : all_segments)
    {
        std::vector<int> offsets = make_offsets<int>(kind, n, gen);
        const int num_segments = int(offsets.size() - 1);
        std::vector<int> expected(num_segments), result(num_segments);
        reduce_by_segment_offsets_serial(in.begin() + guard_length, in.begin() + guard_length + n, offsets.begin(),
                                         num_segments, 5, std::plus<int>{}, expected.begin());

        TestUtils::usm_data_transfer<alloc_type, int> in_usm(q, in.begin(), in.end());
        TestUtils::usm_data_transfer<alloc_type, int> offsets_usm(q, offsets.begin(), offsets.end());
        TestUtils::usm_data_transfer<alloc_type, int> out_usm(q, num_segments);
        oneapi::dpl::reduce_by_segment(CLONE_TEST_POLICY_NAME(policy, NewKernelName),
                                       in_usm.get_data() + guard_length, in_usm.get_data() + guard_length + n,
                                       out_usm.get_data(), num_segments, offsets_usm.get_data(), 5, std::plus<int>{});
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
    for (segments kind : all_segments)
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
            oneapi::dpl::reduce_by_segment(CLONE_TEST_POLICY_NAME(policy, KernelName),
                                           oneapi::dpl::begin(in_buf) + guard_length,
                                           oneapi::dpl::begin(in_buf) + guard_length + n, oneapi::dpl::begin(out_buf),
                                           num_segments, oneapi::dpl::begin(offsets_buf), 5, std::plus<int>{});
        }
        EXPECT_EQ_N(expected.begin(), result.begin(), num_segments,
                    "wrong result from reduce_by_segment with sycl::buffer");
    }
}

class KernelNameUSM;
class KernelNameBuffer;
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
    // The operations: the ones of the RFC (sum, min, max, product, logical, ArgMin), the accumulator types of its
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

    // The same set union with an initial value of the element type: it must be combined with the elements as an
    // element (the bit 7), while only an empty segment gets it converted to the accumulator (the bit pattern 7)
    test_segments<11, int, std::uint64_t, std::int64_t, int>(
        7, bit_set_union{}, [](std::size_t i) { return int(32 + (i * 7) % 32); }, std::uint64_t{0});

    // An accumulator which cannot keep the fraction of the initial value: an empty segment gets the initial value
    // converted to the accumulator type (2), not the initial value itself (2.5)
    test_segments<12, int, float, std::int64_t, int>(
        2.5f, int_sum_plus{}, [](std::size_t i) { return int(i % 101) - 50; }, -1.f);

    // Set union into an accumulator of an arithmetic type which can be made from an input element, but with another
    // meaning
    test_segments<13, int, std::uint64_t, std::uint32_t, std::int64_t>(
        std::uint64_t{1}, bit_mask_union{}, [](std::size_t i) { return int(32 + (i * 7) % 32); }, std::uint64_t{0});

    // Sum with an operation whose call operator is not const
    test_segments<17, int, std::int64_t, std::int64_t, int>(
        std::int64_t{5}, non_const_plus{}, [](std::size_t i) { return int(i % 101) - 50; }, std::int64_t{-1});

    test_unordered<42>();
    test_rfc_example<43>();
    test_boundaries<44>();
    test_input_bounds<45>();
    test_long_segments<46>();
    test_one_past_powers_of_two<48>(std::int64_t(1) << 19);
    test_one_past_powers_of_two<49>(std::int64_t(1) << 20);
    test_many_segments<50>();
    test_large_inputs_by_work_groups<54>();
    test_long_chains<56>();

    test_argmin_on_host();
#if TEST_DPCPP_BACKEND_PRESENT
    test_argmin_on_device();
    test_usm<sycl::usm::alloc::shared, KernelNameUSM>();
    test_usm<sycl::usm::alloc::device, KernelNameUSM>();
    test_buffer<KernelNameBuffer>();
#endif

    return done();
}
