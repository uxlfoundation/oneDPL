// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2026 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

// Checks of oneapi::dpl::__unseq_backend::__simd_find_first_of for iterators with a narrow difference type.
// The brick splits the first sequence into blocks whose size grows up to a limit; neither the block sizes nor
// the block ends may overflow the difference type, which for counting_iterator<std::int8_t> cannot hold even
// the smallest block and for counting_iterator<std::int16_t> would overflow at the end of the last block. With a wide
// value type the smallest block fits into std::int8_t, but twice the next block does not.
// A second sequence longer than 64 tiles of 16 KB is walked in tiles, each tile against the elements of the first
// sequence that can still improve the result; a match of a later tile must still win when it is at an earlier position.

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/pstl/unseq_backend_simd.h>

#include <algorithm>   // for std::min
#include <cstddef>     // for std::size_t
#include <cstdint>     // for std::int8_t, std::int16_t, std::int64_t
#include <functional>  // for std::equal_to
#include <iterator>    // for std::iterator_traits
#include <limits>      // for std::numeric_limits
#include <type_traits> // for std::is_same_v
#include <vector>      // for std::vector

#include "support/utils.h"

namespace dpl_unseq = oneapi::dpl::__unseq_backend;

// Compares the position found by the brick with the expected one, which the test knows by construction
// (a reference search would double the run time); the positions are compared as long long, so that an 8-bit
// difference type is not printed as a character
template <typename It1, typename It2>
void
check(It1 first, It1 last, It2 s_first, It2 s_last, long long expected, const char* message)
{
    const auto result = dpl_unseq::__simd_find_first_of(first, last, s_first, s_last, std::equal_to<>{});
    EXPECT_EQ(expected, static_cast<long long>(result - first), message);
}

// The second sequence: `count` values that are not in the first sequence (which holds 0, 1, ..., n1 - 1),
// followed by the value to find, if any - so a match is found only through the last element
template <typename T>
std::vector<T>
make_second(std::size_t count, bool with_match, T value)
{
    std::vector<T> s;
    for (std::size_t i = 0; i < count; ++i)
        s.push_back(T(-1 - static_cast<long long>(i % std::size_t(std::numeric_limits<T>::max()))));
    if (with_match)
        s.push_back(value);
    return s;
}

// `first` points to the values 0, 1, 2, ... of a sequence with the difference type T; `long_n2` is the length
// of a second sequence longer than the largest block, or 0 to skip such checks
template <typename T, typename It1>
void
test_narrow_difference_type(It1 first, std::size_t long_n2)
{
    using It = oneapi::dpl::counting_iterator<T>;
    static_assert(std::is_same_v<typename std::iterator_traits<It1>::difference_type, T>);

    // The largest first sequence the difference type can describe
    const T n1 = std::numeric_limits<T>::max();
    const It1 last = first + n1;

    // Match positions: every position for a small n1, otherwise both ends of the sequence and a sparse sweep
    std::vector<T> positions;
    for (long long p = 0; p < n1; ++p)
        if (n1 <= 256 || p < 150 || p >= n1 - 150 || p % 257 == 0)
            positions.push_back(T(p));

    for (const T p : positions)
    {
        // The second sequence shorter than any block
        const std::vector<T> s = make_second(3, true, p);
        check(first, last, s.begin(), s.end(), p, "wrong position of a match through the last element of a short range 2");

        // The second sequence with a narrow difference type as well
        const It s_first(p);
        const It s_last = s_first + T(std::min<long long>(3, n1 - p));
        check(first, last, s_first, s_last, p, "wrong position of a match with narrow difference types of both ranges");
    }

    // No match
    for (const std::size_t n2 : {std::size_t(1), std::size_t(3)})
    {
        const std::vector<T> s = make_second(n2, false, T(0));
        check(first, last, s.begin(), s.end(), n1, "a match found where there is none");
    }

    if (long_n2 == 0)
        return;

    // The second sequence longer than the largest block: a match at both ends of the first sequence and no match
    for (const T p : {T(0), T(n1 / 2), T(n1 - 1)})
    {
        const std::vector<T> s = make_second(long_n2, true, p);
        check(first, last, s.begin(), s.end(), p, "wrong position of a match through the last element of a long range 2");
    }
    const std::vector<T> s = make_second(long_n2, false, T(0));
    check(first, last, s.begin(), s.end(), n1, "a match found where there is none with a long range 2");
}

// A match of a[p] with the element s[j] of a second sequence longer than 64 tiles of 16 KB / sizeof(T): j at the tile
// boundaries, p across the blocks of the first sequence, which start from a single element for such a second sequence.
// The predicate is asymmetric, so that swapped arguments are detected as well: a[i] matches s[j] when a[i] == s[j] + 1
template <typename T>
void
test_tiled_second_sequence()
{
    const std::size_t tile = 16 * 1024 / sizeof(T);
    const std::size_t n1 = 40;
    const std::size_t n2 = 65 * tile + 3;
    auto pred = [](T a, T s) { return a == T(s + 1); };

    // The first sequence holds 1000, 1001, ...; the filler of the second one (0) matches none of them
    std::vector<T> a(n1);
    for (std::size_t i = 0; i < n1; ++i)
        a[i] = T(1000 + i);
    std::vector<T> s(n2, T(0));

    auto run = [&](long long expected, const char* message) {
        const auto result = dpl_unseq::__simd_find_first_of(a.begin(), a.end(), s.begin(), s.end(), pred);
        EXPECT_EQ(expected, static_cast<long long>(result - a.begin()), message);
    };

    run(n1, "a match found where there is none with a tiled range 2");
    for (const std::size_t j : {std::size_t(0), tile - 1, tile, 2 * tile - 1, 2 * tile, n2 - 1})
    {
        for (const std::size_t p : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(7), std::size_t(8),
                                    std::size_t(20), n1 - 1})
        {
            s[j] = T(a[p] - 1);
            run(p, "wrong position of a match with a tiled range 2");

            // An earlier tile refers to a later position: the match of the later tile must win
            if (p > 0 && j >= tile)
            {
                s[j - tile] = T(a[p] - 1 + 1);
                run(p, "the earlier tile of range 2 won over the earlier position of range 1");
                s[j - tile] = T(0);
            }
            s[j] = T(0);
        }
    }
}

int
main()
{
    // The second size 200 is larger than the largest block of 127 elements. With std::int16_t a second sequence
    // longer than the largest block (8192 elements) would cost hundreds of millions of comparisons, while the
    // blocks shorter than the second sequence are already covered with std::int8_t
    test_narrow_difference_type<std::int8_t>(oneapi::dpl::counting_iterator<std::int8_t>(0), 200);
    test_narrow_difference_type<std::int16_t>(oneapi::dpl::counting_iterator<std::int16_t>(0), 0);

    // 8-byte values: the blocks of 32, 64 and 127 elements
    auto widen = [](std::int8_t x) { return std::int64_t(x); };
    test_narrow_difference_type<std::int8_t>(
        oneapi::dpl::make_transform_iterator(oneapi::dpl::counting_iterator<std::int8_t>(0), widen), 200);

    // The second sequence longer than 64 tiles
    test_tiled_second_sequence<std::int32_t>();
    test_tiled_second_sequence<std::int64_t>();

    return TestUtils::done();
}
