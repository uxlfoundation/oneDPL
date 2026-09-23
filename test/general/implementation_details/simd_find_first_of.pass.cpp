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

// `first` points to the values 0, 1, 2, ... of a sequence with the difference type T
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

    // The second sequence longer than the largest block, a match at both ends of the first sequence. Each such
    // check away from the start scans up to the whole first sequence once per element of the second one, so
    // a debug build keeps only the match at the start: the full scan is still done by the no-match check below
#if PSTL_USE_DEBUG
    for (const T p : {T(0)})
#else
    for (const T p : {T(0), T(n1 / 2), T(n1 - 1)})
#endif
    {
        const std::vector<T> s = make_second(long_n2, true, p);
        check(first, last, s.begin(), s.end(), p, "wrong position of a match through the last element of a long range 2");
    }

    // No match
    for (const std::size_t n2 : {std::size_t(1), std::size_t(3), long_n2})
    {
        const std::vector<T> s = make_second(n2, false, T(0));
        check(first, last, s.begin(), s.end(), n1, "a match found where there is none");
    }
}

int
main()
{
    // The second sizes are larger than the largest block of the first sequence (127 and 8192 elements)
    test_narrow_difference_type<std::int8_t>(oneapi::dpl::counting_iterator<std::int8_t>(0), 200);
    test_narrow_difference_type<std::int16_t>(oneapi::dpl::counting_iterator<std::int16_t>(0), 8300);

    // 8-byte values: the blocks of 32, 64 and 127 elements
    auto widen = [](std::int8_t x) { return std::int64_t(x); };
    test_narrow_difference_type<std::int8_t>(
        oneapi::dpl::make_transform_iterator(oneapi::dpl::counting_iterator<std::int8_t>(0), widen), 200);

    return TestUtils::done();
}
