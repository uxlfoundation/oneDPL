// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2025 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

// Compile-time checks for oneapi::dpl::__unseq_backend::__is_value_storable_v, the condition that selects the vector
// code path of min_element and minmax_element.

#include "support/test_config.h"

#include <oneapi/dpl/pstl/unseq_backend_simd.h>

#include <cstddef>     // for std::ptrdiff_t
#include <cstdint>     // for std::int32_t
#include <functional>  // for std::less
#include <iterator>    // for std::random_access_iterator_tag, std::back_insert_iterator
#include <type_traits> // for std::is_default_constructible_v, std::is_copy_constructible_v
#include <utility>     // for std::pair
#include <vector>      // for std::vector

#include "support/utils.h"

namespace dpl_unseq = oneapi::dpl::__unseq_backend;

//----------------------------------------------------------------------------//
// Reference types
//----------------------------------------------------------------------------//

// A reference type that does not convert to the value type.
struct OpaqueRef
{
};

template <typename _ValueType, typename _ReferenceType>
struct FakeIterator
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = _ValueType;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = _ReferenceType;

    reference
    operator*() const;
};

// An iterator whose reference narrows to its value type; the bricks are instantiated for it in main().
struct NarrowingIterator
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::int32_t;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = double;

    const double* ptr;

    reference
    operator*() const
    {
        return *ptr;
    }
    reference
    operator[](difference_type __i) const
    {
        return ptr[__i];
    }
    NarrowingIterator
    operator+(difference_type __i) const
    {
        return NarrowingIterator{ptr + __i};
    }
    difference_type
    operator-(const NarrowingIterator& __other) const
    {
        return ptr - __other.ptr;
    }
};

//----------------------------------------------------------------------------//
// __is_value_storable_v
//----------------------------------------------------------------------------//

// Accepted value types.
static_assert(dpl_unseq::__is_value_storable_v<int*>);
static_assert(dpl_unseq::__is_value_storable_v<const int*>);
static_assert(dpl_unseq::__is_value_storable_v<std::vector<int>::iterator>);
static_assert(dpl_unseq::__is_value_storable_v<std::vector<int>::const_iterator>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::OnlyLessCompare*>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::ExplicitDefaultCtorCompare*>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::AggregateOfExplicitDefaultCtorCompare*>);
// Default construction is not required: the reduction object is always built from a value.
static_assert(!std::is_default_constructible_v<TestUtils::NoDefaultCtorWrapper<int>>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::NoDefaultCtorWrapper<int>*>);
static_assert(!std::is_default_constructible_v<TestUtils::BraceInitOnlyCompare>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::BraceInitOnlyCompare*>);
// The requirements are copy construction, copy assignment and copy-initialization from the reference type, and nothing
// else - in particular the move operations are not required.
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::CopyOnlyNoMoveCompare*>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::VoidAssignCompare*>);
static_assert(dpl_unseq::__is_value_storable_v<const TestUtils::ConstCopyOnlyCompare*>);
// A proxy reference is accepted as long as the value type can be copy-initialized from it.
static_assert(dpl_unseq::__is_value_storable_v<std::vector<bool>::iterator>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<int, int>>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<std::pair<int, int>, std::pair<int&, int&>>>);
// Narrowing is accepted: the bricks copy-initialize the value, they do not list-initialize it.
static_assert(dpl_unseq::__is_value_storable_v<NarrowingIterator>);

// Rejected because of the value type: copy assignment, copy construction.
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::NoCopyAssignCompare*>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::MoveOnlyCompare*>);

// Rejected because the value cannot be copy-initialized from what the iterator dereferences to, which is how the
// bricks read an element - copy-constructibility of the value type alone does not imply that.
static_assert(std::is_copy_constructible_v<TestUtils::ExplicitCopyCtorCompare>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::ExplicitCopyCtorCompare*>);
static_assert(std::is_copy_constructible_v<TestUtils::ConstCopyOnlyCompare>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::ConstCopyOnlyCompare*>);
static_assert(!dpl_unseq::__is_value_storable_v<FakeIterator<int, OpaqueRef>>);

// Rejected conservatively: the conversion is checked from an rvalue of the reference type, while the bricks initialize
// from *__first, which is a prvalue here and needs no move constructor; such an iterator only misses vectorization.
static_assert(!dpl_unseq::__is_value_storable_v<
              FakeIterator<TestUtils::CopyOnlyNoMoveCompare, TestUtils::CopyOnlyNoMoveCompare>>);

// Rejected because an output iterator reports void as its value type.
static_assert(!dpl_unseq::__is_value_storable_v<std::back_insert_iterator<std::vector<int>>>);

int
main()
{
#if _ONEDPL_UDR_PRESENT
    // Instantiating the bricks for NarrowingIterator catches a switch of their reduction object back to
    // list-initialization at compile time. Truncated to the value type the data is {3, 1, 2, 1}, so the minimum is the
    // first of the two ones and the maximum is the single three.
    const double __data[] = {3.5, 1.25, 2.75, 1.75};
    const NarrowingIterator __first{__data};
    const std::ptrdiff_t __n = sizeof(__data) / sizeof(__data[0]);

    EXPECT_EQ(1, dpl_unseq::__simd_min_element(__first, __n, std::less<>{}) - __first,
              "wrong __simd_min_element on NarrowingIterator");
    const auto __minmax = dpl_unseq::__simd_minmax_element(__first, __n, std::less<>{});
    EXPECT_EQ(1, __minmax.first - __first, "wrong minimum from __simd_minmax_element on NarrowingIterator");
    EXPECT_EQ(0, __minmax.second - __first, "wrong maximum from __simd_minmax_element on NarrowingIterator");
#endif
    return TestUtils::done();
}
