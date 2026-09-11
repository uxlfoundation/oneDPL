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

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "support/utils.h"

namespace dpl_unseq = oneapi::dpl::__unseq_backend;

// The value types with restricted operations are defined in test/support/utils.h, so that
// test/parallel_api/algorithm/alg.sorting/alg.min.max/minmax_element.pass.cpp runs the algorithms on the very same set
// that is checked against the trait here. TestUtils::NoDefaultCtorWrapper<int> is the type that is not
// default-constructible.

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

// Rejected because of the value type: copy assignment, copy construction.
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::NoCopyAssignCompare*>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::MoveOnlyCompare*>);

// Rejected because the value cannot be copy-initialized from what the iterator dereferences to, which is how the
// bricks read an element. Copy-constructibility of the value type alone does not imply that: it also holds for an
// explicit copy constructor and for a type whose copy constructor is deleted for a non-const lvalue.
static_assert(std::is_copy_constructible_v<TestUtils::ExplicitCopyCtorCompare>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::ExplicitCopyCtorCompare*>);
static_assert(std::is_copy_constructible_v<TestUtils::ConstCopyOnlyCompare>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::ConstCopyOnlyCompare*>);
static_assert(!dpl_unseq::__is_value_storable_v<FakeIterator<int, OpaqueRef>>);

// Rejected conservatively: the conversion is checked from an rvalue of the reference type, while the bricks initialize
// from *__first, which is a prvalue here and needs no move constructor. Such an iterator stays correct through the
// serial fallback, it only misses vectorization.
static_assert(!dpl_unseq::__is_value_storable_v<
              FakeIterator<TestUtils::CopyOnlyNoMoveCompare, TestUtils::CopyOnlyNoMoveCompare>>);

// Rejected because an output iterator reports void as its value type.
static_assert(!dpl_unseq::__is_value_storable_v<std::back_insert_iterator<std::vector<int>>>);

int
main()
{
    return TestUtils::done();
}
