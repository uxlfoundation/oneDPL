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
// The requirements are default construction, copy construction and copy assignment, and nothing else.
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::CopyOnlyNoMoveCompare*>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::VoidAssignCompare*>);
static_assert(dpl_unseq::__is_value_storable_v<const TestUtils::ConstCopyOnlyCompare*>);
// The reference type is not part of the requirement.
static_assert(dpl_unseq::__is_value_storable_v<std::vector<bool>::iterator>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<int, int>>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<std::pair<int, int>, std::pair<int&, int&>>>);
static_assert(
    dpl_unseq::__is_value_storable_v<FakeIterator<TestUtils::CopyOnlyNoMoveCompare, TestUtils::CopyOnlyNoMoveCompare>>);
// Accepted although the bricks do not compile for them: these iterators do not meet the requirements of a forward
// iterator, which is not detected here.
static_assert(std::is_copy_constructible_v<TestUtils::ExplicitCopyCtorCompare>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::ExplicitCopyCtorCompare*>);
static_assert(dpl_unseq::__is_value_storable_v<TestUtils::ConstCopyOnlyCompare*>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<int, OpaqueRef>>);

// Rejected because of the value type: default construction (the first two), copy assignment, copy construction.
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::NoDefaultCtorWrapper<int>*>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::BraceInitOnlyCompare*>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::NoCopyAssignCompare*>);
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::MoveOnlyCompare*>);

// Rejected because an output iterator reports void as its value type.
static_assert(!dpl_unseq::__is_value_storable_v<std::back_insert_iterator<std::vector<int>>>);

int
main()
{
    return TestUtils::done();
}
