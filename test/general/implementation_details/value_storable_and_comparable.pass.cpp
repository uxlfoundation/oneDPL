// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2025 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

// Compile-time checks for oneapi::dpl::__unseq_backend::__is_value_storable_and_comparable_v and for the requirements
// it is built from.
// The only one of them that is not a standard type trait, oneapi::dpl::__unseq_backend::__is_brace_constructible_v, is
// checked on its own as well. Every requirement is checked both ways: a type that satisfies it and a type that does not.

#include "support/test_config.h"

#include <oneapi/dpl/pstl/unseq_backend_simd.h>
#include <oneapi/dpl/pstl/utils.h>

#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "support/utils.h"

namespace dpl_internal = oneapi::dpl::__internal;
namespace dpl_unseq = oneapi::dpl::__unseq_backend;

//----------------------------------------------------------------------------//
// Value types
//----------------------------------------------------------------------------//

// Satisfies every requirement: default-constructible, copy-constructible, copy-assignable, less-than comparable.
struct Regular
{
    int val = 0;
    bool
    operator<(const Regular& other) const
    {
        return val < other.val;
    }
};

// An explicit default constructor is enough, since _ValueType{} is a direct initialization, which may use it.
struct ExplicitDefaultCtor
{
    int val;
    explicit ExplicitDefaultCtor() : val(0) {}
    bool
    operator<(const ExplicitDefaultCtor& other) const
    {
        return val < other.val;
    }
};

struct ExplicitDefaultCtorMember
{
    int val;
    explicit ExplicitDefaultCtorMember() : val(0) {}
};

// Default-constructible, but not brace-initializable: an aggregate is initialized member by member, and the member is
// copy-initialized from an empty list, which may not use its explicit default constructor.
struct AggregateOfExplicitDefaultCtor
{
    ExplicitDefaultCtorMember member;
    bool
    operator<(const AggregateOfExplicitDefaultCtor& other) const
    {
        return member.val < other.member.val;
    }
};

struct NoDefaultCtor
{
    int val;
    explicit NoDefaultCtor(int v) : val(v) {}
    bool
    operator<(const NoDefaultCtor& other) const
    {
        return val < other.val;
    }
};

struct NoCopyAssign
{
    int val = 0;
    NoCopyAssign() = default;
    NoCopyAssign(const NoCopyAssign&) = default;
    NoCopyAssign&
    operator=(const NoCopyAssign&) = delete;
    bool
    operator<(const NoCopyAssign& other) const
    {
        return val < other.val;
    }
};

// The assignment does not return VoidAssign&, which is enough here because the result is never used.
struct VoidAssign
{
    int val = 0;
    void
    operator=(const VoidAssign& other)
    {
        val = other.val;
    }
    bool
    operator<(const VoidAssign& other) const
    {
        return val < other.val;
    }
};

// The destructor is not noexcept, which is enough here because storing a value never has to be non-throwing.
struct ThrowingDtor
{
    int val = 0;
    ~ThrowingDtor() noexcept(false) {}
    bool
    operator<(const ThrowingDtor& other) const
    {
        return val < other.val;
    }
};

struct MoveOnly
{
    int val = 0;
    MoveOnly() = default;
    MoveOnly(MoveOnly&&) = default;
    MoveOnly&
    operator=(MoveOnly&&) = default;
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly&
    operator=(const MoveOnly&) = delete;
    bool
    operator<(const MoveOnly& other) const
    {
        return val < other.val;
    }
};

// Deleting the move operations while keeping the copy ones is enough here, because the value is never moved.
struct CopyOnlyNoMove
{
    int val = 0;
    CopyOnlyNoMove() = default;
    CopyOnlyNoMove(const CopyOnlyNoMove&) = default;
    CopyOnlyNoMove&
    operator=(const CopyOnlyNoMove&) = default;
    CopyOnlyNoMove(CopyOnlyNoMove&&) = delete;
    CopyOnlyNoMove&
    operator=(CopyOnlyNoMove&&) = delete;
    bool
    operator<(const CopyOnlyNoMove& other) const
    {
        return val < other.val;
    }
};

// Copyable and assignable from a const lvalue only, which is enough here, because the candidates are read through
// std::as_const and the element is materialized as a const _ValueType.
struct ConstCopyOnly
{
    int val = 0;
    ConstCopyOnly() = default;
    ConstCopyOnly(const ConstCopyOnly&) = default;
    ConstCopyOnly&
    operator=(const ConstCopyOnly&) = default;
    ConstCopyOnly(ConstCopyOnly&) = delete;
    ConstCopyOnly&
    operator=(ConstCopyOnly&) = delete;
    bool
    operator<(const ConstCopyOnly& other) const
    {
        return val < other.val;
    }
};

// A value type whose copy constructor is explicit, which is enough for copying the candidates, because they are copied
// by direct initialization, and so is std::is_copy_constructible_v defined. Copy-initializing an element of such a type
// is ill-formed, so an iterator over it does not meet the requirements of a forward iterator.
struct ExplicitCopyCtor
{
    int val = 0;
    ExplicitCopyCtor() = default;
    explicit ExplicitCopyCtor(const ExplicitCopyCtor& other) : val(other.val) {}
    ExplicitCopyCtor&
    operator=(const ExplicitCopyCtor&) = default;
    bool
    operator<(const ExplicitCopyCtor&) const
    {
        return false;
    }
};

//----------------------------------------------------------------------------//
// __is_brace_constructible_v
//----------------------------------------------------------------------------//

static_assert(dpl_unseq::__is_brace_constructible_v<int>);
static_assert(dpl_unseq::__is_brace_constructible_v<int*>);
static_assert(dpl_unseq::__is_brace_constructible_v<Regular>);
static_assert(dpl_unseq::__is_brace_constructible_v<ExplicitDefaultCtor>);
static_assert(dpl_unseq::__is_brace_constructible_v<MoveOnly>);

static_assert(std::is_default_constructible_v<AggregateOfExplicitDefaultCtor>);
static_assert(!dpl_unseq::__is_brace_constructible_v<AggregateOfExplicitDefaultCtor>);
static_assert(!dpl_unseq::__is_brace_constructible_v<NoDefaultCtor>);

//----------------------------------------------------------------------------//
// Comparison objects
//----------------------------------------------------------------------------//

struct NotBool
{
};

struct IntResultLess
{
    int
    operator()(const int& lhs, const int& rhs) const
    {
        return lhs < rhs;
    }
};

struct NotBoolResultLess
{
    NotBool
    operator()(const int&, const int&) const
    {
        return NotBool{};
    }
};

// Requires modifiable arguments, so it cannot be called on const values.
struct MutableRefLess
{
    bool
    operator()(int& lhs, int& rhs) const
    {
        return lhs < rhs;
    }
};

// Callable on an rvalue only, while the requirement is stated for _Compare&.
struct RvalueOnlyLess
{
    bool
    operator()(const int&, const int&) &&
    {
        return false;
    }
};

struct UnaryLess
{
    bool
    operator()(const int&) const
    {
        return false;
    }
};

// Not copyable: the requirement is stated for _Compare&, so it must not ask for a copy.
struct MoveOnlyLess
{
    MoveOnlyLess() = default;
    MoveOnlyLess(MoveOnlyLess&&) = default;
    MoveOnlyLess(const MoveOnlyLess&) = delete;
    bool
    operator()(const int& lhs, const int& rhs) const
    {
        return lhs < rhs;
    }
};

struct NotNegatableResult
{
    operator bool() const;
    bool
    operator!() const = delete;
};

// The result of the comparison is convertible to bool, but cannot be negated, while the vectorized bricks do negate it.
// Such a comparison object does not meet the Compare requirements the standard states for the algorithms, so it is not
// rejected here: the requirement accepts it in both C++17 and C++20, and instantiating the brick for it is a compile
// error rather than a fallback to the serial implementation.
struct NotNegatableResultLess
{
    NotNegatableResult
    operator()(const int& lhs, const int& rhs) const;
};

//----------------------------------------------------------------------------//
// Reference types
//----------------------------------------------------------------------------//

// A reference type that does not convert to the value type: an iterator reporting it does not meet the requirements of
// a forward iterator, which state that *__first is convertible to the value type, so the requirement does not look at
// the reference type at all.
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
// __is_value_storable_and_comparable_v
//----------------------------------------------------------------------------//

// Accepted: the value type is storable and the comparator is callable on const values.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, std::less<int>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<const int*, std::less<int>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<std::vector<int>::iterator, std::less<int>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<std::vector<int>::const_iterator, std::less<>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<Regular*, std::less<Regular>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<ExplicitDefaultCtor*, std::less<ExplicitDefaultCtor>>);
// Accepted: the requirements are brace initialization, copy construction and copy assignment, and nothing else.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<CopyOnlyNoMove*, std::less<CopyOnlyNoMove>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<VoidAssign*, std::less<VoidAssign>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<ThrowingDtor*, std::less<ThrowingDtor>>);
// Accepted: copying and storing a value only ever reads it through a const reference.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<const ConstCopyOnly*, std::less<ConstCopyOnly>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, MoveOnlyLess>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, IntResultLess>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, bool (*)(const int&, const int&)>);
// Accepted although the bricks do not compile for it: a comparison object that does not meet the Compare requirements
// of the algorithms is not detected here, see NotNegatableResultLess above.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, NotNegatableResultLess>);
// The comparator max_element passes down to the min_element brick.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, dpl_internal::__reorder_pred<std::less<int>>>);
// The reference type is not part of the requirement, so a proxy reference and an iterator returning the value type by
// value are accepted like any other.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<std::vector<bool>::iterator, std::less<bool>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<FakeIterator<int, int>, std::less<int>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<FakeIterator<std::pair<int, int>, std::pair<int&, int&>>,
                                                             std::less<std::pair<int, int>>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<FakeIterator<CopyOnlyNoMove, CopyOnlyNoMove>,
                                                             std::less<CopyOnlyNoMove>>);
// Accepted although the bricks do not compile for them: an element of these iterators cannot be copy-initialized into
// the value type, so they do not meet the requirements of a forward iterator, which is not detected here either. The
// value types themselves are copy-constructible, which is stated in terms of direct initialization.
static_assert(std::is_copy_constructible_v<ExplicitCopyCtor>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<ExplicitCopyCtor*, std::less<ExplicitCopyCtor>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<ConstCopyOnly*, std::less<ConstCopyOnly>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<FakeIterator<int, OpaqueRef>, std::less<int>>);

// Rejected because of the value type: the first two fail brace initialization, the third copy assignment, and the
// move-only one copy construction, and with it every other requirement that copies a value.
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<NoDefaultCtor*, std::less<NoDefaultCtor>>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<AggregateOfExplicitDefaultCtor*,
                                                              std::less<AggregateOfExplicitDefaultCtor>>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<NoCopyAssign*, std::less<NoCopyAssign>>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<MoveOnly*, std::less<MoveOnly>>);

// Rejected because an output iterator reports void as its value type.
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<std::back_insert_iterator<std::vector<int>>,
                                                              std::less<int>>);

// Rejected because of the comparator.
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, NotBoolResultLess>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, MutableRefLess>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, RvalueOnlyLess>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, UnaryLess>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, std::less<Regular>>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, int>);

int
main()
{
    return TestUtils::done();
}
