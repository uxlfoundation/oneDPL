// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2025 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

// Compile-time checks for oneapi::dpl::__internal::__is_value_storable_and_comparable_v and for each of the
// requirements it is built from: __convertible_to_v, __semiregular_v and __predicate_v, plus the C++17 building blocks
// of __semiregular_v (__constructible_from, __move_constructible, __copy_constructible, __assignable_from, __movable,
// __copyable). Every requirement is checked both ways: a type that satisfies it and a type that does not.

#include "support/test_config.h"

#include <oneapi/dpl/pstl/utils.h>

#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "support/utils.h"

namespace dpl_internal = oneapi::dpl::__internal;

//----------------------------------------------------------------------------//
// Value types
//----------------------------------------------------------------------------//

// Satisfies every requirement: default-constructible, copyable, less-than comparable.
struct Regular
{
    int val = 0;
    bool
    operator<(const Regular& other) const
    {
        return val < other.val;
    }
};

// std::default_initializable accepts an explicit default constructor, since T{} stays valid.
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

// std::is_assignable_v is satisfied, but the assignment does not return VoidAssign&.
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

// std::destructible, and hence __constructible_from, requires the destructor to be noexcept.
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

//----------------------------------------------------------------------------//
// __convertible_to_v
//----------------------------------------------------------------------------//

struct ExplicitFromInt
{
    explicit ExplicitFromInt(int) {}
};

// A destination whose only constructor taking ImplicitSource is deleted and explicit: copy-initialization ignores it
// and picks the conversion operator, so std::is_convertible_v is satisfied, while static_cast selects the deleted
// constructor. This is the difference std::convertible_to catches and std::is_convertible_v does not.
struct ExplicitlyNotConvertible;

struct ImplicitSource
{
    operator ExplicitlyNotConvertible() const;
};

struct ExplicitlyNotConvertible
{
    ExplicitlyNotConvertible() = default;
    explicit ExplicitlyNotConvertible(ImplicitSource) = delete;
};

static_assert(dpl_internal::__convertible_to_v<int, int>);
static_assert(dpl_internal::__convertible_to_v<const int&, int>);
static_assert(dpl_internal::__convertible_to_v<int&, long>);
static_assert(dpl_internal::__convertible_to_v<const Regular&, Regular>);
static_assert(dpl_internal::__convertible_to_v<std::pair<int&, int&>, std::pair<int, int>>);

static_assert(!dpl_internal::__convertible_to_v<int*, int>);
static_assert(!dpl_internal::__convertible_to_v<Regular, int>);
static_assert(!dpl_internal::__convertible_to_v<int, ExplicitFromInt>);
static_assert(!dpl_internal::__convertible_to_v<const MoveOnly&, MoveOnly>);
static_assert(std::is_convertible_v<ImplicitSource, ExplicitlyNotConvertible>);
static_assert(!dpl_internal::__convertible_to_v<ImplicitSource, ExplicitlyNotConvertible>);

//----------------------------------------------------------------------------//
// __semiregular_v
//----------------------------------------------------------------------------//

static_assert(dpl_internal::__semiregular_v<int>);
static_assert(dpl_internal::__semiregular_v<int*>);
static_assert(dpl_internal::__semiregular_v<Regular>);
static_assert(dpl_internal::__semiregular_v<ExplicitDefaultCtor>);
static_assert(dpl_internal::__semiregular_v<std::pair<int, int>>);

static_assert(!dpl_internal::__semiregular_v<NoDefaultCtor>);
static_assert(!dpl_internal::__semiregular_v<NoCopyAssign>);
static_assert(!dpl_internal::__semiregular_v<VoidAssign>);
static_assert(!dpl_internal::__semiregular_v<ThrowingDtor>);
static_assert(!dpl_internal::__semiregular_v<MoveOnly>);
static_assert(!dpl_internal::__semiregular_v<int&>);
// Output iterators report void as their value type, so void must be rejected rather than rejecting the program.
static_assert(!dpl_internal::__semiregular_v<void>);
static_assert(!dpl_internal::__semiregular_v<const void>);

//----------------------------------------------------------------------------//
// __predicate_v
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

static_assert(dpl_internal::__predicate_v<std::less<int>&, const int&, const int&>);
static_assert(dpl_internal::__predicate_v<std::less<>&, const int&, const int&>);
static_assert(dpl_internal::__predicate_v<IntResultLess&, const int&, const int&>);
static_assert(dpl_internal::__predicate_v<MoveOnlyLess&, const int&, const int&>);
static_assert(dpl_internal::__predicate_v<std::less<Regular>&, const Regular&, const Regular&>);

static_assert(!dpl_internal::__predicate_v<NotBoolResultLess&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<MutableRefLess&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<RvalueOnlyLess&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<UnaryLess&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<int&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<std::less<int>&, const Regular&, const Regular&>);

//----------------------------------------------------------------------------//
// C++17 building blocks of __semiregular_v. In C++20 the standard concepts are used directly, so these helpers only
// exist in the C++17 branch.
//----------------------------------------------------------------------------//

#if !_ONEDPL_CPP20_CONCEPTS_PRESENT

static_assert(dpl_internal::__constructible_from<int, int>);
static_assert(dpl_internal::__constructible_from<Regular>);
static_assert(dpl_internal::__constructible_from<NoDefaultCtor, int>);
static_assert(!dpl_internal::__constructible_from<NoDefaultCtor>);
static_assert(!dpl_internal::__constructible_from<ThrowingDtor>);
static_assert(!dpl_internal::__constructible_from<Regular, int>);

static_assert(dpl_internal::__assignable_from<int, int>);
static_assert(dpl_internal::__assignable_from<Regular, const Regular&>);
static_assert(dpl_internal::__assignable_from<MoveOnly, MoveOnly>);
static_assert(!dpl_internal::__assignable_from<VoidAssign, const VoidAssign&>);
static_assert(std::is_assignable_v<VoidAssign&, const VoidAssign&>);
static_assert(!dpl_internal::__assignable_from<NoCopyAssign, const NoCopyAssign&>);
static_assert(!dpl_internal::__assignable_from<MoveOnly, const MoveOnly&>);

static_assert(dpl_internal::__move_constructible<Regular>);
static_assert(dpl_internal::__move_constructible<MoveOnly>);
static_assert(!dpl_internal::__move_constructible<ThrowingDtor>);
static_assert(!dpl_internal::__move_constructible<NoDefaultCtor[2]>);

static_assert(dpl_internal::__copy_constructible<Regular>);
static_assert(dpl_internal::__copy_constructible<NoCopyAssign>);
static_assert(!dpl_internal::__copy_constructible<MoveOnly>);
static_assert(!dpl_internal::__copy_constructible<ThrowingDtor>);

static_assert(dpl_internal::__movable<Regular>);
static_assert(dpl_internal::__movable<MoveOnly>);
static_assert(!dpl_internal::__movable<VoidAssign>);
static_assert(!dpl_internal::__movable<int&>);

static_assert(dpl_internal::__copyable<Regular>);
static_assert(dpl_internal::__copyable<NoDefaultCtor>);
static_assert(!dpl_internal::__copyable<MoveOnly>);
static_assert(!dpl_internal::__copyable<NoCopyAssign>);
static_assert(!dpl_internal::__copyable<VoidAssign>);

// Each building block has to yield false for void instead of failing to compile, since forming void& is ill-formed
// rather than merely unsatisfied.
static_assert(!dpl_internal::__constructible_from<void>);
static_assert(!dpl_internal::__assignable_from<void, void>);
static_assert(!dpl_internal::__move_constructible<void>);
static_assert(!dpl_internal::__copy_constructible<void>);
static_assert(!dpl_internal::__movable<void>);
static_assert(!dpl_internal::__copyable<void>);
static_assert(!dpl_internal::__copy_constructible<const void>);
static_assert(!dpl_internal::__copyable<const void>);

#endif // !_ONEDPL_CPP20_CONCEPTS_PRESENT

//----------------------------------------------------------------------------//
// __is_value_storable_and_comparable_v
//----------------------------------------------------------------------------//

// An iterator whose reference type is not convertible to its value type.
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

// Accepted: the value type is storable and the comparator is callable on const values.
static_assert(dpl_internal::__is_value_storable_and_comparable_v<int*, std::less<int>>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<const int*, std::less<int>>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<std::vector<int>::iterator, std::less<int>>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<std::vector<int>::const_iterator, std::less<>>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<Regular*, std::less<Regular>>);
static_assert(
    dpl_internal::__is_value_storable_and_comparable_v<ExplicitDefaultCtor*, std::less<ExplicitDefaultCtor>>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<int*, MoveOnlyLess>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<int*, IntResultLess>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<int*, bool (*)(const int&, const int&)>);
// The comparator max_element passes down to the min_element brick.
static_assert(
    dpl_internal::__is_value_storable_and_comparable_v<int*, dpl_internal::__reorder_pred<std::less<int>>>);
// A proxy reference is fine as long as it converts to the value type.
static_assert(dpl_internal::__is_value_storable_and_comparable_v<std::vector<bool>::iterator, std::less<bool>>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<FakeIterator<int, int>, std::less<int>>);
static_assert(dpl_internal::__is_value_storable_and_comparable_v<FakeIterator<std::pair<int, int>, std::pair<int&, int&>>,
                                                                std::less<std::pair<int, int>>>);

// Rejected because of the value type.
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<NoDefaultCtor*, std::less<NoDefaultCtor>>);
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<NoCopyAssign*, std::less<NoCopyAssign>>);
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<VoidAssign*, std::less<VoidAssign>>);
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<ThrowingDtor*, std::less<ThrowingDtor>>);
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<MoveOnly*, std::less<MoveOnly>>);

// Rejected because the reference type does not convert to the value type.
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<FakeIterator<int, OpaqueRef>, std::less<int>>);

// Rejected because an output iterator reports void as its value type.
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<std::back_insert_iterator<std::vector<int>>,
                                                                 std::less<int>>);

// Rejected because of the comparator.
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<int*, NotBoolResultLess>);
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<int*, MutableRefLess>);
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<int*, RvalueOnlyLess>);
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<int*, UnaryLess>);
static_assert(!dpl_internal::__is_value_storable_and_comparable_v<int*, std::less<Regular>>);

int
main()
{
    return TestUtils::done();
}
