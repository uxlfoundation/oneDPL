// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DEVICE_VECTOR_IMPL_H
#define _ONEDPL_DEVICE_VECTOR_IMPL_H

#include "device_array_impl.h"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include "oneapi/dpl/pstl/hetero/dpcpp/sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{
namespace compat
{

template <typename T>
class device_pointer;

template <typename T>
class device_reference;

template <typename>
struct __is_device_pointer : std::false_type
{
};

template <typename T>
struct __is_device_pointer<device_pointer<T>> : std::true_type
{
};

// =========================================================================
// device_pointer<T>
// =========================================================================

template <typename T>
class device_pointer
{
    T* __ptr = nullptr;
    const sycl::context* __ctx = nullptr;

  public:
    using iterator_concept = std::random_access_iterator_tag;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::remove_cv_t<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = device_pointer;
    using reference = device_reference<T>;

    device_pointer() = default;
    explicit device_pointer(T* __p, const sycl::context* __c = nullptr) : __ptr(__p), __ctx(__c) {}

    T*
    get() const
    {
        return __ptr;
    }

    const sycl::context*
    get_context_ptr() const
    {
        return __ctx;
    }

    reference
    operator*() const
    {
        return reference(__ptr, __ctx);
    }

    reference
    operator[](difference_type __n) const
    {
        return reference(__ptr + __n, __ctx);
    }

    device_pointer&
    operator++()
    {
        ++__ptr;
        return *this;
    }
    device_pointer
    operator++(int)
    {
        auto __tmp = *this;
        ++__ptr;
        return __tmp;
    }
    device_pointer&
    operator--()
    {
        --__ptr;
        return *this;
    }
    device_pointer
    operator--(int)
    {
        auto __tmp = *this;
        --__ptr;
        return __tmp;
    }

    device_pointer&
    operator+=(difference_type __n)
    {
        __ptr += __n;
        return *this;
    }
    device_pointer&
    operator-=(difference_type __n)
    {
        __ptr -= __n;
        return *this;
    }

    friend device_pointer
    operator+(device_pointer __p, difference_type __n)
    {
        return device_pointer(__p.__ptr + __n, __p.__ctx);
    }
    friend device_pointer
    operator+(difference_type __n, device_pointer __p)
    {
        return device_pointer(__p.__ptr + __n, __p.__ctx);
    }
    friend device_pointer
    operator-(device_pointer __p, difference_type __n)
    {
        return device_pointer(__p.__ptr - __n, __p.__ctx);
    }
    friend difference_type
    operator-(device_pointer __a, device_pointer __b)
    {
        return __a.__ptr - __b.__ptr;
    }

    friend bool
    operator==(device_pointer __a, device_pointer __b)
    {
        return __a.__ptr == __b.__ptr;
    }
    friend bool
    operator!=(device_pointer __a, device_pointer __b)
    {
        return __a.__ptr != __b.__ptr;
    }
    friend bool
    operator<(device_pointer __a, device_pointer __b)
    {
        return __a.__ptr < __b.__ptr;
    }
    friend bool
    operator>(device_pointer __a, device_pointer __b)
    {
        return __a.__ptr > __b.__ptr;
    }
    friend bool
    operator<=(device_pointer __a, device_pointer __b)
    {
        return __a.__ptr <= __b.__ptr;
    }
    friend bool
    operator>=(device_pointer __a, device_pointer __b)
    {
        return __a.__ptr >= __b.__ptr;
    }
};

// =========================================================================
// device_reference<T>
// =========================================================================

template <typename T>
class device_reference
{
    T* __ptr;
    const sycl::context* __ctx;

    T
    __read() const
    {
        T __val;
        sycl::device __dev = sycl::get_pointer_device(__ptr, *__ctx);
        sycl::queue __q{*__ctx, __dev};
        __q.memcpy(&__val, __ptr, sizeof(T)).wait();
        return __val;
    }

    void
    __write(const T& __val) const
    {
        sycl::device __dev = sycl::get_pointer_device(__ptr, *__ctx);
        sycl::queue __q{*__ctx, __dev};
        __q.memcpy(__ptr, &__val, sizeof(T)).wait();
    }

  public:
    device_reference(T* __p, const sycl::context* __c) : __ptr(__p), __ctx(__c) {}

    operator T() const { return __read(); }

    const device_reference&
    operator=(const T& __val) const
    {
        __write(__val);
        return *this;
    }

    const device_reference&
    operator=(const device_reference& __other) const
    {
        __write(static_cast<T>(__other));
        return *this;
    }

    const device_reference&
    operator+=(const T& __val) const
    {
        __write(__read() + __val);
        return *this;
    }
    const device_reference&
    operator-=(const T& __val) const
    {
        __write(__read() - __val);
        return *this;
    }
    const device_reference&
    operator*=(const T& __val) const
    {
        __write(__read() * __val);
        return *this;
    }
    const device_reference&
    operator/=(const T& __val) const
    {
        __write(__read() / __val);
        return *this;
    }
    const device_reference&
    operator%=(const T& __val) const
    {
        __write(__read() % __val);
        return *this;
    }
    const device_reference&
    operator&=(const T& __val) const
    {
        __write(__read() & __val);
        return *this;
    }
    const device_reference&
    operator|=(const T& __val) const
    {
        __write(__read() | __val);
        return *this;
    }
    const device_reference&
    operator^=(const T& __val) const
    {
        __write(__read() ^ __val);
        return *this;
    }
    const device_reference&
    operator<<=(const T& __val) const
    {
        __write(__read() << __val);
        return *this;
    }
    const device_reference&
    operator>>=(const T& __val) const
    {
        __write(__read() >> __val);
        return *this;
    }

    const device_reference&
    operator++() const
    {
        __write(__read() + T(1));
        return *this;
    }
    T
    operator++(int) const
    {
        T __old = __read();
        __write(__old + T(1));
        return __old;
    }
    const device_reference&
    operator--() const
    {
        __write(__read() - T(1));
        return *this;
    }
    T
    operator--(int) const
    {
        T __old = __read();
        __write(__old - T(1));
        return __old;
    }

    device_pointer<T>
    operator&() const
    {
        return device_pointer<T>(__ptr, __ctx);
    }

    friend void
    swap(const device_reference& __a, const device_reference& __b)
    {
        T __tmp = static_cast<T>(__a);
        __a = static_cast<T>(__b);
        __b = __tmp;
    }
};

// =========================================================================
// Pointer cast utilities
// =========================================================================

template <typename T>
T*
raw_pointer_cast(device_pointer<T> __ptr)
{
    return __ptr.get();
}

template <typename T>
device_pointer<T>
device_pointer_cast(T* __ptr)
{
    return device_pointer<T>(__ptr);
}

// =========================================================================
// device_vector<T, Alloc>
// =========================================================================

template <typename T, typename Alloc = device_allocator<T>>
class device_vector
{
    device_array<T, Alloc> __impl;

    const sycl::context*
    __ctx_ptr() const
    {
        return __impl.context_ptr();
    }

    sycl::queue
    __make_queue() const
    {
        return sycl::queue{__impl.get_context(), __impl.get_device()};
    }

  public:
    using value_type = T;
    using allocator_type = Alloc;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = device_reference<T>;
    using const_reference = device_reference<const T>;
    using pointer = device_pointer<T>;
    using const_pointer = device_pointer<const T>;
    using iterator = device_pointer<T>;
    using const_iterator = device_pointer<const T>;

    // --- Construction from queue ---
    explicit device_vector(sycl::queue __q) : __impl(static_cast<size_type>(0), __q) {}

    device_vector(size_type __count, sycl::queue __q) : __impl(__count, T{}, __q) {}

    device_vector(size_type __count, const T& __value, sycl::queue __q) : __impl(__count, __value, __q) {}

    device_vector(size_type __count, no_init_t, sycl::queue __q) : __impl(__count, __q) {}

    template <typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt> &&
                                                            !__is_device_pointer<std::decay_t<InputIt>>::value>>
    device_vector(InputIt __first, InputIt __last, sycl::queue __q) : __impl(__first, __last, __q)
    {
    }

    template <typename U>
    device_vector(device_pointer<U> __first, device_pointer<U> __last, sycl::queue __q)
        : __impl(static_cast<size_type>(__last - __first), __q)
    {
        if (__first != __last)
        {
            __q.memcpy(__impl.data(), __first.get(), static_cast<size_type>(__last - __first) * sizeof(T)).wait();
        }
    }

    device_vector(std::initializer_list<T> __init, sycl::queue __q) : __impl(__init, __q) {}

    explicit device_vector(const std::vector<T>& __src, sycl::queue __q) : __impl(__src, __q) {}

    // --- Construction from context + device ---
    explicit device_vector(sycl::context __c, sycl::device __d) : __impl(static_cast<size_type>(0), __c, __d) {}

    device_vector(size_type __count, sycl::context __c, sycl::device __d) : __impl(__count, T{}, __c, __d) {}

    device_vector(size_type __count, const T& __value, sycl::context __c, sycl::device __d)
        : __impl(__count, __value, __c, __d)
    {
    }

    device_vector(size_type __count, no_init_t, sycl::context __c, sycl::device __d) : __impl(__count, __c, __d) {}

    template <typename InputIt, typename = std::enable_if_t<!std::is_integral_v<InputIt> &&
                                                            !__is_device_pointer<std::decay_t<InputIt>>::value>>
    device_vector(InputIt __first, InputIt __last, sycl::context __c, sycl::device __d)
        : __impl(__first, __last, __c, __d)
    {
    }

    template <typename U>
    device_vector(device_pointer<U> __first, device_pointer<U> __last, sycl::context __c, sycl::device __d)
        : __impl(static_cast<size_type>(__last - __first), __c, __d)
    {
        if (__first != __last)
        {
            sycl::queue __q{__c, __d};
            __q.memcpy(__impl.data(), __first.get(), static_cast<size_type>(__last - __first) * sizeof(T)).wait();
        }
    }

    device_vector(std::initializer_list<T> __init, sycl::context __c, sycl::device __d) : __impl(__init, __c, __d) {}

    explicit device_vector(const std::vector<T>& __src, sycl::context __c, sycl::device __d) : __impl(__src, __c, __d)
    {
    }

    // Allocator-aware construction
    device_vector(size_type __count, sycl::queue __q, const Alloc& __a) : __impl(__count, T{}, __q, __a) {}

    device_vector(size_type __count, sycl::context __c, sycl::device __d, const Alloc& __a)
        : __impl(__count, T{}, __c, __d, __a)
    {
    }

    // Copy / move
    device_vector(const device_vector&) = default;
    device_vector(device_vector&&) noexcept = default;
    device_vector&
    operator=(const device_vector&) = default;
    device_vector&
    operator=(device_vector&&) noexcept = default;

    ~device_vector() = default;

    // Assign from host vector
    device_vector&
    operator=(const std::vector<T>& __src)
    {
        sycl::queue __q = __make_queue();
        __impl.assign(__src, __q);
        return *this;
    }

    // Convert to host vector
    explicit operator std::vector<T>() const
    {
        sycl::queue __q = __make_queue();
        return __impl.to_vector(__q);
    }

    // --- Element access (proxy references) ---
    reference
    operator[](size_type __pos)
    {
        return reference(__impl.data() + __pos, __ctx_ptr());
    }
    const_reference
    operator[](size_type __pos) const
    {
        return const_reference(__impl.data() + __pos, __ctx_ptr());
    }

    reference
    front()
    {
        return (*this)[0];
    }
    const_reference
    front() const
    {
        return (*this)[0];
    }
    reference
    back()
    {
        return (*this)[__impl.size() - 1];
    }
    const_reference
    back() const
    {
        return (*this)[__impl.size() - 1];
    }

    // --- Pointer access ---
    pointer
    data()
    {
        return pointer(__impl.data(), __ctx_ptr());
    }
    const_pointer
    data() const
    {
        return const_pointer(__impl.data(), __ctx_ptr());
    }

    // --- Iterators ---
    iterator
    begin()
    {
        return pointer(__impl.data(), __ctx_ptr());
    }
    const_iterator
    begin() const
    {
        return const_pointer(__impl.data(), __ctx_ptr());
    }
    iterator
    end()
    {
        return pointer(__impl.data() + __impl.size(), __ctx_ptr());
    }
    const_iterator
    end() const
    {
        return const_pointer(__impl.data() + __impl.size(), __ctx_ptr());
    }

    // --- Capacity ---
    size_type
    size() const
    {
        return __impl.size();
    }
    size_type
    capacity() const
    {
        return __impl.capacity();
    }
    bool
    empty() const
    {
        return __impl.empty();
    }

    void
    resize(size_type __count)
    {
        __impl.resize(__count, T{});
    }
    void
    resize(size_type __count, const T& __value)
    {
        __impl.resize(__count, __value);
    }
    void
    resize(size_type __count, no_init_t)
    {
        __impl.resize(__count);
    }
    void
    reserve(size_type __new_cap)
    {
        __impl.reserve(__new_cap);
    }
    void
    reserve(size_type __new_cap, sycl::queue __q)
    {
        __impl.reserve(__new_cap, __q);
    }
    void
    clear()
    {
        __impl.clear();
    }

    // --- Swap ---
    void
    swap(device_vector& __other)
    {
        std::swap(__impl, __other.__impl);
    }

    // --- Access to underlying device_array ---
    device_array<T, Alloc>&
    base()
    {
        return __impl;
    }
    const device_array<T, Alloc>&
    base() const
    {
        return __impl;
    }

    // --- Allocator ---
    allocator_type
    get_allocator() const
    {
        return __impl.get_allocator();
    }

    // --- Context / device ---
    sycl::context
    get_context() const
    {
        return __impl.get_context();
    }
    sycl::device
    get_device() const
    {
        return __impl.get_device();
    }
};

} // namespace compat
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_DEVICE_VECTOR_IMPL_H
