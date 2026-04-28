// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DEVICE_ARRAY_IMPL_H
#define _ONEDPL_DEVICE_ARRAY_IMPL_H

#include <cstddef>
#include <cstring>
#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include "oneapi/dpl/pstl/hetero/dpcpp/sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{

// =========================================================================
// device_allocator<T>
// =========================================================================

template <typename T>
struct device_allocator
{
    T*
    allocate(std::size_t __n, sycl::context __ctx, sycl::device __dev)
    {
        return sycl::malloc_device<T>(__n, __dev, __ctx);
    }

    void
    deallocate(T* __p, std::size_t, sycl::context __ctx, sycl::device)
    {
        sycl::free(__p, __ctx);
    }
};

// =========================================================================
// device_span<T>
// =========================================================================

template <typename T>
class device_span
{
    T* __ptr = nullptr;
    std::size_t __size = 0;

  public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    using iterator = T*;

    device_span() = default;
    device_span(T* __p, std::size_t __sz) : __ptr(__p), __size(__sz) {}

    template <std::size_t N>
    device_span(T (&__arr)[N]) : __ptr(__arr), __size(N)
    {
    }

    // Converting constructor: device_span<T> -> device_span<const T>
    template <typename U, typename = std::enable_if_t<std::is_same_v<std::add_const_t<U>, T>>>
    device_span(device_span<U> __other) : __ptr(__other.data()), __size(__other.size())
    {
    }

    T*
    begin() const
    {
        return __ptr;
    }
    T*
    end() const
    {
        return __ptr + __size;
    }
    T*
    data() const
    {
        return __ptr;
    }
    std::size_t
    size() const
    {
        return __size;
    }
    bool
    empty() const
    {
        return __size == 0;
    }

    T&
    operator[](std::size_t __i) const
    {
        return __ptr[__i];
    }
    T&
    front() const
    {
        return __ptr[0];
    }
    T&
    back() const
    {
        return __ptr[__size - 1];
    }

    device_span
    first(std::size_t __count) const
    {
        return {__ptr, __count};
    }
    device_span
    last(std::size_t __count) const
    {
        return {__ptr + __size - __count, __count};
    }
    device_span
    subspan(std::size_t __offset, std::size_t __count) const
    {
        return {__ptr + __offset, __count};
    }
};

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#if __cplusplus >= 202002L
#    include <ranges>
template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<oneapi::dpl::experimental::device_span<T>> = true;

template <typename T>
inline constexpr bool std::ranges::enable_view<oneapi::dpl::experimental::device_span<T>> = true;
#endif

namespace oneapi
{
namespace dpl
{
namespace experimental
{

// =========================================================================
// no_init_t tag
// =========================================================================

struct no_init_t
{
    explicit no_init_t() = default;
};
inline constexpr no_init_t no_init{};

// =========================================================================
// device_array<T, Alloc>
// =========================================================================

template <typename T, typename Alloc = device_allocator<T>>
class device_array
{
  public:
    using value_type = T;
    using allocator_type = Alloc;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

  private:
    T* __data = nullptr;
    size_type __size = 0;
    size_type __capacity = 0;
    sycl::context __ctx;
    sycl::device __dev;
    Alloc __alloc;

    void
    __free_storage()
    {
        if (__data)
        {
            __alloc.deallocate(__data, __capacity, __ctx, __dev);
            __data = nullptr;
        }
    }

    void
    __deallocate()
    {
        __free_storage();
        __size = 0;
        __capacity = 0;
    }

    T*
    __allocate(size_type __n)
    {
        if (__n == 0)
            return nullptr;
        return __alloc.allocate(__n, __ctx, __dev);
    }

    sycl::queue
    __make_queue() const
    {
        return sycl::queue{__ctx, __dev};
    }

    void
    __memcpy_to_device(T* __dst, const T* __src, size_type __count, sycl::queue& __q)
    {
        if (__count > 0)
            __q.memcpy(__dst, __src, __count * sizeof(T)).wait();
    }

    void
    __memcpy_to_host(T* __dst, const T* __src, size_type __count, sycl::queue& __q) const
    {
        if (__count > 0)
            __q.memcpy(__dst, __src, __count * sizeof(T)).wait();
    }

    void
    __fill_on_device(T* __dst, size_type __count, const T& __value, sycl::queue& __q)
    {
        if (__count > 0)
            __q.fill(__dst, __value, __count).wait();
    }

    void
    __memcpy_device_to_device(T* __dst, const T* __src, size_type __count, sycl::queue& __q)
    {
        if (__count > 0)
            __q.memcpy(__dst, __src, __count * sizeof(T)).wait();
    }

  public:
    // =====================================================================
    // Construction
    // =====================================================================

    device_array(size_type __count, sycl::queue __q)
        : __size(__count), __capacity(__count), __ctx(__q.get_context()), __dev(__q.get_device())
    {
        __data = __allocate(__count);
    }

    device_array(size_type __count, sycl::context __c, sycl::device __d)
        : __size(__count), __capacity(__count), __ctx(__c), __dev(__d)
    {
        __data = __allocate(__count);
    }

    device_array(size_type __count, const T& __value, sycl::queue __q)
        : __size(__count), __capacity(__count), __ctx(__q.get_context()), __dev(__q.get_device())
    {
        __data = __allocate(__count);
        __fill_on_device(__data, __count, __value, __q);
    }

    device_array(size_type __count, const T& __value, sycl::queue __q, const Alloc& __a)
        : __size(__count), __capacity(__count), __ctx(__q.get_context()), __dev(__q.get_device()), __alloc(__a)
    {
        __data = __allocate(__count);
        __fill_on_device(__data, __count, __value, __q);
    }

    device_array(size_type __count, const T& __value, sycl::context __c, sycl::device __d)
        : __size(__count), __capacity(__count), __ctx(__c), __dev(__d)
    {
        __data = __allocate(__count);
        sycl::queue __q = __make_queue();
        __fill_on_device(__data, __count, __value, __q);
    }

    device_array(size_type __count, const T& __value, sycl::context __c, sycl::device __d, const Alloc& __a)
        : __size(__count), __capacity(__count), __ctx(__c), __dev(__d), __alloc(__a)
    {
        __data = __allocate(__count);
        sycl::queue __q = __make_queue();
        __fill_on_device(__data, __count, __value, __q);
    }

    template <typename InputIt, typename = std::enable_if_t<!std::is_same_v<std::decay_t<InputIt>, sycl::queue> &&
                                                            !std::is_integral_v<InputIt>>>
    device_array(InputIt __first, InputIt __last, sycl::queue __q) : __ctx(__q.get_context()), __dev(__q.get_device())
    {
        std::vector<T> __tmp(__first, __last);
        __size = __tmp.size();
        __capacity = __size;
        __data = __allocate(__size);
        __memcpy_to_device(__data, __tmp.data(), __size, __q);
    }

    device_array(std::initializer_list<T> __init, sycl::queue __q)
        : __size(__init.size()), __capacity(__init.size()), __ctx(__q.get_context()), __dev(__q.get_device())
    {
        __data = __allocate(__size);
        __memcpy_to_device(__data, __init.begin(), __size, __q);
    }

    device_array(const std::vector<T>& __src, sycl::queue __q)
        : __size(__src.size()), __capacity(__src.size()), __ctx(__q.get_context()), __dev(__q.get_device())
    {
        __data = __allocate(__size);
        __memcpy_to_device(__data, __src.data(), __size, __q);
    }

    template <typename InputIt, typename = std::enable_if_t<!std::is_same_v<std::decay_t<InputIt>, sycl::queue> &&
                                                            !std::is_integral_v<InputIt>>>
    device_array(InputIt __first, InputIt __last, sycl::context __c, sycl::device __d) : __ctx(__c), __dev(__d)
    {
        std::vector<T> __tmp(__first, __last);
        __size = __tmp.size();
        __capacity = __size;
        __data = __allocate(__size);
        sycl::queue __q = __make_queue();
        __memcpy_to_device(__data, __tmp.data(), __size, __q);
    }

    device_array(std::initializer_list<T> __init, sycl::context __c, sycl::device __d)
        : __size(__init.size()), __capacity(__init.size()), __ctx(__c), __dev(__d)
    {
        __data = __allocate(__size);
        sycl::queue __q = __make_queue();
        __memcpy_to_device(__data, __init.begin(), __size, __q);
    }

    device_array(const std::vector<T>& __src, sycl::context __c, sycl::device __d)
        : __size(__src.size()), __capacity(__src.size()), __ctx(__c), __dev(__d)
    {
        __data = __allocate(__size);
        sycl::queue __q = __make_queue();
        __memcpy_to_device(__data, __src.data(), __size, __q);
    }

    // Copy
    device_array(const device_array& __other)
        : __size(__other.__size), __capacity(__other.__size), __ctx(__other.__ctx), __dev(__other.__dev),
          __alloc(__other.__alloc)
    {
        __data = __allocate(__size);
        if (__size > 0)
        {
            sycl::queue __q = __make_queue();
            __memcpy_device_to_device(__data, __other.__data, __size, __q);
        }
    }

    // Move
    device_array(device_array&& __other) noexcept
        : __data(__other.__data), __size(__other.__size), __capacity(__other.__capacity), __ctx(__other.__ctx),
          __dev(__other.__dev), __alloc(std::move(__other.__alloc))
    {
        __other.__data = nullptr;
        __other.__size = 0;
        __other.__capacity = 0;
    }

    device_array&
    operator=(const device_array& __other)
    {
        if (this != &__other)
        {
            __deallocate();
            __ctx = __other.__ctx;
            __dev = __other.__dev;
            __alloc = __other.__alloc;
            __size = __other.__size;
            __capacity = __other.__size;
            __data = __allocate(__size);
            if (__size > 0)
            {
                sycl::queue __q = __make_queue();
                __memcpy_device_to_device(__data, __other.__data, __size, __q);
            }
        }
        return *this;
    }

    device_array&
    operator=(device_array&& __other) noexcept
    {
        if (this != &__other)
        {
            __deallocate();
            __data = __other.__data;
            __size = __other.__size;
            __capacity = __other.__capacity;
            __ctx = __other.__ctx;
            __dev = __other.__dev;
            __alloc = std::move(__other.__alloc);
            __other.__data = nullptr;
            __other.__size = 0;
            __other.__capacity = 0;
        }
        return *this;
    }

    ~device_array() { __deallocate(); }

    // =====================================================================
    // Host-device transfer
    // =====================================================================

    std::vector<T>
    to_vector() const
    {
        sycl::queue __q = __make_queue();
        return to_vector(__q);
    }

    std::vector<T>
    to_vector(sycl::queue __q) const
    {
        std::vector<T> __result(__size);
        __memcpy_to_host(__result.data(), __data, __size, __q);
        return __result;
    }

    void
    assign(const T* __first, const T* __last)
    {
        sycl::queue __q = __make_queue();
        assign(__first, __last, __q);
    }

    void
    assign(const T* __first, const T* __last, sycl::queue __q)
    {
        size_type __new_size = static_cast<size_type>(__last - __first);
        if (__new_size > __capacity)
        {
            __deallocate();
            __capacity = __new_size;
            __data = __allocate(__capacity);
        }
        __size = __new_size;
        __memcpy_to_device(__data, __first, __size, __q);
    }

    void
    assign(const std::vector<T>& __src)
    {
        assign(__src.data(), __src.data() + __src.size());
    }

    void
    assign(const std::vector<T>& __src, sycl::queue __q)
    {
        assign(__src.data(), __src.data() + __src.size(), __q);
    }

    T
    read(size_type __pos) const
    {
        sycl::queue __q = __make_queue();
        return read(__pos, __q);
    }

    T
    read(size_type __pos, sycl::queue __q) const
    {
        T __val;
        __q.memcpy(&__val, __data + __pos, sizeof(T)).wait();
        return __val;
    }

    void
    write(size_type __pos, const T& __value)
    {
        sycl::queue __q = __make_queue();
        write(__pos, __value, __q);
    }

    void
    write(size_type __pos, const T& __value, sycl::queue __q)
    {
        __q.memcpy(__data + __pos, &__value, sizeof(T)).wait();
    }

    sycl::event
    async_read(size_type __pos, T& __out, sycl::queue __q, const std::vector<sycl::event>& __depends_on = {}) const
    {
        return __q.memcpy(&__out, __data + __pos, sizeof(T), __depends_on);
    }

    sycl::event
    async_write(size_type __pos, const T& __value, sycl::queue __q, const std::vector<sycl::event>& __depends_on = {})
    {
        return __q.memcpy(__data + __pos, &__value, sizeof(T), __depends_on);
    }

    sycl::event
    async_to_vector(std::vector<T>& __out, sycl::queue __q, const std::vector<sycl::event>& __depends_on = {}) const
    {
        __out.resize(__size);
        if (__size == 0)
            return sycl::event{};
        return __q.memcpy(__out.data(), __data, __size * sizeof(T), __depends_on);
    }

    sycl::event
    async_assign(const T* __first, const T* __last, sycl::queue __q, const std::vector<sycl::event>& __depends_on = {})
    {
        size_type __new_size = static_cast<size_type>(__last - __first);
        if (__new_size > __capacity)
        {
            __deallocate();
            __capacity = __new_size;
            __data = __allocate(__capacity);
        }
        __size = __new_size;
        if (__size == 0)
            return sycl::event{};
        return __q.memcpy(__data, __first, __size * sizeof(T), __depends_on);
    }

    // =====================================================================
    // Device iteration — raw USM pointers
    // =====================================================================

    iterator
    begin()
    {
        return __data;
    }
    const_iterator
    begin() const
    {
        return __data;
    }
    iterator
    end()
    {
        return __data + __size;
    }
    const_iterator
    end() const
    {
        return __data + __size;
    }
    pointer
    data()
    {
        return __data;
    }
    const_pointer
    data() const
    {
        return __data;
    }

    // =====================================================================
    // Capacity
    // =====================================================================

    size_type
    size() const
    {
        return __size;
    }
    size_type
    capacity() const
    {
        return __capacity;
    }
    bool
    empty() const
    {
        return __size == 0;
    }

    void
    resize(size_type __count)
    {
        sycl::queue __q = __make_queue();
        resize(__count, __q);
    }

    void
    resize(size_type __count, sycl::queue __q)
    {
        if (__count <= __capacity)
        {
            __size = __count;
            return;
        }
        T* __new_data = __allocate(__count);
        if (__size > 0)
            __q.memcpy(__new_data, __data, __size * sizeof(T)).wait();
        __free_storage();
        __data = __new_data;
        __size = __count;
        __capacity = __count;
    }

    void
    resize(size_type __count, const T& __value)
    {
        sycl::queue __q = __make_queue();
        resize(__count, __value, __q);
    }

    void
    resize(size_type __count, const T& __value, sycl::queue __q)
    {
        size_type __old_size = __size;
        if (__count > __capacity)
        {
            T* __new_data = __allocate(__count);
            if (__old_size > 0)
                __q.memcpy(__new_data, __data, __old_size * sizeof(T)).wait();
            __free_storage();
            __data = __new_data;
            __capacity = __count;
        }
        __size = __count;
        if (__count > __old_size)
            __fill_on_device(__data + __old_size, __count - __old_size, __value, __q);
    }

    void
    reserve(size_type __new_cap)
    {
        if (__new_cap <= __capacity)
            return;
        sycl::queue __q = __make_queue();
        T* __new_data = __allocate(__new_cap);
        if (__size > 0)
            __q.memcpy(__new_data, __data, __size * sizeof(T)).wait();
        __free_storage();
        __data = __new_data;
        __capacity = __new_cap;
    }

    void
    clear()
    {
        __size = 0;
    }

    // =====================================================================
    // Views
    // =====================================================================

    device_span<T>
    span()
    {
        return device_span<T>(__data, __size);
    }

    device_span<const T>
    span() const
    {
        return device_span<const T>(__data, __size);
    }

    // =====================================================================
    // Allocator access
    // =====================================================================

    allocator_type
    get_allocator() const
    {
        return __alloc;
    }

    // =====================================================================
    // Context / device access
    // =====================================================================

    sycl::context
    get_context() const
    {
        return __ctx;
    }

    const sycl::context*
    context_ptr() const
    {
        return &__ctx;
    }

    sycl::device
    get_device() const
    {
        return __dev;
    }
};

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_DEVICE_ARRAY_IMPL_H
