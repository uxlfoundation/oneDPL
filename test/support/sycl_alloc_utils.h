// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _TEST_SYCL_ALLOC_UTILS_H
#define _TEST_SYCL_ALLOC_UTILS_H

#if TEST_DPCPP_BACKEND_PRESENT

#include <list>
#include <memory>

#include "utils_sycl_defs.h"

namespace TestUtils
{
////////////////////////////////////////////////////////////////////////////////
//
// RAII service class to allocate shared/device memory (USM)
// Usage model"
// 1. allocate USM memory and copying data to USM:
//    usm_data_transfer<alloc_type, _ValueType> dt_helper(queue, first, count);
// or
//    usm_data_transfer<alloc_type, _ValueType> dt_helper(queue, std::begin(data), std::end(data));
// or just allocate USM memory"
//    usm_data_transfer<alloc_type, _ValueType> dt_helper(queue, count);
// 2. get a USM pointer by usm_data_transfer::get_data() and passed one into a parallel algorithm with dpc++ policy.
// 3. Retrieve data back (in case of device allocation type) to the host for further checking result.
//    dt_helper.retrieve_data(dest_host);
//
// Count of elements may be zero. In this case no USM memory will be allocated.
// Also retrieve_data function will do nothing in this case.
// This behavior is implemented based on tests use cases.
//
template<sycl::usm::alloc _alloc_type, typename _ValueType>
class usm_data_transfer
{
    static_assert(_alloc_type == sycl::usm::alloc::shared || _alloc_type == sycl::usm::alloc::device,
                  "Invalid allocation type for usm_data_transfer class");

    template<sycl::usm::alloc __type>
    using __alloc_type = ::std::integral_constant<sycl::usm::alloc, __type>;
    using __shared_alloc_type = __alloc_type<sycl::usm::alloc::shared>;
    using __device_alloc_type = __alloc_type<sycl::usm::alloc::device>;

    template<typename _Size>
    _ValueType* allocate( _Size __sz, __shared_alloc_type)
    {
        return sycl::malloc_shared<_ValueType>(__sz, __queue);
    }
    template<typename _Size>
    _ValueType* allocate( _Size __sz, __device_alloc_type)
    {
        return sycl::malloc_device<_ValueType>(__sz, __queue);
    }

public:

    using __difference_type = typename ::std::iterator_traits<_ValueType*>::difference_type;

    template<typename _Size>
    usm_data_transfer(sycl::queue __q, _Size __sz)
        : __queue(__q), __count(__sz)
    {
        if (__count > 0)
        {
            std::cout<<"allocating usm data: "<<__count<<" elements of size, "<<sizeof(_ValueType)<<"\n";
            __ptr = allocate(__count, __alloc_type<_alloc_type>{});
            std::cout<<"allocated usm data: "<<__ptr<<"\n";
            assert(__ptr != nullptr);
        }
    }

    template<typename _Iterator, typename _Size>
    usm_data_transfer(sycl::queue __q, _Iterator __it, _Size __sz)
        : usm_data_transfer(__q, __sz)
    {
        if (__count > 0)
        {
            update_data(__it);
        }
    }

    template<typename _Iterator>
    usm_data_transfer(sycl::queue __q, _Iterator __itBegin, _Iterator __itEnd)
        : usm_data_transfer(__q, __itBegin, __itEnd - __itBegin)
    {
    }

    usm_data_transfer(usm_data_transfer&) = delete;
    usm_data_transfer(usm_data_transfer&& other)
    {
        ::std::swap(__queue, other.__queue);
        ::std::swap(__ptr,   other.__ptr);
        ::std::swap(__count, other.__count);
    }

    ~usm_data_transfer()
    {
        std::cout<<"destructor\n";
        reset();
    }

    usm_data_transfer& operator=(usm_data_transfer&) = delete;
    usm_data_transfer& operator=(usm_data_transfer&& other)
    {
        std::cout<<"move assignment operator\n";
        reset();

        ::std::swap(__queue, other.__queue);
        ::std::swap(__ptr,   other.__ptr);
        ::std::swap(__count, other.__count);

        return *this;
    }

    void reset()
    {
        // std::cout<<"reset usm data\n";
        // if (__count > 0)
        // {
        //     std::cout<<"freeing usm data: "<<__ptr<<"\n";
        //     assert(__ptr != nullptr);
        //     sycl::free(__ptr, __queue);

        //     __ptr = nullptr;
        //     __count = 0;
        // }
    }

    _ValueType* get_data() const
    {
        return __ptr;
    }

    template <typename _Iterator>
    void update_data(_Iterator __it)
    {
        update_data(__it, 0, __count);
    }

    template<typename _Iterator, typename _Size>
    void update_data(_Iterator __it, __difference_type __offset, _Size __objects_count)
    {
        assert(0 <= __offset);
        assert(0 <= __objects_count);
        assert(__offset + __objects_count <= __count);

        std::cout<<"update data usm_data_transfer "<<__objects_count<<", "<<__offset<<"\n";
        if (__count > 0 && __objects_count > 0)
        {
            assert(__ptr != nullptr);

            //TODO: support copying data provided by non-contiguous iterator
            auto __src = std::addressof(*__it);
            // we remove 1 from __objects_count to avoid of out-of-range iterator dereferencing
            assert(std::addressof(*(__it + __objects_count - 1)) - __src == __objects_count - 1);

            copy_data_impl(__src, __ptr + __offset, __objects_count);
        }
    }

    template <typename _Iterator>
    void retrieve_data(_Iterator __it)
    {
        retrieve_data(__it, 0, __count);
    }

    template<typename _Iterator>
    void retrieve_data(_Iterator __it, __difference_type __offset, __difference_type __objects_count)
    {
        assert(0 <= __offset);
        assert(0 <= __objects_count);
        assert(__offset + __objects_count <= __count);

        std::cout<<"retrieve data usm_data_transfer"<<__objects_count<<", "<<__offset<<"\n";

        if (__count > 0 && __objects_count > 0)
        {
            assert(__ptr != nullptr);

            //TODO: support copying data provided by non-contiguous iterator
            auto __dst = std::addressof(*__it);
            // we remove 1 from __objects_count to avoid of out-of-range iterator dereferencing
            assert(std::addressof(*(__it + __objects_count - 1)) - __dst == __objects_count - 1);

            copy_data_impl(__ptr + __offset, __dst, __objects_count);
        }
    }

private:

    void copy_data_impl(_ValueType* __src, _ValueType* __ptr, __difference_type __count)
    {
        std::cout<<"copy "<<std::addressof(*__src)<<" -> "<<std::addressof(*__ptr)<<", count = "<<__count<<"\n";
#if TEST_LIBSYCL_VERSION >= 50300
        std::cout<<"queue copy\n";
        __queue.fill(__ptr, 0, __count);
#else
        std::cout<<"queue submit\n";
        auto __p = __ptr;
        auto __c = __count;
        __queue.submit([__src, __c, __p](sycl::handler& __cgh) {
            __cgh.parallel_for(sycl::range<1>(__c), [__src, __c, __p](sycl::item<1>__item) {
                ::std::size_t __id = __item.get_linear_id();
                *(__p + __id) = (__src + __id);
                });
            });
#endif // TEST_LIBSYCL_VERSION >= 50300
        __queue.wait();
    }

private:

    sycl::queue       __queue;
    __difference_type __count = 0;
    _ValueType*       __ptr = nullptr;
};

//--------------------------------------------------------------------------------------------------------------------//

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // _TEST_SYCL_ALLOC_UTILS_H
