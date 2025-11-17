// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_FIXED_RESOURCE_POLICY_H
#define _ONEDPL_FIXED_RESOURCE_POLICY_H

#include "oneapi/dpl/internal/dynamic_selection_impl/policy_base.h"
#include "oneapi/dpl/functional"
#include "oneapi/dpl/internal/dynamic_selection_impl/default_backend.h"

#if _DS_BACKEND_SYCL
#    include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace experimental
{

#if _DS_BACKEND_SYCL
template <typename ResourceType = sycl::queue, typename ResourceAdapter = oneapi::dpl::identity,
          typename Backend = default_backend<ResourceType, ResourceAdapter>>
#else
template <typename ResourceType, typename ResourceAdapter = oneapi::dpl::identity,
          typename Backend = default_backend<ResourceType, ResourceAdapter>>
#endif
class fixed_resource_policy
    : public policy_base<fixed_resource_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend>
{
  protected:
    using base_t = policy_base<fixed_resource_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend>;
    friend base_t;
    using resource_container_size_t = typename base_t::resource_container_size_t;
    using selection_type = typename base_t::selection_type;

    struct selector_t
    {
        typename base_t::resource_container_t resources_;
        ::std::size_t index_ = 0;
    };

    std::shared_ptr<selector_t> selector_;

    void
    initialize_impl(std::size_t index = 0)
    {
        if (!selector_)
        {
            selector_ = std::make_shared<selector_t>();
        }
        selector_->index_ = index;
        auto u = base_t::get_resources();
        selector_->resources_ = u;
    }

    template <typename... Args>
    std::shared_ptr<selection_type>
    try_select_impl(Args&&...)
    {
        if (selector_)
        {
            return std::make_shared<selection_type>(*this, selector_->resources_[selector_->index_]);
        }
        else
        {
            throw std::logic_error("select called before initialization");
        }
    }

  public:
    using resource_type = typename base_t::resource_type;
    using typename base_t::backend_t;

    fixed_resource_policy(std::size_t index = 0) { base_t::initialize(index); }
    fixed_resource_policy(deferred_initialization_t) {}

    fixed_resource_policy(const std::vector<ResourceType>& u, std::size_t index = 0)
    {
        base_t::initialize(u, oneapi::dpl::identity(), index);
    }

    fixed_resource_policy(const std::vector<ResourceType>& u, ResourceAdapter adapter, std::size_t index = 0)
    {
        base_t::initialize(u, adapter, index);
    }
};

//CTAD deduction guides for initializer_list

//supports fixed_resource_policy p{ {t1, t2} }
template <typename T>
fixed_resource_policy(std::initializer_list<T>)
    -> fixed_resource_policy<T, oneapi::dpl::identity,
                             oneapi::dpl::experimental::default_backend<T, oneapi::dpl::identity>>;

//supports fixed_resource_policy p{ {t1, t2}, offset }
template <typename T, typename I,
          typename = std::enable_if_t<std::is_convertible_v<I, std::size_t>>>
fixed_resource_policy(std::initializer_list<T>, I)
    -> fixed_resource_policy<T, oneapi::dpl::identity, 
                             oneapi::dpl::experimental::default_backend<T, oneapi::dpl::identity>>;

//supports fixed_resource_policy p{ {t1, t2}, adapter }
//assumes Adapter is not convertible to size_t (to prevent ambiguity)
template <typename T, typename Adapter,
          typename = std::enable_if_t<std::negation_v<std::is_convertible<Adapter, std::size_t>>>>
fixed_resource_policy(std::initializer_list<T>, Adapter)
    -> fixed_resource_policy<T, Adapter, oneapi::dpl::experimental::default_backend<T, Adapter>>;

//supports fixed_resource_policy p{ {t1, t2}, adapter, offset }
template <typename T, typename Adapter, typename I, typename = std::enable_if_t<std::is_convertible_v<I, std::size_t>>>
fixed_resource_policy(std::initializer_list<T>, Adapter, I)
    -> fixed_resource_policy<T, Adapter, oneapi::dpl::experimental::default_backend<T, Adapter>>;

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_FIXED_RESOURCE_POLICY_H
