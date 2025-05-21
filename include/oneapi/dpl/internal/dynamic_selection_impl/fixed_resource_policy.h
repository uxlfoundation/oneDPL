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

#if _DS_BACKEND_SYCL != 0
#    include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"
#endif

namespace oneapi {
namespace dpl {
namespace experimental {

#if _DS_BACKEND_SYCL != 0
template <typename ResourceType = sycl::queue, typename Backend = default_backend<sycl::queue>>
#else
template <typename ResourceType, typename Backend>
#endif
class fixed_resource_policy : public policy_base<fixed_resource_policy<ResourceType, Backend>, ResourceType, Backend> {
    using base_t = policy_base<fixed_resource_policy<ResourceType, Backend>, ResourceType, Backend>;
    using resource_container_size_t = typename base_t::resource_container_size_t;
    using resource_type = typename base_t::resource_type;

    struct selector_t {
        typename base_t::resource_container_t resources_;
        resource_container_size_t index_ = 0;
    };

    std::shared_ptr<selector_t> selector_;
    std::size_t initial_index_ = 0;

public:
    using base_t::base_t;
    using typename base_t::selection_type;

    fixed_resource_policy(std::size_t index = 0) : initial_index_(index) { this->initialize(); }
    fixed_resource_policy(deferred_initialization_t) : initial_index_(0) {}
    fixed_resource_policy(const std::vector<resource_type>& u, std::size_t index = 0)
        : initial_index_(index) {
        this->initialize(u);
    }

    void ensure_selector_initialized() {
        if (!selector_) selector_ = std::make_shared<selector_t>();
    }

    void initialize_impl(const std::vector<resource_type>& u) {
        auto* s = this->template selector<selector_t>(selector_);
        s->resources_ = u;
        s->index_ = initial_index_;
    }

    template <typename... Args>
    selection_type select_impl(Args&&...) {
        auto* s = this->template selector<selector_t>(selector_);
        if (!s->resources_.empty()) {
            return selection_type{*this, s->resources_[s->index_]};
        }
        return selection_type{*this};  // default constructed if empty
    }
};

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_FIXED_RESOURCE_POLICY_H

