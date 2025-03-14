// icpx -fsycl -fno-sycl-unnamed-lambda -fsycl-device-code-split=per_kernel
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <iostream>

int main()
{
    sycl::queue q;

    int n = 10;
    int *ptr = sycl::malloc_shared<int>(n, q);
    q.fill(ptr, 1, n).wait();

    oneapi::dpl::execution::device_policy<class kernel> policy{q};

    // OK
    auto res1 = oneapi::dpl::reduce(policy, ptr, ptr + n);
    // Error case (1)
    auto res2 = oneapi::dpl::reduce(std::move(policy), ptr, ptr + n);

    // Error case (2)
    const auto& policy_ref = policy;
    auto res3 = oneapi::dpl::reduce(policy_ref, ptr, ptr + n);

    std::cout << res1 << " " << res2 << res3 << std::endl;

    return 0;
}
