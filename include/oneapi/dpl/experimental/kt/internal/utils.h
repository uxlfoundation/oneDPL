#ifndef _ONEDPL_KT_GROUP_UTILS_H
#define _ONEDPL_KT_GROUP_UTILS_H

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{
struct item_array_order
{
    // Array elements between a work-item have a stride of the sub-group size. Within the
    // sub-group, adjacent elements for the operation are held between adjacent work items
    // for each index i in the array.
    // E.g. If the desired, operation is over 0,1,2,3,4,...,11 with a sub-group size of 4,
    // Item 0 holds 0, 4, 8
    // Item 1 holds 1, 5, 9
    // Item 2 holds 2, 6, 10
    // Item 3 holds 3, 7, 11
    struct sub_group_stride{};

    // TODO: future types
    //struct work_group_stride{};
    //struct unit_stride{};
};

}
}
}

#endif
