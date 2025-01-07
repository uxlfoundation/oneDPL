# Convex hull

This example contains the oneDPL-based implementation of quickhull algorithm.
Quickhull algorithm description:
1. Initial phase
  1) Find two points that guaranteed to belong to the convex hull. Min and max points in `X` can be used for it.
  2) Divide the initial set of points in two subsets by the line formed by two points from previous step. This subset will be processed recursively.

2. Iteration Phase
  1) Divide current subset by dividing line `[p1,p2]` into right and left subsets.
  2) New point `p` of the convex hull is found as farthest point of right subset from the dividing line.
  3) If the right subset has more than 1 point, repeat the iteration phase with the right subset and dividing lines `[p1,p]` and `[p,p2]`.

The implementation based on `std::copy_if`, `std::max_element` and `std::minmax_element` algorithms of oneDPL.
Each of the algorithms use `oneapi::dpl::par_unseq` policy. In order to get effect of the policy usage problem size should be big enough.
By default problem size was set as 5 000 000 points. With point set with less than 500 000 points par_unseq policy could be inefficient.
Correctness of the convex hull is checked by `std::any_of` algorithm using `counting iterator`.

| Optimized for                   | Description                                                                                    |
|---------------------------------|------------------------------------------------------------------------------------------------|
| OS                              | Linux* Ubuntu* 18.04                                                                           |
| Hardware                        | Skylake or newer                                                                               |
| Software                        | Intel&reg; oneAPI DPC++ Library (oneDPL); Intel&reg; oneAPI Threading Building Blocks (oneTBB) |
| Time to complete                | At most 1 minute                                                                               |

## License

This code example is licensed under [Apache License Version 2.0 with LLVM exceptions](https://github.com/uxlfoundation/oneDPL/blob/main/LICENSE.txt). Refer to the "[LICENSE](licensing/LICENSE.txt)" file for the full license text and copyright notice.

## Building the 'Convex hull' Program

### On a Linux* System
Perform the following steps:

1. Source oneDPL and oneTBB

2. Build the program using the following `cmake` commands.
```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

3. Run the program:
```
    $ make run
```

4. Clean the program using:
```
    $ make clean
```

## Running the Program
### Example of Output

```
success
The convex hull has been stored to a file ConvexHull.csv
```
