# Bounded Output Support for Heterogeneous Range-Based Set Algorithms

This document describes the mechanisms introduced/affected by the PR that adds proper **bounded
output** support to the `oneapi::dpl::ranges::set_*` algorithms (`set_union`, `set_intersection`,
`set_difference`, `set_symmetric_difference`) when executed with **hetero (SYCL) policies**.

The core problem: a range-based set algorithm must stop writing once the **output range** is full and
return the **stop positions** in both source ranges (i.e. how many input elements were actually
consumed). On the device this requires detecting an out-of-bounds (OOB) write, recovering the
corresponding source position, and reconciling it with the natural final position of the operation.

---

## 1. High-level call chain

How a public range-based set algorithm reaches the reduce-then-scan kernels.

```mermaid
flowchart TD
    A["oneapi::dpl::ranges::set_union / set_intersection /<br/>set_difference / set_symmetric_difference"] --> B["__pattern_set_* (algorithm_ranges_impl_hetero.h)"]
    B -->|"__begin_and_size(r1), __begin_and_size(r2)"| C{"empty-range<br/>fast path?"}
    C -->|"yes"| D["return consumed iterators<br/>(first1 + idx, first2 + idx)"]
    C -->|"no"| E["__parallel_set_op&lt;/*_Bounded*/ true&gt;<br/>(parallel_backend_sycl.h)"]
    E --> F["__set_op_impl / __parallel_set_scan"]
    F --> G["__parallel_transform_reduce_then_scan&lt;_Bounded&gt;"]
    G --> H["Reduce kernel<br/>(step 1)"]
    G --> I["Scan kernel<br/>(step 2: writes + OOB detection)"]
    I --> J["__set_op_impl_return_t&lt;_Bounded&gt;<br/>= (stop1, stop2, out_size)"]
    J --> B
    B --> K["build std::ranges::set_*_result&lt;...&gt;<br/>with in1/in2 stop iterators + out end"]
```

**Key idea:** the `_Bounded` non-type template parameter is threaded from the public pattern all the
way down into the kernel submitters. When `_Bounded == false` the whole stop-position machinery
compiles away to a single `std::size_t` result (no overhead for the unbounded code paths).

---

## 2. Bounded write-path decision (per sub-group)

`__scan_through_elements_helper` decides at runtime whether a sub-group may write freely (fast path)
or must go through the bounded write op that checks every write against the output size.

```mermaid
flowchart TD
    A["__scan_through_elements_helper&lt;_Bounded, ...&gt;"] --> B{"__capture_output?"}
    B -->|"no"| Z["noop write op (reduce phase)"]
    B -->|"yes"| C{"_Bounded?"}
    C -->|"no"| U["unbounded write op"]
    C -->|"yes"| D["out_size = size(out_rng)<br/>carry_in = sub_group_carry<br/>max_writes = iters_per_item * sg_size * max_outputs_per_input<br/>write_offset = is_unique ? 1 : 0"]
    D --> E{"carry_in + max_writes<br/>+ write_offset &gt; out_size ?"}
    E -->|"no (cannot overflow)"| U
    E -->|"yes (may overflow)"| F["bounded write op:<br/>each write checks id vs out_size,<br/>fires __on_oob_reached on first overflow"]
```

> **Underflow note.** The guard is written as `carry_in + max_writes + write_offset > out_size`
> rather than `carry_in + max_writes > out_size - write_offset`. With unsigned arithmetic the latter
> underflows (wraps to a huge value) when `out_size < write_offset` -- e.g. `out_size == 0`, or
> `out_size == 1` for unique patterns -- and would incorrectly select the unbounded (fast) path.

---

## 3. Two-step OOB detection and source-position recovery

Set operations run on a *balanced-path* partition: a sub-group consumes a diagonal of the merge
matrix, so the output index where the range fills up does **not** map directly to a source position.
Recovering it with sub-group collectives would be expensive, so we recover it in a cheap **second
serial pass** that replays only the single offending diagonal.

```mermaid
sequenceDiagram
    participant SG as Scan kernel (sub-group)
    participant WB as Bounded write op
    participant CB as __on_oob_reached callback
    participant FZ as __finalize_oob_detected
    participant GS as __gen_scan_input (replay)
    participant PC as __src_pos_capturing_temp_data
    participant ST as Stop-pos storage

    SG->>WB: write element at output id
    WB->>WB: first overflow when id reaches out_size
    WB->>CB: __on_oob_reached(diagonal-local output offset)
    CB->>CB: save start_id_reached_on_oob + oob_offset
    Note over SG,CB: collectives finished, recovery is serial
    SG->>FZ: __finalize_oob_detected(oob_offset, in_rng, start_id_on_oob)
    alt two-step detection (set ops)
        FZ->>PC: construct catcher(oob_offset)
        FZ->>GS: replay __gen_scan_input(in_rng, start_id_on_oob, catcher)
        GS->>PC: set(idx, value, src_idx) for each produced output
        PC-->>FZ: saved src_pos at idx == oob_offset
    else direct detection
        FZ-->>FZ: oob_offset already is the source position
    end
    FZ->>ST: __update_oob_pos(src_pos)
```

`__detect_oob_in_two_steps_v<_GenScanInput>` selects between the two branches at compile time:
`__gen_set_op_from_known_balanced_path` enables the replay path; all other generators map the OOB
offset to the source position directly.

---

## 4. Stop-position storage and host-side finalization

Two independent positions are tracked per source range and combined on the host.

```mermaid
flowchart LR
    subgraph Device["Device (scan kernel)"]
        A["normal writes"]
        B["OOB write"]
    end

    subgraph Storage["_SetOpFinalAndOOBPosType"]
        F["final_pos = (f1, f2)"]
        O["oob_pos = (o1, o2)"]
    end

    A -->|"__create_final_pos_saver to __update_final_pos (atomic fetch_max)"| F
    B -->|"__finalize_oob_detected to __update_oob_pos"| O
    Storage -->|"copy back"| H["Host"]
    H -->|"__compute_stop_pos() = (min(f1,o1), min(f2,o2))"| S["stop_pos (stop1, stop2)"]
    S --> R["std::ranges::set_*_result"]
```

| Field        | Initial value          | How it is updated                                   | Meaning                                            |
|--------------|------------------------|-----------------------------------------------------|----------------------------------------------------|
| `final_pos`  | `{0, 0}`               | device, atomic `fetch_max` per work-group           | furthest source position the operation consumed    |
| `oob_pos`    | `{size1, size2}`       | device, single writer on first overflow             | source position where the output range filled up   |
| stop (host)  | --                     | `min(final_pos, oob_pos)` element-wise              | actual returned stop position in each source range |

The `min` reconciliation is what makes the result correct in both regimes:

- **Output large enough:** no OOB occurs, `oob_pos` stays at `{size1, size2}`, so the stop position is
  the natural `final_pos`.
- **Output too small:** `oob_pos` records where writing stopped; it is `<= final_pos`, so it wins the
  `min` and becomes the reported stop position.

---

## 5. Supporting utilities introduced by the PR

| Symbol | File | Role |
|---|---|---|
| `__no_callback_tag`, `__is_no_callback_v` | `utils.h` | No-op placeholder so kernel helpers can branch on absent callbacks via `if constexpr`. |
| `__begin_and_size` | `utils_ranges.h` | Returns `(begin, size)` together for concise pattern code. |
| `_SetOpFinalAndOOBPosType`, `__compute_stop_pos` | `utils_ranges_sycl.h` | Device-copyable storage holding `final_pos` + `oob_pos` and computing their `min`. |
| `__create_initial_final_and_oob_pos_state<_Bounded>` | `utils_ranges_sycl.h` | Initializes storage (`{0,0}` / `{size1,size2}`); collapses to `size_t{0}` when unbounded. |
| `__create_transform_result_op<_Bounded>` / `__clamp_max` | `utils_ranges_sycl.h` | Clamps the reported output size to the output range capacity. |
| `__src_pos_capturing_temp_data` | `parallel_backend_sycl_reduce_then_scan.h` | Temp-data stand-in that captures the source position at a target diagonal-local offset. |
| `__finalize_oob_detected`, `__create_on_oob_reached`, `__create_final_pos_saver` | `parallel_backend_sycl_reduce_then_scan.h` | OOB recovery, OOB callback, and final-position saver. |
| `__device_storage::type`, `__result_storage::type` | `parallel_backend_sycl_utils.h` | Expose storage element type for the stop-position traits. |

---

## 6. Component interaction map

End-to-end view of how the host driver, the two device kernels, the shared stop-position storage and
the supporting utilities interact during one bounded set operation.

```mermaid
flowchart TB
    subgraph Host["Host driver (parallel_backend_sycl.h)"]
        H1["__pattern_set_* builds ranges"]
        H2["__create_initial_final_and_oob_pos_state&lt;_Bounded&gt;"]
        H3["__create_transform_result_op&lt;_Bounded&gt; (clamp out_size)"]
        H4["read storage and call __compute_stop_pos()"]
        H5["return std::ranges::set_*_result"]
    end

    subgraph Shared["Shared device storage"]
        S1["_SetOpFinalAndOOBPosType<br/>final_pos + oob_pos"]
    end

    subgraph Reduce["Reduce kernel (step 1)"]
        R1["sub-group reductions to scratch"]
        R2["init storage to initial state<br/>(global item 0, block 0)"]
    end

    subgraph Scan["Scan kernel (step 2)"]
        K1["__scan_through_elements_helper&lt;_Bounded&gt;"]
        K2["bounded write op<br/>(per-element id vs out_size)"]
        K3["__create_final_pos_saver to<br/>__update_final_pos (atomic fetch_max)"]
        K4["__on_oob_reached then<br/>__finalize_oob_detected then __update_oob_pos"]
    end

    subgraph Util["Stateless utilities"]
        U1["__no_callback_tag"]
        U2["__src_pos_capturing_temp_data"]
        U3["__detect_oob_in_two_steps_v"]
    end

    H1 --> H2
    H2 -->|"initial state"| S1
    H1 -->|"submit"| Reduce
    R2 -.->|"write initial state"| S1
    Reduce -->|"scratch carries"| Scan
    H3 -->|"clamp op"| K1
    K1 --> K2
    K2 -->|"in-range write"| K3
    K2 -->|"first overflow"| K4
    K3 -->|"fetch_max"| S1
    K4 -->|"min-candidate"| S1
    U1 -.->|"no-op branch"| K1
    U2 -.->|"replay catcher"| K4
    U3 -.->|"compile-time path"| K4
    Scan -->|"completion event"| H4
    S1 -->|"copy back"| H4
    H4 --> H5
```

**How to read it.** Solid arrows are data/control flow during execution. Dashed arrows show
compile-time wiring or no-op placeholders that collapse away when `_Bounded == false`. The single
`_SetOpFinalAndOOBPosType` instance is the only shared mutable state between the kernels and the host:
the reduce kernel seeds it, the scan kernel updates `final_pos` (many writers, `fetch_max`) and
`oob_pos` (one writer), and the host reduces both into the returned stop positions.

---

*Rendered by GitHub: the Mermaid diagrams above display natively in the PR description, PR comments,
and any committed .md file.*
