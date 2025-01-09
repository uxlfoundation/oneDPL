import argparse
import re


def parse_arguments():
    na_value = "N/A"
    parser = argparse.ArgumentParser(description="Analyze build logs and ctest output, and generate a summary.")
    parser.add_argument("--build-log", required=True, help="Path to the build log file.")
    parser.add_argument("--ctest-log", required=True, help="Path to the ctest log file.")
    parser.add_argument("--output-file", required=True, help="Path to the output file.")
    parser.add_argument("--os", default=na_value, help="Operating System.")
    parser.add_argument("--compiler-version", default=na_value, help="Version of the compiler.")
    parser.add_argument("--cmake-version", default=na_value, help="Version of CMake.")
    parser.add_argument("--cpu-model", default=na_value, help="CPU model.")
    return parser.parse_args()


def read_file(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        return f.read()


def generate_environment_table(os_info, compiler_version, cmake_version, cpu_model):
    table = []
    table.append( "| Environment Parameter | Value               |")
    table.append( "|-----------------------|---------------------|")
    table.append(f"| OS                    | {os_info}           |")
    table.append(f"| Compiler Ver.         | {compiler_version}  |")
    table.append(f"| CMake Ver.            | {cmake_version}     |")
    table.append(f"| CPU Model             | {cpu_model}         |")
    return "\n".join(table)


def generate_warning_table(build_log_content):
    # Match : Description
    warning_types = {
        r"-Wpass-failed=transform-warning": "Loop not vectorized (-Wpass-failed=transform-warning)",
        r"-Wrecommended-option": "Use of opiton <B> recommended over <A> (-Wrecommended-option)",
        r"-Wdeprecated-declarations": "-Wdeprecated-declarations",
        r"-Wsign-compare": "Comparison of integer expressions of different signedness (-Wsign-compare)",
        r"-Wunused-variable": "-Wunused-variable",
        r"-Wunused-but-set-variable": "-Wunused-but-set-variable",
        r"-Wunused-local-typedef": "-Wunused-local-typedef",
        r"-Wmacro-redefined": "-Wmacro-redefined",
        r"-Wmissing-field-initializers": "-Wmissing-field-initializers",
        r"-Wdeprecated-copy-with-user-provided-copy": "-Wdeprecated-copy-with-user-provided-copy",
        r"-Wunused-parameter": "-Wunused-parameter",
        r"C4244|C4267": "Conversion, possible data loss (C4244, C4267)",
        r"C4018": "Signed/unsigned mismatch (C4018)",
        r"STL4038": "Functionality from newer standards (STL4038)",
        r"STL4008": "Deprecated functionality (STL4008)",
        r"C4127": "Conditional expression is constant, use if constexpr instead (C4127)",
        r"C4100": "Unreferenced formal parameter (C4100)",
        r"C4189": "Local variable is initialized but not referenced (C4189)",
        "Other": "Other"
    }
    # MSVC warning format: ": warning"
    # GCC/Clang warning format: "warning:"
    warnings = re.findall(r": warning|warning:", build_log_content)

    warning_histogram = {warning: 0 for warning in warning_types.keys()}
    for warning in warning_types.keys():
        warning_histogram[warning] = len(re.findall(warning, build_log_content))
    warning_histogram["Other"] = len(warnings) - sum(warning_histogram.values())

    table = []
    table.append("| Warning Type   | Count |")
    table.append("|----------------|-------|")
    for warning, count in warning_histogram.items():
        if count > 0:
            table.append(f"| {warning_types[warning]} | {count} |")
    return "\n".join(table)


def generate_ctest_table(ctest_log_content):
    # No need to parse the data into Markdown table: it is already in a readable pseudo-table format
    result_lines = re.findall(r".*Test\s*#.*sec.*", ctest_log_content)
    code_block = ["```"] + result_lines + ["```"]
    return "\n".join(code_block)


def extract_ctest_summary(ctest_log_content):
    match = re.search(r".*tests passed.*tests failed.*", ctest_log_content)
    if match is None:
        return ""
    else:
        return match.group(0)


def combine_tables(environment_table, warning_table, ctest_table, ctest_summary):
    # Make the CTest summary collapsible since it can be long
    title = f"<summary><b>CTest: {ctest_summary} (expand for details)</b></summary>"
    collapsible_ctest_table = f"<details>\n{title}\n\n{ctest_table}\n\n</details>"
    # Additional empty line to separate the tables
    return "\n\n".join([environment_table, warning_table, collapsible_ctest_table])


if __name__ == "__main__":
    args = parse_arguments()
    build_log_content = read_file(args.build_log)
    ctest_log_content = read_file(args.ctest_log)

    environment_table = generate_environment_table(args.os, args.compiler_version, args.cmake_version, args.cpu_model)
    warning_table = generate_warning_table(build_log_content)
    ctest_table = generate_ctest_table(ctest_log_content)
    ctest_summary = extract_ctest_summary(ctest_log_content)
    summary = combine_tables(environment_table, warning_table, ctest_table, ctest_summary)

    with open(args.output_file, 'w', encoding="utf-8") as f:
        f.write(summary)
