import argparse
import re

from collections import Counter


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
    warning_regex = re.compile(
        r"""
            \[(\-W[a-zA-Z0-9\-=]+)\] |  # GCC/Clang warnings: "[-W<some-flag>]" or "[-W<some-flag>=<value>]"
            ((?:STL|C|D|LNK)\d{4})     # MSVC warnings: "<STL|C|D|LNK>xxxx"
        """,
        re.VERBOSE
    )
    warnings = tuple(
        match.group(1) or match.group(2)
        for match in warning_regex.finditer(build_log_content)
    )
    warning_histogram = Counter(warnings)

    warning_examples = {}
    for w in warning_histogram:
        matches = tuple(line for line in build_log_content.splitlines() if w in line)
        # Prioritize warnigs from the core library ("include" is expected in the message as a part of the path)
        lib_match = next((m for m in matches if "include" in m), None)
        first_match = matches[0]
        warning_examples[w] = lib_match or first_match

    table = []
    table.append("| Warning Type   | Count | Message example |")
    table.append("|----------------|-------|-----------------|")
    for warning, count in warning_histogram.items():
        example = warning_examples[warning]
        table.append(f"| {warning} | {count} | {example} |")
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
