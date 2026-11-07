import argparse
import os
import platform
import re
import subprocess

from collections import Counter

NA_VALUE = "N/A"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze build logs and ctest output, and generate a summary.")
    parser.add_argument("--build-log", required=True, help="Path to the build log file.")
    parser.add_argument("--ctest-log", required=True, help="Path to the ctest log file.")
    parser.add_argument("--output-file", required=True, help="Path to the output file.")
    parser.add_argument("--compiler", help="Name of the C++ compiler executable, queried for its version.")
    return parser.parse_args()


def read_file(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        return f.read()


def first_output_line(command):
    """Return the first non-empty line the command prints, or "N/A" if it prints nothing.

    Both streams are captured and the exit code is ignored on purpose: cl writes
    its version banner to stderr and then fails on the "--version" option it does
    not recognize, but that banner is exactly what is wanted here.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return NA_VALUE
    for line in (result.stdout + result.stderr).splitlines():
        if line.strip():
            return line.strip()
    return NA_VALUE


def detect_os():
    if platform.system() == "Windows":
        # The marketing name ("Microsoft Windows Server 2022 Datacenter") says more
        # than platform.platform()'s "Windows-10-10.0.20348-SP0".
        return first_output_line(
            ["powershell", "-command", "(Get-CimInstance -ClassName Win32_OperatingSystem).Caption"])
    return platform.platform()


def detect_cpu_model():
    if platform.system() == "Linux":
        try:
            for line in read_file("/proc/cpuinfo").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return NA_VALUE
    if platform.system() == "Darwin":
        return first_output_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    # platform.processor() only reports the CPU family on Windows, so ask WMI for
    # the model name that the other platforms report.
    return first_output_line(["powershell", "-command", "(Get-CimInstance -ClassName Win32_Processor).Name"])


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

    compiler_version = first_output_line([args.compiler, "--version"]) if args.compiler else NA_VALUE
    environment_table = generate_environment_table(detect_os(), compiler_version,
                                                   first_output_line(["cmake", "--version"]), detect_cpu_model())
    warning_table = generate_warning_table(build_log_content)
    ctest_table = generate_ctest_table(ctest_log_content)
    ctest_summary = extract_ctest_summary(ctest_log_content)
    summary = combine_tables(environment_table, warning_table, ctest_table, ctest_summary)

    with open(args.output_file, 'w', encoding="utf-8") as f:
        f.write(summary)

    # Publish to the job summary as well, so that callers do not have to copy the
    # file there themselves in a platform-specific way.
    step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_file:
        with open(step_summary_file, 'a', encoding="utf-8") as f:
            f.write(summary)
