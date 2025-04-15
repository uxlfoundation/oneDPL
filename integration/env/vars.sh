#!/bin/sh
# shellcheck shell=sh

##===----------------------------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

# ############################################################################

# executing function in a *subshell* to localize vars and effects on `cd`
rreadlink() (
  target=$1 fname="" targetDir="" CDPATH=
  { \unalias command; \unset -f command; } >/dev/null 2>&1 || :
  # shellcheck disable=2030,2034
  [ -n "${ZSH_VERSION:-}" ] && options[POSIX_BUILTINS]=on
  while :; do
    [ -L "$target" ] || [ -e "$target" ] || { command printf '%s\n' "   ERROR: rreadlink(): '$target' does not exist." >&2; return 1; }
    command cd "$(command dirname -- "$target")" >/dev/null 2>&1 || { command printf '%s\n' ":: ERROR: rreadlink() processing failure" >&2; return 1; }
    fname=$(command basename -- "$target")
    [ "$fname" = '/' ] && fname=''
    if [ -L "$fname" ] ; then
      target=$(command ls -l "$fname")
      target=${target#* -> } # delete everything left of first " -> " string
      continue
    fi
    break
  done
  targetDir=$(command pwd -P)
  if   [ "$fname" = '.' ] ;  then
    command printf '%s\n' "${targetDir%/}"
  elif [ "$fname" = '..' ] ; then
    command printf '%s\n' "$(command dirname -- "${targetDir}")"
  else
    command printf '%s\n' "${targetDir%/}/$fname"
  fi
)

_vars_get_proc_name() {
  # shellcheck disable=2031
  if [ -n "${ZSH_VERSION:-}" ] ; then
    script="$(ps -p "$$" -o comm=)"
  else
    script="$1"
    while [ -L "$script" ] ; do
      script="$(readlink "$script")"
    done
  fi
  basename -- "$script"
}

_vars_this_script_name="vars.sh"
if [ "$_vars_this_script_name" = "$(_vars_get_proc_name "$0")" ] ; then
  echo "   ERROR: Incorrect usage: this script must be sourced."
  echo "   Usage: . path/to/${_vars_this_script_name}"
  return 255 2>/dev/null || exit 255
fi


# ############################################################################

# Prepend path segment(s) to path-like env vars (PATH, CPLUS_INCLUDE_PATH, etc.).

# prepend_path() avoids dangling ":" that affects some env vars (PATH and CPLUS_INCLUDE_PATH)
# prepend_manpath() includes dangling ":" needed by MANPATH.
# PATH > https://www.gnu.org/software/libc/manual/html_node/Standard-Environment.html
# MANPATH > https://manpages.debian.org/stretch/man-db/manpath.1.en.html

# Usage:
#   env_var=$(prepend_path "$prepend_to_var" "$existing_env_var")
#   export env_var
#
#   env_var=$(prepend_manpath "$prepend_to_var" "$existing_env_var")
#   export env_var
#
# Inputs:
#   $1 == path segment to be prepended to $2
#   $2 == value of existing path-like environment variable

prepend_path() (
  path_to_add="$1"
  path_is_now="$2"

  if [ "" = "${path_is_now}" ] ; then   # avoid dangling ":"
    printf "%s" "${path_to_add}"
  else
    printf "%s" "${path_to_add}:${path_is_now}"
  fi
)

prepend_manpath() (
  path_to_add="$1"
  path_is_now="$2"

  if [ "" = "${path_is_now}" ] ; then   # include dangling ":"
    printf "%s" "${path_to_add}:"
  else
    printf "%s" "${path_to_add}:${path_is_now}"
  fi
)


# ############################################################################

# Extract the name and location of this sourced script.

# Generally, "ps -o comm=" is limited to a 15 character result, but it works
# fine for this usage, because we are primarily interested in finding the name
# of the execution shell, not the name of any calling script.

vars_script_name=""
vars_script_shell="$(ps -p "$$" -o comm=)"
# ${var:-} needed to pass "set -eu" checks
# see https://unix.stackexchange.com/a/381465/103967
# see https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_02
if [ -n "${ZSH_VERSION:-}" ] && [ -n "${ZSH_EVAL_CONTEXT:-}" ] ; then     # zsh 5.x and later
  # shellcheck disable=2249
  case $ZSH_EVAL_CONTEXT in (*:file*) vars_script_name="${(%):-%x}" ;; esac ;
elif [ -n "${KSH_VERSION:-}" ] ; then                                     # ksh, mksh or lksh
  if [ "$(set | grep -Fq "KSH_VERSION=.sh.version" ; echo $?)" -eq 0 ] ; then # ksh
    vars_script_name="${.sh.file}" ;
  else # mksh or lksh or [lm]ksh masquerading as ksh or sh
    # force [lm]ksh to issue error msg; which contains this script's path/filename, e.g.:
    # mksh: /home/ubuntu/intel/oneapi/vars.sh[137]: ${.sh.file}: bad substitution
    vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*sh: \(.*\)\[[0-9]*\]:')" ;
  fi
elif [ -n "${BASH_VERSION:-}" ] ; then        # bash
  # shellcheck disable=2128,3028
  (return 0 2>/dev/null) && vars_script_name="${BASH_SOURCE}" ;
elif [ "dash" = "$vars_script_shell" ] ; then # dash
  # force dash to issue error msg; which contains this script's rel/path/filename, e.g.:
  # dash: 146: /home/ubuntu/intel/oneapi/vars.sh: Bad substitution
  vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
  vars_script_name="$(expr "${vars_script_name:-}" : '^.*dash: [0-9]*: \(.*\):')" ;
elif [ "sh" = "$vars_script_shell" ] ; then   # could be dash masquerading as /bin/sh
  # force a shell error msg; which should contain this script's path/filename
  # sample error msg shown; assume this file is named "vars.sh"; as required by setvars.sh
  vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
  if [ "$(printf "%s" "$vars_script_name" | grep -Eq "sh: [0-9]+: .*vars\.sh: " ; echo $?)" -eq 0 ] ; then # dash as sh
    # sh: 155: /home/ubuntu/intel/oneapi/vars.sh: Bad substitution
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*sh: [0-9]*: \(.*\):')" ;
  fi
else  # unrecognized shell or dash being sourced from within a user's script
  # force a shell error msg; which should contain this script's path/filename
  # sample error msg shown; assume this file is named "vars.sh"; as required by setvars.sh
  vars_script_name="$( (echo "${.sh.file}") 2>&1 )" || : ;
  if [ "$(printf "%s" "$vars_script_name" | grep -Eq "^.+: [0-9]+: .*vars\.sh: " ; echo $?)" -eq 0 ] ; then # dash
    # .*: 164: intel/oneapi/vars.sh: Bad substitution
    vars_script_name="$(expr "${vars_script_name:-}" : '^.*: [0-9]*: \(.*\):')" ;
  else
    vars_script_name="" ;
  fi
fi

if [ "" = "$vars_script_name" ] ; then
  >&2 echo "   ERROR: Unable to proceed: possible causes listed below."
  >&2 echo "   This script must be sourced. Did you execute or source this script?" ;
  >&2 echo "   Unrecognized/unsupported shell (supported: bash, zsh, ksh, m/lksh, dash)." ;
  >&2 echo "   May fail in dash if you rename this script (assumes \"vars.sh\")." ;
  >&2 echo "   Can be caused by sourcing from ZSH version 4.x or older." ;
  return 255 2>/dev/null || exit 255
fi


# ############################################################################

_onedpl_scrip_path=$(dirname -- "$(rreadlink "${vars_script_name:-}")")
DPL_ROOT=$(dirname -- "${_onedpl_scrip_path}") ; export DPL_ROOT
CPLUS_INCLUDE_PATH=$(prepend_path "${DPL_ROOT}/include" "${CPLUS_INCLUDE_PATH:-}") ; export CPLUS_INCLUDE_PATH
PKG_CONFIG_PATH=$(prepend_path "${DPL_ROOT}/lib/pkgconfig" "${PKG_CONFIG_PATH:-}") ; export PKG_CONFIG_PATH
CMAKE_PREFIX_PATH=$(prepend_path "${DPL_ROOT}/lib/cmake/oneDPL" "${CMAKE_PREFIX_PATH:-}") ; export CMAKE_PREFIX_PATH

DPL_LIB_NAME="libpstloffload.so"

if [ -e "$DPL_ROOT/lib/$DPL_LIB_NAME" ] ; then
    LIBRARY_PATH=$(prepend_path "${DPL_ROOT}/lib" "${LIBRARY_PATH:-}") ; export LIBRARY_PATH
    LD_LIBRARY_PATH=$(prepend_path "${DPL_ROOT}/lib" "${LD_LIBRARY_PATH:-}") ; export LD_LIBRARY_PATH
fi
