# Apply the patch queue in patches/llama.cpp/ to the 3rdparty/llama.cpp submodule.
#
# The submodule (isHuangXin/llama.cpp) is not writable from this repository, so
# fixes that must live inside it are carried as patches. Each patch is applied
# at most once: if it is already applied (git apply --reverse --check succeeds)
# it is skipped; if it neither applies nor reverse-applies, configuration fails.
#
# Use from CMake:  include(cmake/ApplyLlamaPatches.cmake)  (option BITNET_APPLY_LLAMA_PATCHES)
# Use standalone:  cmake -P cmake/ApplyLlamaPatches.cmake

find_package(Git QUIET)
if (NOT GIT_FOUND)
    message(FATAL_ERROR "git is required to apply patches/llama.cpp/*.patch (or set -DBITNET_APPLY_LLAMA_PATCHES=OFF)")
endif()

get_filename_component(_bitnet_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_llama_dir "${_bitnet_root}/3rdparty/llama.cpp")
file(GLOB _llama_patches "${_bitnet_root}/patches/llama.cpp/*.patch")
list(SORT _llama_patches)

foreach (_patch IN LISTS _llama_patches)
    get_filename_component(_name "${_patch}" NAME)
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${_patch}"
        WORKING_DIRECTORY "${_llama_dir}" RESULT_VARIABLE _applied OUTPUT_QUIET ERROR_QUIET)
    if (_applied EQUAL 0)
        message(STATUS "llama.cpp patch already applied: ${_name}")
        continue()
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${_patch}"
        WORKING_DIRECTORY "${_llama_dir}" RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if (NOT _rc EQUAL 0)
        message(FATAL_ERROR "Cannot apply ${_name} to ${_llama_dir}: ${_err}")
    endif()
    message(STATUS "llama.cpp patch applied: ${_name}")
endforeach()
