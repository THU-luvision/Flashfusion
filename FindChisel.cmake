###############################################################################
# Find Chisel
#
# This sets the following variables:
# EFUSION_FOUND - True if EFUSION was found.
# EFUSION_INCLUDE_DIRS - Directories containing the EFUSION include files.
# EFUSION_LIBRARIES - Libraries needed to use EFUSION.

find_path(CHISEL_INCLUDE_DIR open_chisel/Chisel.h
          PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/CHISEL/src/
          PATH_SUFFIXES CHISEL
)


find_library(CHISEL_LIBRARY
             NAMES libchisel.so
             PATHS
               ${CMAKE_CURRENT_SOURCE_DIR}/CHISEL/build
               ${CMAKE_CURRENT_SOURCE_DIR}/CHISEL/src/build
             PATH_SUFFIXES ${CHISEL_PATH_SUFFIXES}
)

set(CHISEL_INCLUDE_DIR  ${CHISEL_INCLUDE_DIR})
message(STATUS "chisel dir: ${CHISEL_INCLUDE_DIR}")
set(CHISEL_LIBRARIES ${CHISEL_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHISEL DEFAULT_MSG CHISEL_LIBRARY CHISEL_INCLUDE_DIR)

if(NOT WIN32)
  mark_as_advanced(CHISEL_LIBRARY CHISEL_INCLUDE_DIR)
endif()


