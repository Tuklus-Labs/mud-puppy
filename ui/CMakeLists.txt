cmake_minimum_required(VERSION 3.20)
project(mud-puppy-studio VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Phos framework (FindPhos.cmake sets up Phos::Phos with all transitive deps)
list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/Projects/phos/cmake")
find_package(Phos REQUIRED)

# libsoup-3.0 for WebSocket client (not provided by Phos::Phos)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SOUP libsoup-3.0)
if(NOT SOUP_FOUND)
    message(FATAL_ERROR
        "libsoup-3.0 not found. Install with:\n"
        "  Arch:   sudo pacman -S libsoup3\n"
        "  Debian: sudo apt install libsoup-3.0-dev\n"
        "  Fedora: sudo dnf install libsoup3-devel")
endif()

# libcurl for HF API calls
find_package(CURL REQUIRED)

add_executable(mud-puppy-studio
    src/main.cpp
    src/training_manager.cpp
    src/ws_bridge.cpp
    src/hf_client.cpp
    src/dataset_peek.cpp
    src/checkpoint_scan.cpp
)

target_include_directories(mud-puppy-studio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SOUP_INCLUDE_DIRS}
)

target_link_libraries(mud-puppy-studio PRIVATE
    Phos::Phos
    ${SOUP_LIBRARIES}
    CURL::libcurl
)

# Copy manifest.toml next to binary so phos::resolve_manifest() finds it.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/manifest.toml
    ${CMAKE_CURRENT_BINARY_DIR}/manifest.toml
    COPYONLY
)

# Copy web/dist (when built by the web workstream) next to the binary.
# This block is intentionally a no-op until the web workstream produces dist/.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/web/dist)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/web/dist
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/web)
endif()
