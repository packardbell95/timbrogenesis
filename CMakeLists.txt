cmake_minimum_required(VERSION 3.16)

project(Timbrogenesis VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SNDFILE QUIET IMPORTED_TARGET sndfile)
endif()

if(NOT TARGET PkgConfig::SNDFILE)
    find_path(SNDFILE_INCLUDE_DIR NAMES sndfile.h)
    find_library(SNDFILE_LIBRARY NAMES sndfile)

    if(NOT SNDFILE_INCLUDE_DIR OR NOT SNDFILE_LIBRARY)
        message(FATAL_ERROR
            "libsndfile development files were not found. Install libsndfile1-dev "
            "(Debian/Ubuntu) or provide sndfile.h and the sndfile library.")
    endif()

    add_library(SndFile::sndfile UNKNOWN IMPORTED)
    set_target_properties(SndFile::sndfile PROPERTIES
        IMPORTED_LOCATION "${SNDFILE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SNDFILE_INCLUDE_DIR}"
    )
    set(SNDFILE_TARGET SndFile::sndfile)
else()
    set(SNDFILE_TARGET PkgConfig::SNDFILE)
endif()

if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
endif()

if(NOT TARGET PkgConfig::FFTW3)
    find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
    find_library(FFTW3_LIBRARY NAMES fftw3)

    if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
        message(FATAL_ERROR
            "FFTW3 development files were not found. Install libfftw3-dev "
            "(Debian/Ubuntu) or provide fftw3.h and the FFTW3 library.")
    endif()

    add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
        IMPORTED_LOCATION "${FFTW3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
    )
    set(FFTW3_TARGET FFTW3::fftw3)
else()
    set(FFTW3_TARGET PkgConfig::FFTW3)
endif()

add_executable(timbrogenesis
    src/audio.cpp
    src/evolution.cpp
    src/fitness.cpp
    src/main.cpp
    src/metadata.cpp
    src/mutation.cpp
    src/spectral_morph.cpp
    src/spectrum.cpp
    src/synthesis.cpp
)

target_include_directories(timbrogenesis PRIVATE src)
target_link_libraries(timbrogenesis PRIVATE
    ${SNDFILE_TARGET}
    ${FFTW3_TARGET}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(timbrogenesis PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
    )
endif()

if(BUILD_TESTING)
    add_executable(audio-tests
        src/audio.cpp
        tests/audio_tests.cpp
    )
    target_include_directories(audio-tests PRIVATE src)
    target_link_libraries(audio-tests PRIVATE ${SNDFILE_TARGET})

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(audio-tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
    endif()

    add_test(NAME audio-tests COMMAND audio-tests)

    add_executable(synthesis-tests
        src/synthesis.cpp
        tests/synthesis_tests.cpp
    )
    target_include_directories(synthesis-tests PRIVATE src)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(synthesis-tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
    endif()

    add_test(NAME synthesis-tests COMMAND synthesis-tests)

    add_executable(spectrum-tests
        src/spectrum.cpp
        tests/spectrum_tests.cpp
    )
    target_include_directories(spectrum-tests PRIVATE src)
    target_link_libraries(spectrum-tests PRIVATE ${FFTW3_TARGET})

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(spectrum-tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
    endif()

    add_test(NAME spectrum-tests COMMAND spectrum-tests)

    add_executable(spectral-morph-tests
        src/spectral_morph.cpp
        src/spectrum.cpp
        tests/spectral_morph_tests.cpp
    )
    target_include_directories(spectral-morph-tests PRIVATE src)
    target_link_libraries(spectral-morph-tests PRIVATE ${FFTW3_TARGET})

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(spectral-morph-tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
    endif()

    add_test(NAME spectral-morph-tests COMMAND spectral-morph-tests)

    add_executable(spectral-morph-integration-tests
        src/audio.cpp
        src/evolution.cpp
        src/fitness.cpp
        src/metadata.cpp
        src/mutation.cpp
        src/spectral_morph.cpp
        src/spectrum.cpp
        src/synthesis.cpp
        tests/spectral_morph_integration_tests.cpp
    )
    target_include_directories(spectral-morph-integration-tests PRIVATE src)
    target_link_libraries(spectral-morph-integration-tests PRIVATE
        ${SNDFILE_TARGET}
        ${FFTW3_TARGET}
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(spectral-morph-integration-tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
    endif()

    add_test(
        NAME spectral-morph-integration-tests
        COMMAND spectral-morph-integration-tests
    )

    add_test(
        NAME spectral-morph-rejects-evolution-options
        COMMAND timbrogenesis source.wav target.wav
                --mode spectral-morph --candidates 1
    )
    set_tests_properties(spectral-morph-rejects-evolution-options PROPERTIES
        PASS_REGULAR_EXPRESSION "--candidates is only valid in evolution mode"
    )

    add_test(
        NAME evolution-rejects-morph-options
        COMMAND timbrogenesis source.wav target.wav
                --mode evolution --morph-steps 10
    )
    set_tests_properties(evolution-rejects-morph-options PROPERTIES
        PASS_REGULAR_EXPRESSION "--morph-steps is only valid in spectral-morph mode"
    )

    add_executable(fitness-tests
        src/fitness.cpp
        src/spectrum.cpp
        src/synthesis.cpp
        tests/fitness_tests.cpp
    )
    target_include_directories(fitness-tests PRIVATE src)
    target_link_libraries(fitness-tests PRIVATE ${FFTW3_TARGET})

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(fitness-tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
    endif()

    add_test(NAME fitness-tests COMMAND fitness-tests)

    add_executable(evolution-tests
        src/audio.cpp
        src/evolution.cpp
        src/fitness.cpp
        src/metadata.cpp
        src/mutation.cpp
        src/spectral_morph.cpp
        src/spectrum.cpp
        src/synthesis.cpp
        tests/evolution_tests.cpp
    )
    target_include_directories(evolution-tests PRIVATE src)
    target_link_libraries(evolution-tests PRIVATE
        ${SNDFILE_TARGET}
        ${FFTW3_TARGET}
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(evolution-tests PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
    endif()

    add_test(NAME evolution-tests COMMAND evolution-tests)
endif()
