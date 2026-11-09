#include "audio.hpp"

#include <sndfile.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory()
    {
        const auto uniqueValue =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
            / ("timbrogenesis-tests-" + std::to_string(uniqueValue));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void requireThrows(Function function, const std::string& expectedMessage)
{
    try {
        function();
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(expectedMessage)
                != std::string::npos,
            "Unexpected error message: " + std::string(error.what()));
        return;
    }

    throw std::runtime_error(
        "Expected an exception containing: " + expectedMessage);
}

void writeStereoWav(const std::filesystem::path& path)
{
    SF_INFO info{};
    info.channels = 2;
    info.samplerate = 48'000;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* file = sf_open(path.string().c_str(), SFM_WRITE, &info);
    require(file != nullptr, "Could not create stereo test WAV.");

    const std::vector<double> samples(64, 0.0);
    const sf_count_t framesWritten = sf_writef_double(file, samples.data(), 32);
    const int closeResult = sf_close(file);

    require(framesWritten == 32, "Could not write stereo test WAV.");
    require(closeResult == 0, "Could not close stereo test WAV.");
}

void testRoundTrip(const std::filesystem::path& directory)
{
    AudioBuffer original;
    original.sampleRateHz = 48'000;
    original.samples = {0.0, 0.25, -0.5, 0.75, -1.0, 1.0};

    const auto path = directory / "roundtrip.wav";
    saveAudio(path.string(), original);
    const AudioBuffer loaded = loadAudio(path.string());

    require(loaded.sampleRateHz == original.sampleRateHz,
        "Round trip changed the sample rate.");
    require(loaded.samples.size() == original.samples.size(),
        "Round trip changed the sample count.");

    for (std::size_t index = 0; index < original.samples.size(); ++index) {
        require(
            std::abs(loaded.samples[index] - original.samples[index]) < 1e-7,
            "Round trip changed a sample beyond tolerance.");
    }
}

void testStereoRejection(const std::filesystem::path& directory)
{
    const auto path = directory / "stereo.wav";
    writeStereoWav(path);

    requireThrows(
        [&path]() { loadAudio(path.string()); }, "Input must be mono");
}

void testCompatibilityValidation()
{
    const AudioBuffer reference{{0.0, 0.0}, 48'000};
    const AudioBuffer compatible{{1.0, -1.0}, 48'000};
    validateCompatibleAudio(reference, compatible);

    const AudioBuffer wrongRate{{0.0, 0.0}, 44'100};
    requireThrows([&reference, &wrongRate]() {
        validateCompatibleAudio(reference, wrongRate);
    }, "sample rates must match");

    const AudioBuffer wrongLength{{0.0}, 48'000};
    requireThrows([&reference, &wrongLength]() {
        validateCompatibleAudio(reference, wrongLength);
    }, "sample counts must match");
}

}  // namespace

int main()
{
    try {
        const TemporaryDirectory temporaryDirectory;
        testRoundTrip(temporaryDirectory.path());
        testStereoRejection(temporaryDirectory.path());
        testCompatibilityValidation();
        std::cout << "All audio tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Audio test failure: " << error.what() << '\n';
        return 1;
    }
}
