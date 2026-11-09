#include "audio.hpp"

#include <sndfile.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct SndFileCloser {
    void operator()(SNDFILE* file) const noexcept
    {
        if (file != nullptr) {
            sf_close(file);
        }
    }
};

using SndFilePtr = std::unique_ptr<SNDFILE, SndFileCloser>;

std::runtime_error audioError(const std::string& operation,
    const std::string& path,
    SNDFILE* file = nullptr)
{
    return std::runtime_error(
        operation + " '" + path + "': " + sf_strerror(file));
}

}  // namespace

AudioBuffer loadAudio(const std::string& path)
{
    SF_INFO info{};
    SndFilePtr file(sf_open(path.c_str(), SFM_READ, &info));

    if (!file) {
        throw audioError("Unable to open audio file", path);
    }

    const int containerFormat = info.format & SF_FORMAT_TYPEMASK;
    if (containerFormat != SF_FORMAT_WAV) {
        throw std::runtime_error("Input must be a WAV file: '" + path + "'.");
    }

    if (info.channels != 1) {
        throw std::runtime_error("Input must be mono, but '" + path + "' has "
            + std::to_string(info.channels) + " channels.");
    }

    if (info.samplerate <= 0) {
        throw std::runtime_error(
            "Input has an invalid sample rate: '" + path + "'.");
    }

    if (info.frames < 0) {
        throw std::runtime_error(
            "Input has an invalid sample count: '" + path + "'.");
    }

    const auto frameCount = static_cast<std::uintmax_t>(info.frames);
    if (frameCount > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Input is too large to load: '" + path + "'.");
    }

    AudioBuffer audio;
    audio.sampleRateHz = static_cast<std::uint32_t>(info.samplerate);
    audio.samples.resize(static_cast<std::size_t>(frameCount));

    const sf_count_t framesRead =
        sf_readf_double(file.get(), audio.samples.data(), info.frames);

    if (framesRead != info.frames) {
        throw audioError("Unable to read all samples from", path, file.get());
    }

    return audio;
}

void saveAudio(const std::string& path, const AudioBuffer& audio)
{
    if (audio.sampleRateHz == 0
        || audio.sampleRateHz
            > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Cannot save audio with an invalid sample rate.");
    }

    if (audio.samples.size()
        > static_cast<std::size_t>(std::numeric_limits<sf_count_t>::max())) {
        throw std::runtime_error("Audio buffer is too large to save.");
    }

    SF_INFO info{};
    info.channels = 1;
    info.samplerate = static_cast<int>(audio.sampleRateHz);
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SndFilePtr file(sf_open(path.c_str(), SFM_WRITE, &info));
    if (!file) {
        throw audioError("Unable to create audio file", path);
    }

    const auto frameCount = static_cast<sf_count_t>(audio.samples.size());
    const sf_count_t framesWritten =
        sf_writef_double(file.get(), audio.samples.data(), frameCount);

    if (framesWritten != frameCount) {
        throw audioError("Unable to write all samples to", path, file.get());
    }

    sf_write_sync(file.get());

    const int writeError = sf_error(file.get());
    if (writeError != SF_ERR_NO_ERROR) {
        throw audioError("Unable to finalize audio file", path, file.get());
    }
}

void validateCompatibleAudio(
    const AudioBuffer& source, const AudioBuffer& target)
{
    if (source.sampleRateHz != target.sampleRateHz) {
        throw std::runtime_error("Source and target sample rates must match.");
    }

    if (source.samples.size() != target.samples.size()) {
        throw std::runtime_error("Source and target sample counts must match.");
    }
}
