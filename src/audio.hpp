#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct AudioBuffer {
    std::vector<double> samples;
    std::uint32_t sampleRateHz = 0;
};

AudioBuffer loadAudio(const std::string& path);
void saveAudio(const std::string& path, const AudioBuffer& audio);

void validateCompatibleAudio(
    const AudioBuffer& source, const AudioBuffer& target);
