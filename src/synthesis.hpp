#pragma once

#include "audio.hpp"

#include <cstddef>

struct SampleRange {
    std::size_t beginSample = 0;
    std::size_t endSample = 0;
};

struct SineAtom {
    double startSeconds = 0.0;
    double durationSeconds = 0.0;
    double frequencyHz = 0.0;
    double amplitude = 0.0;
    double phaseRadians = 0.0;
    bool operator==(const SineAtom&) const = default;
};

double hannWindow(std::size_t sampleIndex, std::size_t sampleCount);
SampleRange sineAtomSampleRange(const AudioBuffer& audio, const SineAtom& atom);
void applySineAtom(AudioBuffer& audio, const SineAtom& atom);
void removeSineAtom(AudioBuffer& audio, const SineAtom& atom);
