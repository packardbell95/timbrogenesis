#include "synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace {

void validateAtom(const AudioBuffer& audio, const SineAtom& atom)
{
    if (audio.sampleRateHz == 0) {
        throw std::invalid_argument(
            "Cannot apply a sine atom to audio with an invalid sample rate.");
    }

    if (audio.samples.empty()) {
        throw std::invalid_argument(
            "Cannot apply a sine atom to an empty audio buffer.");
    }

    if (!std::isfinite(atom.startSeconds) || atom.startSeconds < 0.0) {
        throw std::invalid_argument(
            "Sine atom start time must be finite and non-negative.");
    }

    if (!std::isfinite(atom.durationSeconds) || atom.durationSeconds <= 0.0) {
        throw std::invalid_argument(
            "Sine atom duration must be finite and positive.");
    }

    if (!std::isfinite(atom.frequencyHz) || atom.frequencyHz < 0.0) {
        throw std::invalid_argument(
            "Sine atom frequency must be finite and non-negative.");
    }

    const double nyquistFrequencyHz =
        static_cast<double>(audio.sampleRateHz) / 2.0;
    if (atom.frequencyHz > nyquistFrequencyHz) {
        throw std::invalid_argument(
            "Sine atom frequency cannot exceed the Nyquist frequency.");
    }

    if (!std::isfinite(atom.amplitude) || !std::isfinite(atom.phaseRadians)) {
        throw std::invalid_argument(
            "Sine atom amplitude and phase must be finite.");
    }
}

std::size_t secondsToSamples(double seconds, std::uint32_t sampleRateHz)
{
    const double roundedSamples =
        std::round(seconds * static_cast<double>(sampleRateHz));
    const double maximumSampleCount =
        static_cast<double>(std::numeric_limits<std::size_t>::max());

    if (!std::isfinite(roundedSamples)
        || roundedSamples >= maximumSampleCount) {
        return std::numeric_limits<std::size_t>::max();
    }

    return static_cast<std::size_t>(roundedSamples);
}

}  // namespace

SampleRange sineAtomSampleRange(const AudioBuffer& audio, const SineAtom& atom)
{
    validateAtom(audio, atom);
    const double audioDurationSeconds =
        static_cast<double>(audio.samples.size())
        / static_cast<double>(audio.sampleRateHz);
    if (atom.startSeconds >= audioDurationSeconds) {
        return {audio.samples.size(), audio.samples.size()};
    }

    const std::size_t startSample =
        secondsToSamples(atom.startSeconds, audio.sampleRateHz);
    const std::size_t requestedSampleCount =
        secondsToSamples(atom.durationSeconds, audio.sampleRateHz);
    if (startSample >= audio.samples.size() || requestedSampleCount == 0) {
        const std::size_t clamped = std::min(startSample, audio.samples.size());
        return {clamped, clamped};
    }
    const std::size_t sampleCount =
        std::min(requestedSampleCount, audio.samples.size() - startSample);
    return {startSample, startSample + sampleCount};
}

double hannWindow(std::size_t sampleIndex, std::size_t sampleCount)
{
    if (sampleCount == 0 || sampleIndex >= sampleCount) {
        throw std::out_of_range("Hann window sample index is out of range.");
    }

    if (sampleCount == 1) {
        return 1.0;
    }

    const double normalizedPosition =
        static_cast<double>(sampleIndex) / static_cast<double>(sampleCount - 1);

    return 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * normalizedPosition));
}

void applySineAtom(AudioBuffer& audio, const SineAtom& atom)
{
    const SampleRange range = sineAtomSampleRange(audio, atom);
    if (range.beginSample == range.endSample) {
        return;
    }
    const std::size_t sampleCount = range.endSample - range.beginSample;

    for (std::size_t atomSample = 0; atomSample < sampleCount; ++atomSample) {
        const double timeSeconds = static_cast<double>(atomSample)
            / static_cast<double>(audio.sampleRateHz);
        const double oscillator =
            std::sin(2.0 * std::numbers::pi * atom.frequencyHz * timeSeconds
                + atom.phaseRadians);

        audio.samples[range.beginSample + atomSample] +=
            atom.amplitude * hannWindow(atomSample, sampleCount) * oscillator;
    }
}

void removeSineAtom(AudioBuffer& audio, const SineAtom& atom)
{
    SineAtom inverse = atom;
    inverse.amplitude = -inverse.amplitude;
    applySineAtom(audio, inverse);
}
