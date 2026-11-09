#include "fitness.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace {

constexpr double absoluteMagnitudeFloor = 1e-12;

std::size_t checkedValueCount(std::size_t frameCount,
    std::size_t binCount,
    const std::string& representationName)
{
    if (frameCount != 0
        && binCount > std::numeric_limits<std::size_t>::max() / frameCount) {
        throw std::invalid_argument(
            representationName + " dimensions overflow its storage size.");
    }
    return frameCount * binCount;
}

void validateAnalysisMetadata(std::size_t binCount,
    std::size_t fftSize,
    std::size_t hopSize,
    std::uint32_t sampleRateHz,
    const std::string& representationName)
{
    if (fftSize == 0) {
        throw std::invalid_argument(
            representationName + " FFT size must be greater than zero.");
    }
    if (hopSize == 0 || hopSize > fftSize) {
        throw std::invalid_argument(
            representationName + " hop size is invalid.");
    }
    if (sampleRateHz == 0) {
        throw std::invalid_argument(
            representationName + " sample rate must be greater than zero.");
    }
    if (binCount != fftSize / 2 + 1) {
        throw std::invalid_argument(
            representationName + " bin count does not match its FFT size.");
    }
}

void validateSpectrogram(const Spectrogram& spectrogram)
{
    validateAnalysisMetadata(spectrogram.binCount,
        spectrogram.fftSize,
        spectrogram.hopSize,
        spectrogram.sampleRateHz,
        "Spectrogram");
    const std::size_t expectedValueCount = checkedValueCount(
        spectrogram.frameCount, spectrogram.binCount, "Spectrogram");
    if (spectrogram.magnitudes.size() != expectedValueCount) {
        throw std::invalid_argument(
            "Spectrogram magnitude count does not match its dimensions.");
    }
}

void validateReference(const FitnessReference& reference)
{
    validateAnalysisMetadata(reference.binCount,
        reference.fftSize,
        reference.hopSize,
        reference.sampleRateHz,
        "Fitness reference");
    const std::size_t expectedValueCount = checkedValueCount(
        reference.frameCount, reference.binCount, "Fitness reference");
    if (reference.logMagnitudes.size() != expectedValueCount) {
        throw std::invalid_argument(
            "Fitness reference value count does not match its dimensions.");
    }
    if (!std::isfinite(reference.magnitudeFloor)
        || reference.magnitudeFloor <= 0.0
        || !std::isfinite(reference.dynamicRangeDb)
        || reference.dynamicRangeDb <= 0.0) {
        throw std::invalid_argument(
            "Fitness reference spectral floor is invalid.");
    }
    for (double logMagnitude : reference.logMagnitudes) {
        if (!std::isfinite(logMagnitude)) {
            throw std::invalid_argument(
                "Fitness reference contains a non-finite log magnitude.");
        }
    }
}

void validateCompatibility(
    const Spectrogram& candidate, const FitnessReference& target)
{
    validateSpectrogram(candidate);
    validateReference(target);

    if (candidate.frameCount != target.frameCount
        || candidate.binCount != target.binCount) {
        throw std::invalid_argument(
            "Cannot compare spectrograms with different dimensions.");
    }
    if (candidate.fftSize != target.fftSize
        || candidate.hopSize != target.hopSize
        || candidate.sampleRateHz != target.sampleRateHz) {
        throw std::invalid_argument(
            "Cannot compare spectrograms with different analysis settings.");
    }
}

}  // namespace

double magnitudeToLog(double magnitude)
{
    return magnitudeToLog(magnitude, absoluteMagnitudeFloor);
}

double magnitudeToLog(double magnitude, double magnitudeFloor)
{
    if (!std::isfinite(magnitude) || magnitude < 0.0) {
        throw std::invalid_argument(
            "Spectral magnitude must be finite and non-negative.");
    }
    if (!std::isfinite(magnitudeFloor) || magnitudeFloor <= 0.0)
        throw std::invalid_argument(
            "Magnitude floor must be finite and positive.");

    const double logMagnitude = std::log(std::max(magnitude, magnitudeFloor));
    if (!std::isfinite(logMagnitude)) {
        throw std::runtime_error(
            "Log-magnitude conversion produced a non-finite value.");
    }
    return logMagnitude;
}

FitnessReference createFitnessReference(
    const Spectrogram& target, double dynamicRangeDb)
{
    validateSpectrogram(target);
    if (!std::isfinite(dynamicRangeDb) || dynamicRangeDb <= 0.0)
        throw std::invalid_argument(
            "Fitness dynamic range must be finite and positive.");
    double targetMaximumMagnitude = 0.0;
    for (double magnitude : target.magnitudes) {
        if (!std::isfinite(magnitude) || magnitude < 0.0)
            throw std::invalid_argument(
                "Target contains an invalid magnitude.");
        targetMaximumMagnitude = std::max(targetMaximumMagnitude, magnitude);
    }
    const double relativeFloor =
        targetMaximumMagnitude * std::pow(10.0, -dynamicRangeDb / 20.0);
    const double magnitudeFloor = targetMaximumMagnitude == 0.0
        ? absoluteMagnitudeFloor
        : std::max(relativeFloor, absoluteMagnitudeFloor);

    FitnessReference reference{
        .frameCount = target.frameCount,
        .binCount = target.binCount,
        .fftSize = target.fftSize,
        .hopSize = target.hopSize,
        .sampleRateHz = target.sampleRateHz,
        .magnitudeFloor = magnitudeFloor,
        .dynamicRangeDb = dynamicRangeDb,
        .logMagnitudes = {},
    };
    reference.logMagnitudes.reserve(target.magnitudes.size());
    for (double magnitude : target.magnitudes) {
        reference.logMagnitudes.push_back(
            magnitudeToLog(magnitude, magnitudeFloor));
    }
    return reference;
}

double calculateFrameSquaredError(std::span<const double> candidateMagnitudes,
    const FitnessReference& target,
    std::size_t frameIndex)
{
    validateReference(target);
    if (frameIndex >= target.frameCount) {
        throw std::out_of_range("Fitness frame index is out of range.");
    }
    if (candidateMagnitudes.size() != target.binCount) {
        throw std::invalid_argument("Candidate frame has the wrong bin count.");
    }
    double error = 0.0;
    const std::size_t offset = frameIndex * target.binCount;
    for (std::size_t bin = 0; bin < target.binCount; ++bin) {
        const double difference =
            magnitudeToLog(candidateMagnitudes[bin], target.magnitudeFloor)
            - target.logMagnitudes[offset + bin];
        const double squared = difference * difference;
        if (!std::isfinite(squared)
            || error > std::numeric_limits<double>::max() - squared) {
            throw std::runtime_error(
                "Frame fitness accumulation produced a non-finite value.");
        }
        error += squared;
    }
    return error;
}

SpectralResidual calculateSpectralResidual(
    const Spectrogram& champion, const FitnessReference& target)
{
    validateCompatibility(champion, target);
    SpectralResidual residual{
        .frameCount = champion.frameCount,
        .binCount = champion.binCount,
        .fftSize = champion.fftSize,
        .hopSize = champion.hopSize,
        .sampleRateHz = champion.sampleRateHz,
        .values = std::vector<double>(champion.magnitudes.size()),
    };
    for (std::size_t index = 0; index < champion.magnitudes.size(); ++index)
        residual.values[index] = target.logMagnitudes[index]
            - magnitudeToLog(champion.magnitudes[index], target.magnitudeFloor);
    return residual;
}

std::vector<ResidualHotspot> findResidualHotspots(
    const SpectralResidual& residual, const HotspotSettings& settings)
{
    const std::size_t expected = checkedValueCount(
        residual.frameCount, residual.binCount, "Spectral residual");
    if (residual.values.size() != expected || residual.fftSize == 0
        || residual.hopSize == 0 || residual.hopSize > residual.fftSize
        || residual.binCount != residual.fftSize / 2 + 1
        || residual.sampleRateHz == 0)
        throw std::invalid_argument("Spectral residual metadata is invalid.");
    if (!std::isfinite(settings.minimumResidualMagnitude)
        || settings.minimumResidualMagnitude < 0.0)
        throw std::invalid_argument("Minimum hotspot residual is invalid.");
    std::vector<bool> suppressed(expected, false);
    std::vector<ResidualHotspot> hotspots;
    hotspots.reserve(std::min(settings.maximumHotspots, expected));
    while (hotspots.size() < settings.maximumHotspots) {
        std::size_t bestIndex = expected;
        double bestMagnitude = settings.minimumResidualMagnitude;
        for (std::size_t index = 0; index < expected; ++index) {
            const double value = residual.values[index];
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "Spectral residual contains a non-finite value.");
            const double magnitude = std::abs(value);
            if (!suppressed[index] && magnitude > bestMagnitude) {
                bestIndex = index;
                bestMagnitude = magnitude;
            }
        }
        if (bestIndex == expected)
            break;
        const std::size_t frame = bestIndex / residual.binCount;
        const std::size_t bin = bestIndex % residual.binCount;
        const double centerSample =
            static_cast<double>(frame * residual.hopSize)
            + static_cast<double>(residual.fftSize) / 2.0;
        hotspots.push_back({frame,
            bin,
            residual.values[bestIndex],
            bestMagnitude,
            centerSample / static_cast<double>(residual.sampleRateHz),
            binToFrequencyHz(bin, residual.fftSize, residual.sampleRateHz)});
        const std::size_t firstFrame = frame > settings.suppressionFrames
            ? frame - settings.suppressionFrames
            : 0;
        const std::size_t lastFrame = frame
            + std::min(
                settings.suppressionFrames, residual.frameCount - 1 - frame);
        const std::size_t firstBin =
            bin > settings.suppressionBins ? bin - settings.suppressionBins : 0;
        const std::size_t lastBin = bin
            + std::min(settings.suppressionBins, residual.binCount - 1 - bin);
        for (std::size_t nearbyFrame = firstFrame; nearbyFrame <= lastFrame;
             ++nearbyFrame)
            for (std::size_t nearbyBin = firstBin; nearbyBin <= lastBin;
                 ++nearbyBin)
                suppressed[nearbyFrame * residual.binCount + nearbyBin] = true;
    }
    return hotspots;
}

FitnessState createFitnessState(
    const Spectrogram& candidate, const FitnessReference& target)
{
    validateCompatibility(candidate, target);
    FitnessState state;
    state.frameSquaredErrors.reserve(candidate.frameCount);
    for (std::size_t frame = 0; frame < candidate.frameCount; ++frame) {
        const double error = calculateFrameSquaredError(
            std::span<const double>(candidate.magnitudes)
                .subspan(frame * candidate.binCount, candidate.binCount),
            target,
            frame);
        state.frameSquaredErrors.push_back(error);
        if (state.totalSquaredError
            > std::numeric_limits<double>::max() - error) {
            throw std::runtime_error(
                "Fitness-state accumulation produced a non-finite value.");
        }
        state.totalSquaredError += error;
    }
    const std::size_t valueCount = candidate.frameCount * candidate.binCount;
    state.fitness = valueCount == 0
        ? 0.0
        : state.totalSquaredError / static_cast<double>(valueCount);
    return state;
}

double calculateSpectralFitness(
    const Spectrogram& candidate, const FitnessReference& target)
{
    validateCompatibility(candidate, target);
    return createFitnessState(candidate, target).fitness;
}

double calculateSpectralFitness(
    const Spectrogram& candidate, const Spectrogram& target)
{
    return calculateSpectralFitness(candidate, createFitnessReference(target));
}

double calculateAudioFitness(
    const AudioBuffer& candidate, const FitnessReference& target)
{
    const Spectrogram candidateSpectrogram =
        calculateSpectrogram(candidate, target.fftSize, target.hopSize);
    return calculateSpectralFitness(candidateSpectrogram, target);
}

double calculateAudioFitness(const AudioBuffer& candidate,
    const FitnessReference& target,
    StftAnalyzer& analyzer)
{
    if (analyzer.fftSize() != target.fftSize
        || analyzer.hopSize() != target.hopSize) {
        throw std::invalid_argument(
            "STFT analyzer settings do not match the fitness reference.");
    }
    return calculateSpectralFitness(analyzer.calculate(candidate), target);
}
