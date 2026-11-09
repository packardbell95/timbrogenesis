#pragma once

#include "audio.hpp"
#include "spectrum.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

struct FitnessReference {
    std::size_t frameCount = 0;
    std::size_t binCount = 0;
    std::size_t fftSize = 0;
    std::size_t hopSize = 0;
    std::uint32_t sampleRateHz = 0;
    double magnitudeFloor = 1e-12;
    double dynamicRangeDb = 80.0;
    std::vector<double> logMagnitudes;
};

struct HotspotSettings {
    std::size_t suppressionFrames = 2;
    std::size_t suppressionBins = 2;
    std::size_t maximumHotspots = 16;
    double minimumResidualMagnitude = 0.0;
};

struct SpectralResidual {
    std::size_t frameCount = 0;
    std::size_t binCount = 0;
    std::size_t fftSize = 0;
    std::size_t hopSize = 0;
    std::uint32_t sampleRateHz = 0;
    std::vector<double> values;
};

struct ResidualHotspot {
    std::size_t frameIndex = 0;
    std::size_t binIndex = 0;
    double residual = 0.0;
    double magnitude = 0.0;
    double timeSeconds = 0.0;
    double frequencyHz = 0.0;
};

struct FitnessState {
    std::vector<double> frameSquaredErrors;
    double totalSquaredError = 0.0;
    double fitness = 0.0;
};

double magnitudeToLog(double magnitude);
double magnitudeToLog(double magnitude, double magnitudeFloor);

// A completely silent target uses an absolute 1e-12 magnitude floor.
FitnessReference createFitnessReference(
    const Spectrogram& target, double dynamicRangeDb = 80.0);
double calculateFrameSquaredError(std::span<const double> candidateMagnitudes,
    const FitnessReference& target,
    std::size_t frameIndex);
FitnessState createFitnessState(
    const Spectrogram& candidate, const FitnessReference& target);

SpectralResidual calculateSpectralResidual(
    const Spectrogram& champion, const FitnessReference& target);

std::vector<ResidualHotspot> findResidualHotspots(
    const SpectralResidual& residual, const HotspotSettings& settings = {});

// Fitness is an error metric: lower values are better, and zero is ideal.
double calculateSpectralFitness(
    const Spectrogram& candidate, const FitnessReference& target);

double calculateSpectralFitness(
    const Spectrogram& candidate, const Spectrogram& target);

double calculateAudioFitness(
    const AudioBuffer& candidate, const FitnessReference& target);

double calculateAudioFitness(const AudioBuffer& candidate,
    const FitnessReference& target,
    StftAnalyzer& analyzer);
