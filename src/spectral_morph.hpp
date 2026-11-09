#pragma once

#include "audio.hpp"
#include "spectrum.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

enum class MorphStrategy {
    Interpolate,
    PolarInterpolate,
    MagnitudeSourcePhase,
    MagnitudeTargetPhase,
    Replace,
};

std::string_view morphStrategyValue(MorphStrategy strategy) noexcept;

enum class ReplacementOrder {
    Error,
    TargetEnergy,
    SourceEnergy,
    LowToHigh,
    HighToLow,
    EarlyToLate,
    LateToEarly,
    Random,
};

std::string_view replacementOrderValue(ReplacementOrder order) noexcept;

struct CellIndex {
    std::size_t frameIndex = 0;
    std::size_t binIndex = 0;

    bool operator==(const CellIndex&) const = default;
};

struct SpectralMorphSettings {
    MorphStrategy strategy = MorphStrategy::Interpolate;
    std::size_t morphSteps = 10;
    double interpolationDynamicRangeDb = 80.0;
    ReplacementOrder replacementOrder = ReplacementOrder::Error;
    double replacementDynamicRangeDb = 80.0;
    std::uint64_t replacementSeed = 0;
};

struct MorphGeneration {
    std::size_t generation = 0;
    double progress = 0.0;
    MorphStrategy strategy = MorphStrategy::Interpolate;
    ReplacementOrder replacementOrder = ReplacementOrder::Error;
    std::size_t cellsReplaced = 0;
    std::size_t totalCells = 0;
};

using MorphGenerationObserver = std::function<void(
    const MorphGeneration& generation, const AudioBuffer& audio)>;

ComplexSpectrogram interpolateComplexSpectrogram(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress);

ComplexSpectrogram polarInterpolateComplexSpectrogram(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress,
    double dynamicRangeDb = 80.0);

ComplexSpectrogram interpolateMagnitudeWithSourcePhase(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress,
    double dynamicRangeDb = 80.0);

ComplexSpectrogram interpolateMagnitudeWithTargetPhase(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress,
    double dynamicRangeDb = 80.0);

std::vector<CellIndex> createErrorReplacementOrder(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double dynamicRangeDb = 80.0);

std::vector<CellIndex> createReplacementOrder(const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    ReplacementOrder order,
    std::uint64_t randomSeed = 0,
    double dynamicRangeDb = 80.0);

std::size_t calculateReplacementCount(
    std::size_t totalCellCount, double progress);

ComplexSpectrogram replaceComplexSpectrogramCells(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    std::span<const CellIndex> replacementOrder,
    std::size_t replacementCount);

void runSpectralMorph(const AudioBuffer& source,
    const AudioBuffer& target,
    const SpectralMorphSettings& settings,
    ComplexStftAnalyzer& analyzer,
    ComplexStftSynthesizer& synthesizer,
    const MorphGenerationObserver& observer);
