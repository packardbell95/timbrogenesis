#include "spectral_morph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

namespace {

void validateCompatibleSpectrograms(
    const ComplexSpectrogram& source, const ComplexSpectrogram& target)
{
    const auto hasValidStorage = [](const ComplexSpectrogram& spectrum) {
        if (spectrum.frameCount != 0
            && spectrum.binCount
                > std::numeric_limits<std::size_t>::max() / spectrum.frameCount)
            return false;
        return spectrum.coefficients.size()
            == spectrum.frameCount * spectrum.binCount;
    };
    if (source.frameCount != target.frameCount
        || source.binCount != target.binCount
        || source.fftSize != target.fftSize || source.hopSize != target.hopSize
        || source.sampleRateHz != target.sampleRateHz
        || source.originalSampleCount != target.originalSampleCount
        || source.coefficients.size() != target.coefficients.size()
        || !hasValidStorage(source) || !hasValidStorage(target)) {
        throw std::invalid_argument(
            "Source and target complex spectrograms are incompatible.");
    }
}

void validateProgress(double progress)
{
    if (!std::isfinite(progress) || progress < 0.0 || progress > 1.0)
        throw std::invalid_argument(
            "Morph progress must be between zero and one.");
}

double calculateMagnitudeFloor(
    const ComplexSpectrogram& target, double dynamicRangeDb)
{
    if (!std::isfinite(dynamicRangeDb) || dynamicRangeDb <= 0.0)
        throw std::invalid_argument(
            "Morph dynamic range must be finite and positive.");
    constexpr double absoluteMagnitudeFloor = 1e-12;
    double targetMaximumMagnitude = 0.0;
    for (const std::complex<double>& coefficient : target.coefficients) {
        const double magnitude = std::abs(coefficient);
        if (!std::isfinite(magnitude))
            throw std::invalid_argument("Target complex spectrogram contains a "
                                        "non-finite coefficient.");
        targetMaximumMagnitude = std::max(targetMaximumMagnitude, magnitude);
    }
    const double relativeFloor =
        targetMaximumMagnitude * std::pow(10.0, -dynamicRangeDb / 20.0);
    return targetMaximumMagnitude == 0.0
        ? absoluteMagnitudeFloor
        : std::max(relativeFloor, absoluteMagnitudeFloor);
}

enum class PhaseInterpolation {
    ShortestPath,
    Source,
    Target,
};

ComplexSpectrogram interpolateMagnitudeAndPhase(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress,
    double dynamicRangeDb,
    PhaseInterpolation phaseInterpolation)
{
    validateCompatibleSpectrograms(source, target);
    validateProgress(progress);
    const double magnitudeFloor =
        calculateMagnitudeFloor(target, dynamicRangeDb);
    for (const std::complex<double>& coefficient : source.coefficients) {
        if (!std::isfinite(std::abs(coefficient)))
            throw std::invalid_argument("Source complex spectrogram contains a "
                                        "non-finite coefficient.");
    }
    if (progress == 0.0)
        return source;
    if (progress == 1.0)
        return target;
    ComplexSpectrogram result = source;
    for (std::size_t index = 0; index < result.coefficients.size(); ++index) {
        const double sourceMagnitude = std::abs(source.coefficients[index]);
        const double targetMagnitude = std::abs(target.coefficients[index]);
        const double logMagnitude = (1.0 - progress)
                * std::log(std::max(sourceMagnitude, magnitudeFloor))
            + progress * std::log(std::max(targetMagnitude, magnitudeFloor));
        const double magnitude = std::exp(logMagnitude);
        const double sourcePhase = std::arg(source.coefficients[index]);
        const double targetPhase = std::arg(target.coefficients[index]);
        double phase = sourcePhase;
        if (phaseInterpolation == PhaseInterpolation::ShortestPath) {
            const double delta = std::remainder(
                targetPhase - sourcePhase, 2.0 * std::numbers::pi);
            phase += progress * delta;
        } else if (phaseInterpolation == PhaseInterpolation::Target) {
            phase = targetPhase;
        }
        result.coefficients[index] = std::polar(magnitude, phase);
        if (!std::isfinite(result.coefficients[index].real())
            || !std::isfinite(result.coefficients[index].imag()))
            throw std::runtime_error("Magnitude/phase interpolation produced a "
                                     "non-finite coefficient.");
    }
    return result;
}

}  // namespace

std::string_view morphStrategyValue(MorphStrategy strategy) noexcept
{
    switch (strategy) {
    case MorphStrategy::Interpolate:
        return "interpolate";
    case MorphStrategy::PolarInterpolate:
        return "polar-interpolate";
    case MorphStrategy::MagnitudeSourcePhase:
        return "magnitude-source-phase";
    case MorphStrategy::MagnitudeTargetPhase:
        return "magnitude-target-phase";
    case MorphStrategy::Replace:
        return "replace";
    }
    return "unknown";
}

std::string_view replacementOrderValue(ReplacementOrder order) noexcept
{
    switch (order) {
    case ReplacementOrder::Error:
        return "error";
    case ReplacementOrder::TargetEnergy:
        return "target-energy";
    case ReplacementOrder::SourceEnergy:
        return "source-energy";
    case ReplacementOrder::LowToHigh:
        return "low-to-high";
    case ReplacementOrder::HighToLow:
        return "high-to-low";
    case ReplacementOrder::EarlyToLate:
        return "early-to-late";
    case ReplacementOrder::LateToEarly:
        return "late-to-early";
    case ReplacementOrder::Random:
        return "random";
    }
    return "unknown";
}

ComplexSpectrogram interpolateComplexSpectrogram(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress)
{
    validateCompatibleSpectrograms(source, target);
    validateProgress(progress);

    ComplexSpectrogram result = source;
    const double sourceWeight = 1.0 - progress;
    for (std::size_t index = 0; index < result.coefficients.size(); ++index) {
        result.coefficients[index] = sourceWeight * source.coefficients[index]
            + progress * target.coefficients[index];
    }
    return result;
}

ComplexSpectrogram polarInterpolateComplexSpectrogram(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress,
    double dynamicRangeDb)
{
    return interpolateMagnitudeAndPhase(source,
        target,
        progress,
        dynamicRangeDb,
        PhaseInterpolation::ShortestPath);
}

ComplexSpectrogram interpolateMagnitudeWithSourcePhase(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress,
    double dynamicRangeDb)
{
    return interpolateMagnitudeAndPhase(
        source, target, progress, dynamicRangeDb, PhaseInterpolation::Source);
}

ComplexSpectrogram interpolateMagnitudeWithTargetPhase(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double progress,
    double dynamicRangeDb)
{
    return interpolateMagnitudeAndPhase(
        source, target, progress, dynamicRangeDb, PhaseInterpolation::Target);
}

std::vector<CellIndex> createErrorReplacementOrder(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    double dynamicRangeDb)
{
    validateCompatibleSpectrograms(source, target);
    if (!std::isfinite(dynamicRangeDb) || dynamicRangeDb <= 0.0)
        throw std::invalid_argument(
            "Replacement dynamic range must be finite and positive.");

    const double magnitudeFloor =
        calculateMagnitudeFloor(target, dynamicRangeDb);

    struct RankedCell {
        CellIndex cell;
        double error = 0.0;
    };
    std::vector<RankedCell> ranked;
    ranked.reserve(source.coefficients.size());
    for (std::size_t index = 0; index < source.coefficients.size(); ++index) {
        const double sourceMagnitude = std::abs(source.coefficients[index]);
        const double targetMagnitude = std::abs(target.coefficients[index]);
        if (!std::isfinite(sourceMagnitude))
            throw std::invalid_argument("Source complex spectrogram contains a "
                                        "non-finite coefficient.");
        ranked.push_back({
            .cell =
                {
                    .frameIndex = index / source.binCount,
                    .binIndex = index % source.binCount,
                },
            .error =
                std::abs(std::log(std::max(targetMagnitude, magnitudeFloor))
                    - std::log(std::max(sourceMagnitude, magnitudeFloor))),
        });
    }
    std::stable_sort(ranked.begin(),
        ranked.end(),
        [](const RankedCell& left, const RankedCell& right) {
        return left.error > right.error;
    });
    std::vector<CellIndex> order;
    order.reserve(ranked.size());
    for (const RankedCell& cell : ranked)
        order.push_back(cell.cell);
    return order;
}

std::vector<CellIndex> createReplacementOrder(const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    ReplacementOrder order,
    std::uint64_t randomSeed,
    double dynamicRangeDb)
{
    validateCompatibleSpectrograms(source, target);
    if (order == ReplacementOrder::Error)
        return createErrorReplacementOrder(source, target, dynamicRangeDb);

    std::vector<CellIndex> cells;
    cells.reserve(source.coefficients.size());
    for (std::size_t frame = 0; frame < source.frameCount; ++frame)
        for (std::size_t bin = 0; bin < source.binCount; ++bin)
            cells.push_back({frame, bin});

    const auto coefficientIndex = [&source](const CellIndex& cell) {
        return cell.frameIndex * source.binCount + cell.binIndex;
    };
    const auto sortByMagnitude =
        [&cells, &coefficientIndex](
            const std::vector<std::complex<double>>& coefficients) {
        for (const std::complex<double>& coefficient : coefficients) {
            if (!std::isfinite(std::abs(coefficient)))
                throw std::invalid_argument("Energy replacement order received "
                                            "a non-finite coefficient.");
        }
        std::stable_sort(
            cells.begin(), cells.end(), [&](CellIndex left, CellIndex right) {
            return std::abs(coefficients[coefficientIndex(left)])
                > std::abs(coefficients[coefficientIndex(right)]);
        });
    };

    switch (order) {
    case ReplacementOrder::TargetEnergy:
        sortByMagnitude(target.coefficients);
        break;
    case ReplacementOrder::SourceEnergy:
        sortByMagnitude(source.coefficients);
        break;
    case ReplacementOrder::LowToHigh:
        std::stable_sort(
            cells.begin(), cells.end(), [](CellIndex left, CellIndex right) {
            if (left.binIndex != right.binIndex)
                return left.binIndex < right.binIndex;
            return left.frameIndex < right.frameIndex;
        });
        break;
    case ReplacementOrder::HighToLow:
        std::stable_sort(
            cells.begin(), cells.end(), [](CellIndex left, CellIndex right) {
            if (left.binIndex != right.binIndex)
                return left.binIndex > right.binIndex;
            return left.frameIndex < right.frameIndex;
        });
        break;
    case ReplacementOrder::EarlyToLate:
        break;
    case ReplacementOrder::LateToEarly:
        std::stable_sort(
            cells.begin(), cells.end(), [](CellIndex left, CellIndex right) {
            if (left.frameIndex != right.frameIndex)
                return left.frameIndex > right.frameIndex;
            return left.binIndex < right.binIndex;
        });
        break;
    case ReplacementOrder::Random: {
        std::mt19937_64 generator(randomSeed);
        std::shuffle(cells.begin(), cells.end(), generator);
        break;
    }
    case ReplacementOrder::Error:
        break;
    default:
        throw std::invalid_argument("Unknown replacement ordering policy.");
    }
    return cells;
}

std::size_t calculateReplacementCount(
    std::size_t totalCellCount, double progress)
{
    validateProgress(progress);
    if (progress == 1.0)
        return totalCellCount;
    const double count =
        std::floor(progress * static_cast<double>(totalCellCount));
    return std::min(totalCellCount, static_cast<std::size_t>(count));
}

ComplexSpectrogram replaceComplexSpectrogramCells(
    const ComplexSpectrogram& source,
    const ComplexSpectrogram& target,
    std::span<const CellIndex> replacementOrder,
    std::size_t replacementCount)
{
    validateCompatibleSpectrograms(source, target);
    const std::size_t totalCells = source.coefficients.size();
    if (replacementOrder.size() != totalCells)
        throw std::invalid_argument(
            "Replacement order does not cover every cell.");
    if (replacementCount > totalCells)
        throw std::invalid_argument(
            "Replacement cell count exceeds total cells.");

    std::vector<bool> visited(totalCells, false);
    for (const CellIndex& cell : replacementOrder) {
        if (cell.frameIndex >= source.frameCount
            || cell.binIndex >= source.binCount)
            throw std::invalid_argument(
                "Replacement order contains an invalid cell.");
        const std::size_t index =
            cell.frameIndex * source.binCount + cell.binIndex;
        if (visited[index])
            throw std::invalid_argument(
                "Replacement order contains a duplicate cell.");
        visited[index] = true;
    }

    ComplexSpectrogram result = source;
    for (std::size_t index = 0; index < replacementCount; ++index) {
        const CellIndex cell = replacementOrder[index];
        const std::size_t coefficientIndex =
            cell.frameIndex * source.binCount + cell.binIndex;
        result.coefficients[coefficientIndex] =
            target.coefficients[coefficientIndex];
    }
    return result;
}

void runSpectralMorph(const AudioBuffer& source,
    const AudioBuffer& target,
    const SpectralMorphSettings& settings,
    ComplexStftAnalyzer& analyzer,
    ComplexStftSynthesizer& synthesizer,
    const MorphGenerationObserver& observer)
{
    if (settings.morphSteps == 0
        || settings.morphSteps == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Morph step count is invalid.");
    if (settings.strategy != MorphStrategy::Interpolate
        && settings.strategy != MorphStrategy::PolarInterpolate
        && settings.strategy != MorphStrategy::MagnitudeSourcePhase
        && settings.strategy != MorphStrategy::MagnitudeTargetPhase
        && settings.strategy != MorphStrategy::Replace)
        throw std::invalid_argument("Unknown spectral morph strategy.");
    if (settings.strategy == MorphStrategy::Replace
        && settings.replacementOrder == ReplacementOrder::Error
        && (!std::isfinite(settings.replacementDynamicRangeDb)
            || settings.replacementDynamicRangeDb <= 0.0))
        throw std::invalid_argument("Replacement dynamic range is invalid.");
    const bool magnitudePhaseStrategy =
        settings.strategy == MorphStrategy::PolarInterpolate
        || settings.strategy == MorphStrategy::MagnitudeSourcePhase
        || settings.strategy == MorphStrategy::MagnitudeTargetPhase;
    if (magnitudePhaseStrategy
        && (!std::isfinite(settings.interpolationDynamicRangeDb)
            || settings.interpolationDynamicRangeDb <= 0.0))
        throw std::invalid_argument("Interpolation dynamic range is invalid.");
    if (source.sampleRateHz == 0 || source.sampleRateHz != target.sampleRateHz
        || source.samples.size() != target.samples.size()) {
        throw std::invalid_argument("Source and target audio must have "
                                    "matching sample rates and lengths.");
    }
    if (analyzer.fftSize() != synthesizer.fftSize())
        throw std::invalid_argument(
            "Morph analyzer and synthesizer FFT sizes differ.");
    if (!observer)
        throw std::invalid_argument(
            "Spectral morph requires a generation observer.");

    const ComplexSpectrogram sourceSpectrum = analyzer.calculate(source);
    const ComplexSpectrogram targetSpectrum = analyzer.calculate(target);
    validateCompatibleSpectrograms(sourceSpectrum, targetSpectrum);
    const std::vector<CellIndex> replacementOrder =
        settings.strategy == MorphStrategy::Replace
        ? createReplacementOrder(sourceSpectrum,
              targetSpectrum,
              settings.replacementOrder,
              settings.replacementSeed,
              settings.replacementDynamicRangeDb)
        : std::vector<CellIndex>{};
    const std::size_t totalCells = sourceSpectrum.coefficients.size();

    observer(
        {
            .generation = 0,
            .progress = 0.0,
            .strategy = settings.strategy,
            .replacementOrder = settings.replacementOrder,
            .cellsReplaced = 0,
            .totalCells =
                settings.strategy == MorphStrategy::Replace ? totalCells : 0,
        },
        source);
    for (std::size_t step = 1; step < settings.morphSteps; ++step) {
        const double progress = static_cast<double>(step)
            / static_cast<double>(settings.morphSteps);
        const std::size_t cellsReplaced =
            settings.strategy == MorphStrategy::Replace
            ? calculateReplacementCount(totalCells, progress)
            : 0;
        ComplexSpectrogram intermediate;
        switch (settings.strategy) {
        case MorphStrategy::Interpolate:
            intermediate = interpolateComplexSpectrogram(
                sourceSpectrum, targetSpectrum, progress);
            break;
        case MorphStrategy::PolarInterpolate:
            intermediate = polarInterpolateComplexSpectrogram(sourceSpectrum,
                targetSpectrum,
                progress,
                settings.interpolationDynamicRangeDb);
            break;
        case MorphStrategy::MagnitudeSourcePhase:
            intermediate = interpolateMagnitudeWithSourcePhase(sourceSpectrum,
                targetSpectrum,
                progress,
                settings.interpolationDynamicRangeDb);
            break;
        case MorphStrategy::MagnitudeTargetPhase:
            intermediate = interpolateMagnitudeWithTargetPhase(sourceSpectrum,
                targetSpectrum,
                progress,
                settings.interpolationDynamicRangeDb);
            break;
        case MorphStrategy::Replace:
            intermediate = replaceComplexSpectrogramCells(sourceSpectrum,
                targetSpectrum,
                replacementOrder,
                cellsReplaced);
            break;
        }
        observer(
            {
                .generation = step,
                .progress = progress,
                .strategy = settings.strategy,
                .replacementOrder = settings.replacementOrder,
                .cellsReplaced = cellsReplaced,
                .totalCells = settings.strategy == MorphStrategy::Replace
                    ? totalCells
                    : 0,
            },
            synthesizer.reconstruct(intermediate));
    }
    observer(
        {
            .generation = settings.morphSteps,
            .progress = 1.0,
            .strategy = settings.strategy,
            .replacementOrder = settings.replacementOrder,
            .cellsReplaced =
                settings.strategy == MorphStrategy::Replace ? totalCells : 0,
            .totalCells =
                settings.strategy == MorphStrategy::Replace ? totalCells : 0,
        },
        target);
}
