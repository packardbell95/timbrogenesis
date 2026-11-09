#include "spectral_morph.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Function>
void requireInvalidArgument(Function function, const std::string& message)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

AudioBuffer createTone(double frequencyHz, double amplitude = 0.7)
{
    constexpr std::uint32_t sampleRateHz = 8'000;
    AudioBuffer audio;
    audio.sampleRateHz = sampleRateHz;
    audio.samples.resize(513);
    for (std::size_t index = 0; index < audio.samples.size(); ++index) {
        audio.samples[index] = amplitude
            * std::sin(2.0 * std::numbers::pi * frequencyHz
                * static_cast<double>(index)
                / static_cast<double>(sampleRateHz));
    }
    return audio;
}

void testCoefficientInterpolation()
{
    ComplexSpectrogram source{
        .frameCount = 1,
        .binCount = 2,
        .fftSize = 2,
        .hopSize = 1,
        .sampleRateHz = 8'000,
        .originalSampleCount = 1,
        .coefficients = {{2.0, 4.0}, {-6.0, 8.0}},
    };
    ComplexSpectrogram target = source;
    target.coefficients = {{10.0, -4.0}, {2.0, 0.0}};

    require(interpolateComplexSpectrogram(source, target, 0.0).coefficients
            == source.coefficients,
        "Alpha zero did not preserve source coefficients.");
    require(interpolateComplexSpectrogram(source, target, 1.0).coefficients
            == target.coefficients,
        "Alpha one did not preserve target coefficients.");
    const ComplexSpectrogram midpoint =
        interpolateComplexSpectrogram(source, target, 0.5);
    require(midpoint.coefficients[0] == std::complex<double>(6.0, 0.0)
            && midpoint.coefficients[1] == std::complex<double>(-2.0, 4.0),
        "Alpha one-half did not produce coefficient midpoints.");
}

std::vector<AudioBuffer> collectMorph(const AudioBuffer& source,
    const AudioBuffer& target,
    std::size_t steps,
    std::vector<MorphGeneration>& summaries,
    MorphStrategy strategy = MorphStrategy::Interpolate,
    ReplacementOrder replacementOrder = ReplacementOrder::Error,
    std::uint64_t replacementSeed = 0)
{
    ComplexStftAnalyzer analyzer(64, 16);
    ComplexStftSynthesizer synthesizer(64);
    std::vector<AudioBuffer> outputs;
    runSpectralMorph(source,
        target,
        {
            .strategy = strategy,
            .morphSteps = steps,
            .interpolationDynamicRangeDb = 80.0,
            .replacementOrder = replacementOrder,
            .replacementDynamicRangeDb = 80.0,
            .replacementSeed = replacementSeed,
        },
        analyzer,
        synthesizer,
        [&](const MorphGeneration& summary, const AudioBuffer& audio) {
        summaries.push_back(summary);
        outputs.push_back(audio);
    });
    return outputs;
}

ComplexSpectrogram createCoefficientSpectrum(
    const std::vector<std::complex<double>>& coefficients)
{
    return {
        .frameCount = 1,
        .binCount = coefficients.size(),
        .fftSize = coefficients.size() == 1 ? 1 : 2 * (coefficients.size() - 1),
        .hopSize = 1,
        .sampleRateHz = 8'000,
        .originalSampleCount = 1,
        .coefficients = coefficients,
    };
}

void testMagnitudeAndPhaseInterpolation()
{
    const ComplexSpectrogram source = createCoefficientSpectrum({
        std::polar(2.0, 0.5),
    });
    const ComplexSpectrogram target = createCoefficientSpectrum({
        std::polar(8.0, 1.5),
    });
    const std::array<ComplexSpectrogram, 3> midpoints{
        polarInterpolateComplexSpectrogram(source, target, 0.5),
        interpolateMagnitudeWithSourcePhase(source, target, 0.5),
        interpolateMagnitudeWithTargetPhase(source, target, 0.5),
    };
    for (const ComplexSpectrogram& midpoint : midpoints)
        require(std::abs(std::abs(midpoint.coefficients[0]) - 4.0) < 1e-12,
            "Log-magnitude interpolation did not produce a geometric "
            "midpoint.");
    require(std::abs(std::arg(midpoints[0].coefficients[0]) - 1.0) < 1e-12,
        "Polar interpolation produced the wrong midpoint phase.");
    require(std::abs(std::arg(midpoints[1].coefficients[0]) - 0.5) < 1e-12,
        "Source-phase interpolation did not retain source phase.");
    require(std::abs(std::arg(midpoints[2].coefficients[0]) - 1.5) < 1e-12,
        "Target-phase interpolation did not retain target phase.");

    require(polarInterpolateComplexSpectrogram(source, target, 0.0).coefficients
                == source.coefficients
            && polarInterpolateComplexSpectrogram(source, target, 1.0)
                    .coefficients
                == target.coefficients,
        "Polar interpolation did not preserve exact spectral endpoints.");
    require(
        interpolateMagnitudeWithSourcePhase(source, target, 0.0).coefficients
                == source.coefficients
            && interpolateMagnitudeWithSourcePhase(source, target, 1.0)
                    .coefficients
                == target.coefficients,
        "Source-phase interpolation did not preserve exact spectral "
        "endpoints.");
    require(
        interpolateMagnitudeWithTargetPhase(source, target, 0.0).coefficients
                == source.coefficients
            && interpolateMagnitudeWithTargetPhase(source, target, 1.0)
                    .coefficients
                == target.coefficients,
        "Target-phase interpolation did not preserve exact spectral "
        "endpoints.");
}

void testShortestPhasePathAndZeroMagnitudes()
{
    const double degreesToRadians = std::numbers::pi / 180.0;
    const ComplexSpectrogram source = createCoefficientSpectrum({
        std::polar(1.0, 179.0 * degreesToRadians),
        {0.0, 0.0},
    });
    const ComplexSpectrogram target = createCoefficientSpectrum({
        std::polar(1.0, -179.0 * degreesToRadians),
        {0.0, 0.0},
    });
    const ComplexSpectrogram midpoint =
        polarInterpolateComplexSpectrogram(source, target, 0.5);
    require(std::abs(
                std::abs(std::arg(midpoint.coefficients[0])) - std::numbers::pi)
            < 1e-12,
        "Polar phase interpolation did not cross the shortest wrap path.");

    const std::array<ComplexSpectrogram, 3> zeroResults{
        midpoint,
        interpolateMagnitudeWithSourcePhase(source, target, 0.5),
        interpolateMagnitudeWithTargetPhase(source, target, 0.5),
    };
    for (const ComplexSpectrogram& result : zeroResults) {
        for (const std::complex<double>& coefficient : result.coefficients)
            require(std::isfinite(coefficient.real())
                    && std::isfinite(coefficient.imag()),
                "Zero magnitude produced a non-finite coefficient.");
    }
}

ComplexSpectrogram createOrderingSpectrum(const std::vector<double>& magnitudes)
{
    ComplexSpectrogram spectrum{
        .frameCount = 2,
        .binCount = 3,
        .fftSize = 4,
        .hopSize = 1,
        .sampleRateHz = 8'000,
        .originalSampleCount = 2,
        .coefficients = {},
    };
    for (double magnitude : magnitudes)
        spectrum.coefficients.emplace_back(magnitude, 0.0);
    return spectrum;
}

void testErrorReplacementOrderAndTies()
{
    const ComplexSpectrogram source =
        createOrderingSpectrum({1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    const ComplexSpectrogram target =
        createOrderingSpectrum({1.0, 10.0, 100.0, 10.0, 2.0, 1.0});
    const std::vector<CellIndex> order =
        createErrorReplacementOrder(source, target);
    const std::vector<CellIndex> expected{
        {0, 2},
        {0, 1},
        {1, 0},
        {1, 1},
        {0, 0},
        {1, 2},
    };
    require(order == expected,
        "Error replacement order or deterministic tie ordering is incorrect.");
}

void testReplacementOrderingPolicies()
{
    const ComplexSpectrogram source =
        createOrderingSpectrum({1.0, 100.0, 2.0, 10.0, 3.0, 0.5});
    const ComplexSpectrogram target =
        createOrderingSpectrum({1.0, 10.0, 100.0, 10.0, 2.0, 1.0});
    struct ExpectedEndpoints {
        ReplacementOrder order;
        CellIndex first;
        CellIndex last;
    };
    const std::array<ExpectedEndpoints, 7> expected{{
        {ReplacementOrder::Error, {0, 2}, {1, 0}},
        {ReplacementOrder::TargetEnergy, {0, 2}, {1, 2}},
        {ReplacementOrder::SourceEnergy, {0, 1}, {1, 2}},
        {ReplacementOrder::LowToHigh, {0, 0}, {1, 2}},
        {ReplacementOrder::HighToLow, {0, 2}, {1, 0}},
        {ReplacementOrder::EarlyToLate, {0, 0}, {1, 2}},
        {ReplacementOrder::LateToEarly, {1, 0}, {0, 2}},
    }};
    for (const ExpectedEndpoints& item : expected) {
        const std::vector<CellIndex> first =
            createReplacementOrder(source, target, item.order, 123);
        const std::vector<CellIndex> second =
            createReplacementOrder(source, target, item.order, 123);
        require(first == second,
            "A deterministic replacement ordering changed between runs.");
        require(first.front() == item.first && first.back() == item.last,
            "A replacement ordering has incorrect endpoint cells.");
        const ComplexSpectrogram fullyReplaced =
            replaceComplexSpectrogramCells(source, target, first, first.size());
        require(fullyReplaced.coefficients == target.coefficients,
            "A replacement ordering did not support exact full replacement.");
    }
}

void testRandomReplacementOrdering()
{
    const ComplexSpectrogram source =
        createOrderingSpectrum({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const ComplexSpectrogram target =
        createOrderingSpectrum({6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    const std::vector<CellIndex> first =
        createReplacementOrder(source, target, ReplacementOrder::Random, 12345);
    const std::vector<CellIndex> repeated =
        createReplacementOrder(source, target, ReplacementOrder::Random, 12345);
    const std::vector<CellIndex> different =
        createReplacementOrder(source, target, ReplacementOrder::Random, 54321);
    require(first == repeated,
        "The same random seed produced a different replacement order.");
    require(first != different,
        "Controlled different seeds produced the same replacement order.");
    const ComplexSpectrogram fullyReplaced =
        replaceComplexSpectrogramCells(source, target, first, first.size());
    require(fullyReplaced.coefficients == target.coefficients,
        "Random ordering did not support exact full replacement.");
}

void testReplacementCountsAndMasks()
{
    const ComplexSpectrogram source =
        createOrderingSpectrum({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const ComplexSpectrogram target =
        createOrderingSpectrum({11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    const std::vector<CellIndex> order{
        {1, 2},
        {0, 1},
        {1, 0},
        {0, 2},
        {0, 0},
        {1, 1},
    };

    require(calculateReplacementCount(7, 0.0) == 0
            && calculateReplacementCount(7, 0.5) == 3
            && calculateReplacementCount(7, 1.0) == 7,
        "Replacement count does not use floor(alpha * totalCells).");
    const ComplexSpectrogram zero =
        replaceComplexSpectrogramCells(source, target, order, 0);
    const ComplexSpectrogram first =
        replaceComplexSpectrogramCells(source, target, order, 1);
    const ComplexSpectrogram second =
        replaceComplexSpectrogramCells(source, target, order, 2);
    const ComplexSpectrogram full =
        replaceComplexSpectrogramCells(source, target, order, order.size());
    require(zero.coefficients == source.coefficients,
        "Zero replacement progress changed source cells.");
    require(full.coefficients == target.coefficients,
        "Full replacement progress did not copy every target cell.");
    require(first.coefficients[5] == target.coefficients[5]
            && second.coefficients[5] == target.coefficients[5]
            && second.coefficients[1] == target.coefficients[1],
        "Replacement masks are not cumulative prefixes.");
    require(second.coefficients[0] == source.coefficients[0]
            && second.coefficients[2] == source.coefficients[2],
        "Replacement copied cells outside the requested prefix.");
}

void testStepCountEndpointsAndProgress()
{
    const AudioBuffer source = createTone(440.0);
    const AudioBuffer target = createTone(880.0, 0.5);
    std::vector<MorphGeneration> summaries;
    const std::vector<AudioBuffer> outputs =
        collectMorph(source, target, 10, summaries);
    require(outputs.size() == 11 && summaries.size() == 11,
        "Ten morph intervals did not produce eleven generations.");
    require(outputs.front().samples == source.samples,
        "The first morph generation is not the exact source.");
    require(outputs.back().samples == target.samples,
        "The final morph generation is not the exact target.");
    for (std::size_t step = 0; step <= 10; ++step) {
        require(summaries[step].generation == step,
            "Morph generation index is incorrect.");
        require(std::abs(
                    summaries[step].progress - static_cast<double>(step) / 10.0)
                < 1e-15,
            "Morph progress is not an absolute step position.");
        for (double sample : outputs[step].samples)
            require(std::isfinite(sample),
                "Morph output contains a non-finite sample.");
    }
}

void testDeterminismAndValidation()
{
    const AudioBuffer source = createTone(330.0);
    const AudioBuffer target = createTone(1'110.0);
    std::vector<MorphGeneration> firstSummaries;
    std::vector<MorphGeneration> secondSummaries;
    const std::vector<AudioBuffer> first =
        collectMorph(source, target, 4, firstSummaries);
    const std::vector<AudioBuffer> second =
        collectMorph(source, target, 4, secondSummaries);
    require(
        first.size() == second.size(), "Deterministic runs differ in size.");
    for (std::size_t index = 0; index < first.size(); ++index)
        require(first[index].samples == second[index].samples,
            "Repeated spectral morph runs differ.");

    ComplexStftAnalyzer analyzer(64, 16);
    ComplexStftSynthesizer synthesizer(64);
    requireInvalidArgument([&]() {
        runSpectralMorph(source,
            target,
            {.morphSteps = 0},
            analyzer,
            synthesizer,
            [](const MorphGeneration&, const AudioBuffer&) {});
    }, "A zero morph step count was accepted.");
}

void testReplacementMorphOutputs()
{
    const AudioBuffer source = createTone(275.0);
    const AudioBuffer target = createTone(1'375.0, 0.45);
    std::vector<MorphGeneration> summaries;
    const std::vector<AudioBuffer> outputs =
        collectMorph(source, target, 5, summaries, MorphStrategy::Replace);
    require(outputs.size() == 6 && summaries.size() == 6,
        "Replacement morph produced the wrong generation count.");
    require(outputs.front().samples == source.samples
            && outputs.back().samples == target.samples,
        "Replacement morph did not preserve exact endpoints.");
    require(summaries.front().cellsReplaced == 0
            && summaries.back().cellsReplaced == summaries.back().totalCells
            && summaries.back().totalCells > 0,
        "Replacement endpoint metadata has incorrect cell counts.");
    std::size_t previousCount = 0;
    for (std::size_t step = 0; step < outputs.size(); ++step) {
        require(summaries[step].strategy == MorphStrategy::Replace,
            "Replacement generation reported the wrong strategy.");
        require(summaries[step].cellsReplaced >= previousCount,
            "Replacement generation counts are not monotonic.");
        const std::size_t expected = calculateReplacementCount(
            summaries[step].totalCells, summaries[step].progress);
        require(summaries[step].cellsReplaced == expected,
            "Replacement generation metadata has an incorrect count.");
        previousCount = summaries[step].cellsReplaced;
        for (double sample : outputs[step].samples)
            require(std::isfinite(sample),
                "Replacement morph output contains a non-finite sample.");
    }

    std::vector<MorphGeneration> repeatedSummaries;
    const std::vector<AudioBuffer> repeated = collectMorph(
        source, target, 5, repeatedSummaries, MorphStrategy::Replace);
    for (std::size_t index = 0; index < outputs.size(); ++index)
        require(outputs[index].samples == repeated[index].samples,
            "Replacement morph is not deterministic.");
}

void testEveryReplacementOrderingProducesDeterministicOutput()
{
    const AudioBuffer source = createTone(320.0);
    const AudioBuffer target = createTone(1'280.0, 0.4);
    const std::array<ReplacementOrder, 8> orderings{
        ReplacementOrder::Error,
        ReplacementOrder::TargetEnergy,
        ReplacementOrder::SourceEnergy,
        ReplacementOrder::LowToHigh,
        ReplacementOrder::HighToLow,
        ReplacementOrder::EarlyToLate,
        ReplacementOrder::LateToEarly,
        ReplacementOrder::Random,
    };
    for (ReplacementOrder order : orderings) {
        std::vector<MorphGeneration> firstSummaries;
        std::vector<MorphGeneration> secondSummaries;
        const std::vector<AudioBuffer> first = collectMorph(source,
            target,
            3,
            firstSummaries,
            MorphStrategy::Replace,
            order,
            9876);
        const std::vector<AudioBuffer> second = collectMorph(source,
            target,
            3,
            secondSummaries,
            MorphStrategy::Replace,
            order,
            9876);
        require(first.size() == 4 && second.size() == 4,
            "Replacement ordering produced the wrong output count.");
        for (std::size_t index = 0; index < first.size(); ++index) {
            require(first[index].samples == second[index].samples,
                "Replacement ordering produced nondeterministic audio.");
            require(firstSummaries[index].replacementOrder == order,
                "Replacement generation reported the wrong ordering.");
        }
        require(first.front().samples == source.samples
                && first.back().samples == target.samples,
            "Replacement ordering changed an exact endpoint.");
    }
}

void testMagnitudePhaseStrategiesProduceDeterministicAudio()
{
    const AudioBuffer source = createTone(410.0);
    const AudioBuffer target = createTone(970.0, 0.35);
    const std::array<MorphStrategy, 3> strategies{
        MorphStrategy::PolarInterpolate,
        MorphStrategy::MagnitudeSourcePhase,
        MorphStrategy::MagnitudeTargetPhase,
    };
    for (MorphStrategy strategy : strategies) {
        std::vector<MorphGeneration> firstSummaries;
        std::vector<MorphGeneration> secondSummaries;
        const std::vector<AudioBuffer> first =
            collectMorph(source, target, 4, firstSummaries, strategy);
        const std::vector<AudioBuffer> second =
            collectMorph(source, target, 4, secondSummaries, strategy);
        require(first.front().samples == source.samples
                && first.back().samples == target.samples,
            "Magnitude/phase morph changed an exact audio endpoint.");
        for (std::size_t generation = 0; generation < first.size();
             ++generation) {
            require(first[generation].samples == second[generation].samples,
                "Magnitude/phase morph produced nondeterministic audio.");
            for (double sample : first[generation].samples)
                require(std::isfinite(sample),
                    "Magnitude/phase morph produced non-finite audio.");
        }
    }
}

}  // namespace

int main()
{
    try {
        testCoefficientInterpolation();
        testMagnitudeAndPhaseInterpolation();
        testShortestPhasePathAndZeroMagnitudes();
        testErrorReplacementOrderAndTies();
        testReplacementOrderingPolicies();
        testRandomReplacementOrdering();
        testReplacementCountsAndMasks();
        testStepCountEndpointsAndProgress();
        testDeterminismAndValidation();
        testReplacementMorphOutputs();
        testEveryReplacementOrderingProducesDeterministicOutput();
        testMagnitudePhaseStrategiesProduceDeterministicAudio();
        std::cout << "All spectral morph tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Spectral morph test failure: " << error.what() << '\n';
        return 1;
    }
}
