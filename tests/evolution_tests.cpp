#include "evolution.hpp"
#include "fitness.hpp"
#include "metadata.hpp"
#include "mutation.hpp"
#include "spectrum.hpp"
#include "synthesis.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double tolerance = 1e-12;
constexpr std::uint32_t sampleRateHz = 1024;
constexpr std::size_t fftSize = 128;
constexpr std::size_t hopSize = 64;

class TemporaryDirectory {
public:
    TemporaryDirectory()
    {
        const auto uniqueValue =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
            / ("timbrogenesis-evolution-tests-" + std::to_string(uniqueValue));
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
void requireThrows(Function function, const std::string& testName)
{
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(testName + " did not throw.");
}

bool atomsEqual(const SineAtom& left, const SineAtom& right)
{
    return left.startSeconds == right.startSeconds
        && left.durationSeconds == right.durationSeconds
        && left.frequencyHz == right.frequencyHz
        && left.amplitude == right.amplitude
        && left.phaseRadians == right.phaseRadians;
}

AudioBuffer createSilence(
    std::size_t sampleCount = sampleRateHz, std::uint32_t rateHz = sampleRateHz)
{
    return AudioBuffer{std::vector<double>(sampleCount, 0.0), rateHz};
}

SineAtom targetAtom()
{
    return SineAtom{
        .startSeconds = 0.2,
        .durationSeconds = 0.4,
        .frequencyHz = 128.0,
        .amplitude = 0.5,
        .phaseRadians = 0.5,
    };
}

struct ControlledFixture {
    AudioBuffer source;
    AudioBuffer target;
    FitnessReference reference;
    double sourceFitness;
};

SineMutationSettings createFocusedMutationSettings();

ControlledFixture createControlledFixture()
{
    AudioBuffer source = createSilence();
    AudioBuffer target = source;
    applySineAtom(target, targetAtom());
    FitnessReference reference =
        createFitnessReference(calculateSpectrogram(target, fftSize, hopSize));
    const double sourceFitness = calculateAudioFitness(source, reference);
    return ControlledFixture{
        .source = std::move(source),
        .target = std::move(target),
        .reference = std::move(reference),
        .sourceFitness = sourceFitness,
    };
}

void testSameSeedProducesSameAtoms()
{
    const AudioBuffer audio = createSilence();
    const SineMutationSettings settings;
    std::mt19937_64 firstGenerator(123456789);
    std::mt19937_64 secondGenerator(123456789);

    for (std::size_t index = 0; index < 20; ++index) {
        const SineAtom first =
            createRandomSineAtom(audio, settings, firstGenerator);
        const SineAtom second =
            createRandomSineAtom(audio, settings, secondGenerator);
        require(atomsEqual(first, second),
            "The same seed produced different atom sequences.");
    }
}

void testDifferentSeedsProduceDifferentSequences()
{
    const AudioBuffer audio = createSilence();
    const SineMutationSettings settings;
    std::mt19937_64 firstGenerator(1);
    std::mt19937_64 secondGenerator(2);
    bool foundDifference = false;

    for (std::size_t index = 0; index < 5; ++index) {
        foundDifference = foundDifference
            || !atomsEqual(
                createRandomSineAtom(audio, settings, firstGenerator),
                createRandomSineAtom(audio, settings, secondGenerator));
    }
    require(foundDifference,
        "Different seeds produced identical controlled atom sequences.");
}

void testMutationBoundsAndAmplitudeSigns()
{
    const AudioBuffer audio = createSilence();
    const SineMutationSettings settings;
    std::mt19937_64 generator(8675309);
    const double audioDurationSeconds = 1.0;
    const double effectiveMaximumFrequencyHz = 512.0;
    bool foundPositiveAmplitude = false;
    bool foundNegativeAmplitude = false;

    for (std::size_t index = 0; index < 1000; ++index) {
        const SineAtom atom = createRandomSineAtom(audio, settings, generator);
        require(std::isfinite(atom.frequencyHz)
                && atom.frequencyHz >= settings.minimumFrequencyHz
                && atom.frequencyHz <= effectiveMaximumFrequencyHz,
            "A random frequency violated its effective bounds.");
        require(atom.startSeconds >= 0.0
                && atom.startSeconds < audioDurationSeconds,
            "A random start time was outside the recording.");
        require(atom.durationSeconds > 0.0
                && atom.durationSeconds <= settings.maximumDurationSeconds,
            "A random duration violated its bounds.");
        require(atom.startSeconds + atom.durationSeconds
                <= audioDurationSeconds + tolerance,
            "A random atom extended beyond the recording.");
        require(std::abs(atom.amplitude) <= settings.maximumAbsoluteAmplitude,
            "A random amplitude violated its bounds.");
        require(atom.phaseRadians >= 0.0
                && atom.phaseRadians <= 2.0 * std::numbers::pi,
            "A random phase violated its bounds.");
        foundPositiveAmplitude = foundPositiveAmplitude || atom.amplitude > 0.0;
        foundNegativeAmplitude = foundNegativeAmplitude || atom.amplitude < 0.0;
    }

    require(foundPositiveAmplitude && foundNegativeAmplitude,
        "Random mutations did not cover both amplitude signs.");
}

void testShortAudioAndNyquist()
{
    const AudioBuffer shortAudio = createSilence(1, 48'000);
    const SineMutationSettings settings;
    std::mt19937_64 generator(42);
    const SineAtom shortAtom =
        createRandomSineAtom(shortAudio, settings, generator);
    const double shortDurationSeconds = 1.0 / 48'000.0;
    require(shortAtom.durationSeconds > 0.0
            && shortAtom.startSeconds + shortAtom.durationSeconds
                <= shortDurationSeconds + tolerance,
        "Mutation generation did not adapt to extremely short audio.");

    const AudioBuffer lowRateAudio = createSilence(100, 100);
    for (std::size_t index = 0; index < 100; ++index) {
        const SineAtom atom =
            createRandomSineAtom(lowRateAudio, settings, generator);
        require(
            atom.frequencyHz <= 50.0, "A random mutation exceeded Nyquist.");
    }
}

void testMutationValidation()
{
    const AudioBuffer audio = createSilence();
    std::mt19937_64 generator(1);
    SineMutationSettings settings;

    requireThrows([&settings, &generator]() {
        createRandomSineAtom(
            AudioBuffer{{}, sampleRateHz}, settings, generator);
    }, "Empty audio mutation");

    settings.minimumFrequencyHz = 0.0;
    requireThrows([&audio, &settings, &generator]() {
        createRandomSineAtom(audio, settings, generator);
    }, "Zero logarithmic frequency");

    settings = SineMutationSettings{};
    settings.maximumFrequencyHz = 10.0;
    requireThrows([&audio, &settings, &generator]() {
        createRandomSineAtom(audio, settings, generator);
    }, "Reversed frequency bounds");

    settings = SineMutationSettings{};
    settings.maximumDurationSeconds = 0.001;
    requireThrows([&audio, &settings, &generator]() {
        createRandomSineAtom(audio, settings, generator);
    }, "Reversed duration bounds");

    settings = SineMutationSettings{};
    settings.maximumAbsoluteAmplitude = -0.1;
    requireThrows([&audio, &settings, &generator]() {
        createRandomSineAtom(audio, settings, generator);
    }, "Negative maximum amplitude");

    settings = SineMutationSettings{};
    const AudioBuffer veryLowRate = createSilence(10, 20);
    requireThrows([&veryLowRate, &settings, &generator]() {
        createRandomSineAtom(veryLowRate, settings, generator);
    }, "Minimum frequency above Nyquist");
}

void testControlledImprovementAndBestSelection()
{
    const ControlledFixture fixture = createControlledFixture();
    SineAtom partialMutation = targetAtom();
    partialMutation.amplitude *= 0.5;
    SineAtom noOpMutation = targetAtom();
    noOpMutation.amplitude = 0.0;
    const std::vector<SineAtom> mutations{
        noOpMutation,
        partialMutation,
        targetAtom(),
    };

    const GenerationResult result = evaluateMutations(
        fixture.source, fixture.sourceFitness, fixture.reference, mutations);
    require(result.improved, "A known improving mutation was not accepted.");
    require(result.winningMutations.size() == 1
            && atomsEqual(result.winningMutations.front(), targetAtom()),
        "The best supplied mutation was not selected.");
    require(result.fitness < result.previousFitness,
        "An accepted generation did not improve fitness.");
    require(result.fitness < tolerance,
        "The exact target mutation did not approach zero fitness.");
    require(result.audio.samples == fixture.target.samples,
        "Candidates were not independently derived from the parent.");
    require(result.candidatesTested == mutations.size(),
        "The generation reported the wrong candidate count.");
}

void testRangesAndMultipleWinnerSelection()
{
    const AudioBuffer source = createSilence();
    const SineAtom early{0.05, 0.10, 96.0, 0.4, 0.3};
    const SineAtom late{0.75, 0.10, 192.0, 0.35, 0.7};
    require(sineAtomSampleRange(source, early).beginSample == 51,
        "Sample range does not use synthesis rounding.");
    const FrameRange earlyFrames =
        affectedFrameRange(source, early, fftSize, hopSize);
    const FrameRange lateFrames =
        affectedFrameRange(source, late, fftSize, hopSize);
    require(!frameRangesOverlap(earlyFrames, lateFrames),
        "Separated atoms were reported as conflicting.");
    require(!frameRangesOverlap({0, 2}, {2, 4})
            && frameRangesOverlap({0, 3}, {2, 4})
            && frameRangesOverlap({1, 5}, {2, 3})
            && frameRangesOverlap({1, 3}, {1, 3}),
        "Half-open frame conflict rules are incorrect.");

    AudioBuffer target = source;
    applySineAtom(target, early);
    applySineAtom(target, late);
    const FitnessReference reference =
        createFitnessReference(calculateSpectrogram(target, fftSize, hopSize));
    const double initial = calculateAudioFitness(source, reference);
    const GenerationResult both =
        evaluateMutations(source, initial, reference, {early, late}, 8);
    require(both.winningMutations.size() == 2
            && both.audio.samples == target.samples,
        "Frame-disjoint improvements were not accepted together.");
    const GenerationResult capped =
        evaluateMutations(source, initial, reference, {early, late}, 1);
    require(capped.winningMutations.size() == 1,
        "Maximum accepted mutation cap was not enforced.");

    SineAtom weaker = early;
    weaker.amplitude *= 0.5;
    AudioBuffer earlyTarget = source;
    applySineAtom(earlyTarget, early);
    const FitnessReference earlyReference = createFitnessReference(
        calculateSpectrogram(earlyTarget, fftSize, hopSize));
    const GenerationResult conflict = evaluateMutations(source,
        calculateAudioFitness(source, earlyReference),
        earlyReference,
        {weaker, early},
        8);
    require(conflict.winningMutations.size() == 1
            && atomsEqual(conflict.winningMutations.front(), early),
        "Conflicting improvements did not retain the strongest candidate.");
}

void testEditableGenomeOperations()
{
    const AudioBuffer source = createSilence();
    const SineAtom first{0.1, 0.2, 100.0, 0.2, 0.1};
    const SineAtom second{0.6, 0.2, 180.0, -0.15, 0.5};
    Genome genome;
    AudioBuffer cached = source;
    applyGenomeMutation(genome, cached, AddAtomMutation{{11, first}});
    applyGenomeMutation(genome, cached, AddAtomMutation{{29, second}});
    applyGenomeMutation(
        genome, cached, ChangeFrequencyMutation{11, first.frequencyHz, 110.0});
    applyGenomeMutation(
        genome, cached, ChangeAmplitudeMutation{29, second.amplitude, -0.1});
    applyGenomeMutation(
        genome, cached, MoveAtomMutation{11, first.startSeconds, 0.2});
    applyGenomeMutation(genome,
        cached,
        ChangeDurationMutation{29, second.durationSeconds, 0.15});
    applyGenomeMutation(
        genome, cached, ChangePhaseMutation{11, first.phaseRadians, 0.9});
    require(findGenomeAtom(genome, 11) && findGenomeAtom(genome, 29),
        "Stable atom IDs were lost during edits.");
    const AudioBuffer rendered = renderGenome(source, genome);
    for (std::size_t index = 0; index < cached.samples.size(); ++index)
        require(
            std::abs(cached.samples[index] - rendered.samples[index]) < 1e-12,
            "Incremental genome edits differ from a full render.");
    applyGenomeMutation(genome, cached, RemoveAtomMutation{11});
    require(!findGenomeAtom(genome, 11) && findGenomeAtom(genome, 29),
        "Removing one stable ID damaged genome identity.");
    const AudioBuffer afterRemoval = renderGenome(source, genome);
    for (std::size_t index = 0; index < cached.samples.size(); ++index)
        require(std::abs(cached.samples[index] - afterRemoval.samples[index])
                < 1e-12,
            "Incremental removal differs from a full render.");
}

void testIncrementalGenomeEvaluationMatchesFullAnalysis()
{
    const AudioBuffer source = createSilence();
    SineAtom exact = targetAtom();
    AudioBuffer target = source;
    applySineAtom(target, exact);
    StftAnalyzer analyzer(fftSize, hopSize);
    const FitnessReference reference =
        createFitnessReference(analyzer.calculate(target));
    EvolutionState state = createEvolutionState(source, reference, analyzer);

    SineAtom partial = exact;
    partial.amplitude *= 0.5;
    const GenomeGenerationResult added = evaluateGenomeMutations(
        state, reference, {AddAtomMutation{{55, partial}}}, 8, analyzer);
    require(
        added.winningMutations.size() == 1 && state.genome.atoms.size() == 1,
        "A controlled genome addition was not committed.");
    require(
        std::abs(state.fitness.fitness
            - calculateAudioFitness(state.renderedAudio, reference, analyzer))
            < tolerance,
        "Incremental addition fitness differs from full analysis.");

    const ChangeAmplitudeMutation weaker{
        55, partial.amplitude, exact.amplitude * 0.75};
    const ChangeAmplitudeMutation strongest{
        55, partial.amplitude, exact.amplitude};
    const GenomeGenerationResult edited = evaluateGenomeMutations(
        state, reference, {weaker, strongest}, 8, analyzer);
    require(edited.winningMutations.size() == 1
            && std::holds_alternative<ChangeAmplitudeMutation>(
                edited.winningMutations.front()),
        "Conflicting edits to one atom were both accepted.");
    require(state.fitness.fitness < tolerance,
        "The strongest exact edit did not reach the target.");
    const Spectrogram fullSpectrum = analyzer.calculate(state.renderedAudio);
    require(fullSpectrum.magnitudes == state.spectrogram.magnitudes,
        "Committed local spectral frames differ from full analysis.");
}

void testGuidedProposalLocationAndDeterminism()
{
    const ControlledFixture fixture = createControlledFixture();
    const Spectrogram sourceSpectrum =
        calculateSpectrogram(fixture.source, fftSize, hopSize);
    const SpectralResidual residual =
        calculateSpectralResidual(sourceSpectrum, fixture.reference);
    HotspotSettings hotspotSettings;
    hotspotSettings.maximumHotspots = 4;
    const auto hotspots = findResidualHotspots(residual, hotspotSettings);
    require(
        !hotspots.empty(), "Known missing tone produced no guidance hotspot.");
    SineMutationSettings sineSettings = createFocusedMutationSettings();
    GuidedSearchSettings searchSettings;
    std::mt19937_64 firstGenerator(9001);
    std::mt19937_64 secondGenerator(9001);
    const SineAtom first = createGuidedSineAtom(fixture.source,
        residual,
        hotspots.front(),
        sineSettings,
        searchSettings,
        firstGenerator);
    const SineAtom second = createGuidedSineAtom(fixture.source,
        residual,
        hotspots.front(),
        sineSettings,
        searchSettings,
        secondGenerator);
    require(atomsEqual(first, second),
        "Guided atom generation is not deterministic.");
    const double binSpacing =
        static_cast<double>(sampleRateHz) / static_cast<double>(fftSize);
    require(std::abs(first.frequencyHz - hotspots.front().frequencyHz)
            <= searchSettings.guidedFrequencyRadiusBins * binSpacing
                + tolerance,
        "Guided atom frequency is outside its hotspot neighborhood.");
    require(first.startSeconds >= 0.0 && first.durationSeconds > 0.0
            && first.startSeconds + first.durationSeconds <= 1.0 + tolerance,
        "Guided atom violated synthesis time bounds.");
}

void testErrorWeightedAtomSelectionScores()
{
    const AudioBuffer audio = createSilence();
    const Genome genome{{
        GenomeAtom{1, SineAtom{0.05, 0.1, 100.0, 0.1, 0.0}},
        GenomeAtom{2, SineAtom{0.75, 0.1, 100.0, 0.1, 0.0}},
    }};
    const std::size_t frames = 1 + (audio.samples.size() - 1) / hopSize;
    FitnessState fitness;
    fitness.frameSquaredErrors.assign(frames, 0.01);
    const FrameRange highError =
        affectedFrameRange(audio, genome.atoms.front().atom, fftSize, hopSize);
    for (std::size_t frame = highError.beginFrame; frame < highError.endFrame;
         ++frame)
        fitness.frameSquaredErrors[frame] = 100.0;
    const auto weights = calculateAtomSelectionWeights(
        audio, genome, fitness, fftSize, hopSize, 0.05);
    require(weights.size() == 2 && weights[0] > weights[1] && weights[1] > 0.0,
        "Atom relevance did not prioritize error while preserving "
        "exploration.");
}

void testGuidedGenomeEditsFollowResidual()
{
    const AudioBuffer audio = createSilence();
    const SineAtom atom{0.05, 0.1, 104.0, 0.1, 0.2};
    const Genome genome{{GenomeAtom{7, atom}}};
    const std::size_t frames = 1 + (audio.samples.size() - 1) / hopSize;
    SpectralResidual residual{.frameCount = frames,
        .binCount = fftSize / 2 + 1,
        .fftSize = fftSize,
        .hopSize = hopSize,
        .sampleRateHz = sampleRateHz,
        .values = std::vector<double>(frames * (fftSize / 2 + 1), 0.0)};
    const FrameRange atomRange =
        affectedFrameRange(audio, atom, fftSize, hopSize);
    const std::size_t desiredBin = 15;
    residual.values[atomRange.beginFrame * residual.binCount + desiredBin] =
        9.0;
    residual.values[12 * residual.binCount
        + frequencyToNearestBin(atom.frequencyHz, fftSize, sampleRateHz)] = 8.0;
    ResidualHotspot hotspot{atomRange.beginFrame,
        desiredBin,
        9.0,
        9.0,
        0.2,
        binToFrequencyHz(desiredBin, fftSize, sampleRateHz)};
    FitnessState fitness;
    fitness.frameSquaredErrors.assign(frames, 1.0);
    SineMutationSettings sineSettings = createFocusedMutationSettings();
    GuidedSearchSettings search;
    std::mt19937_64 generator(11);

    GenomeMutationSettings frequencyOnly;
    frequencyOnly.addWeight = 0.0;
    frequencyOnly.removeWeight = 0.0;
    frequencyOnly.moveWeight = 0.0;
    frequencyOnly.frequencyWeight = 1.0;
    frequencyOnly.amplitudeWeight = 0.0;
    frequencyOnly.durationWeight = 0.0;
    frequencyOnly.phaseWeight = 0.0;
    const GenomeMutation frequencyMutation = createGuidedGenomeMutation(audio,
        genome,
        99,
        sineSettings,
        frequencyOnly,
        search,
        residual,
        hotspot,
        fitness,
        generator);
    const double desiredFrequency =
        binToFrequencyHz(desiredBin, fftSize, sampleRateHz);
    require(std::abs(
                std::get<ChangeFrequencyMutation>(frequencyMutation).frequencyHz
                - desiredFrequency)
            <= 4.0,
        "Guided frequency edit did not follow nearby residual energy.");

    GenomeMutationSettings moveOnly = frequencyOnly;
    moveOnly.frequencyWeight = 0.0;
    moveOnly.moveWeight = 1.0;
    const GenomeMutation moveMutation = createGuidedGenomeMutation(audio,
        genome,
        100,
        sineSettings,
        moveOnly,
        search,
        residual,
        hotspot,
        fitness,
        generator);
    require(std::get<MoveAtomMutation>(moveMutation).startSeconds > 0.6,
        "Guided move edit did not follow temporal residual energy.");
}

void testSearchBudgetAndGuidanceStatistics()
{
    const ControlledFixture fixture = createControlledFixture();
    EvolutionSettings settings;
    settings.candidatesPerGeneration = 25;
    settings.maximumGenerations = 2;
    GuidedSearchSettings search;
    search.guidedMutationRatio = 0.8;
    search.refinementBudgetRatio = 0.4;
    search.refinementParentCount = 4;
    StftAnalyzer analyzer(fftSize, hopSize);
    std::mt19937_64 generator(1234);
    const EvolutionRunResult run = evolve(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        settings,
        createFocusedMutationSettings(),
        GenomeMutationSettings{},
        search,
        generator,
        analyzer);
    for (std::size_t index = 1; index < run.history.size(); ++index) {
        const GenerationSummary& summary = run.history[index];
        require(summary.candidatesTested <= settings.candidatesPerGeneration
                && summary.candidatesTested
                    == summary.guidedCandidatesTested
                        + summary.randomCandidatesTested
                        + summary.refinementCandidatesTested,
            "Exploration and refinement exceeded or misreported the budget.");
        require(summary.guidedCandidatesTested > 0
                && summary.randomCandidatesTested > 0,
            "The configured hybrid search lost guided or random exploration.");
    }
    require(run.guidedCandidatesTested > 0 && run.randomCandidatesTested > 0,
        "Run-level search statistics were not accumulated.");

    settings.maximumGenerations = 1;
    settings.candidatesPerGeneration = 12;
    search.guidedMutationRatio = 1.0;
    search.refinementBudgetRatio = 0.0;
    StftAnalyzer allGuidedAnalyzer(fftSize, hopSize);
    std::mt19937_64 allGuidedGenerator(1234);
    const EvolutionRunResult allGuided = evolve(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        settings,
        createFocusedMutationSettings(),
        GenomeMutationSettings{},
        search,
        allGuidedGenerator,
        allGuidedAnalyzer);
    require(allGuided.history[1].guidedCandidatesTested == 12
            && allGuided.history[1].randomCandidatesTested == 0,
        "A guided ratio of one did not guide all eligible proposals.");
}

void testBoundedAmplitudeAndPhaseSearch()
{
    const AudioBuffer source = createSilence();
    SineAtom exact = targetAtom();
    exact.phaseRadians = std::numbers::pi / 2.0;
    AudioBuffer target = source;
    applySineAtom(target, exact);
    StftAnalyzer analyzer(fftSize, hopSize);
    const FitnessReference reference =
        createFitnessReference(analyzer.calculate(target));
    EvolutionState state = createEvolutionState(source, reference, analyzer);
    AddAtomMutation parent{{91, exact}};
    parent.atom.atom.phaseRadians = 0.17;
    std::vector<GenomeMutation> phaseVariants;
    std::mt19937_64 generator(77);
    SineMutationSettings sineSettings = createFocusedMutationSettings();
    GuidedSearchSettings search;
    for (std::size_t variant = 0; variant < 4; ++variant)
        phaseVariants.push_back(refineGenomeMutation(source,
            state.genome,
            parent,
            variant,
            fftSize,
            hopSize,
            sineSettings,
            search,
            generator));
    const GenomeGenerationResult phaseResult =
        evaluateGenomeMutations(state, reference, phaseVariants, 8, analyzer);
    require(phaseResult.winningMutations.size() == 1
            && std::get<AddAtomMutation>(phaseResult.winningMutations.front())
                    .atom.atom.phaseRadians
                == std::numbers::pi / 2.0,
        "Phase grid did not select the known target phase.");

    AudioBuffer amplitudeTarget = source;
    exact.amplitude = sineSettings.maximumAbsoluteAmplitude * 0.5;
    applySineAtom(amplitudeTarget, exact);
    const FitnessReference amplitudeReference =
        createFitnessReference(analyzer.calculate(amplitudeTarget));
    EvolutionState amplitudeState =
        createEvolutionState(source, amplitudeReference, analyzer);
    parent.atom.atom = exact;
    parent.atom.atom.amplitude = 0.03;
    std::vector<GenomeMutation> amplitudeVariants;
    for (std::size_t variant = 4; variant < 12; ++variant)
        amplitudeVariants.push_back(refineGenomeMutation(source,
            amplitudeState.genome,
            parent,
            variant,
            fftSize,
            hopSize,
            sineSettings,
            search,
            generator));
    const GenomeGenerationResult amplitudeResult = evaluateGenomeMutations(
        amplitudeState, amplitudeReference, amplitudeVariants, 8, analyzer);
    require(amplitudeResult.winningMutations.size() == 1
            && std::abs(std::get<AddAtomMutation>(
                   amplitudeResult.winningMutations.front())
                            .atom.atom.amplitude)
                == std::abs(exact.amplitude),
        "Amplitude grid did not select the known target magnitude.");

    AudioBuffer excessSource = source;
    SineAtom excess = targetAtom();
    applySineAtom(excessSource, excess);
    const AudioBuffer silenceTarget = source;
    const FitnessReference silenceReference =
        createFitnessReference(analyzer.calculate(silenceTarget));
    EvolutionState cancellationState =
        createEvolutionState(excessSource, silenceReference, analyzer);
    AddAtomMutation cancellationParent{{123, excess}};
    cancellationParent.atom.atom.amplitude = 0.01;
    std::vector<GenomeMutation> cancellationVariants;
    for (std::size_t variant = 4; variant < 12; ++variant)
        cancellationVariants.push_back(refineGenomeMutation(excessSource,
            cancellationState.genome,
            cancellationParent,
            variant,
            fftSize,
            hopSize,
            sineSettings,
            search,
            generator));
    const GenomeGenerationResult cancellation = evaluateGenomeMutations(
        cancellationState, silenceReference, cancellationVariants, 8, analyzer);
    require(cancellation.winningMutations.size() == 1
            && std::get<AddAtomMutation>(cancellation.winningMutations.front())
                    .atom.atom.amplitude
                == -excess.amplitude
            && cancellationState.fitness.fitness < tolerance,
        "Bounded amplitude search did not find destructive cancellation.");
}

void testNoImprovementAndTiePreserveParent()
{
    const ControlledFixture fixture = createControlledFixture();
    const double targetFitness =
        calculateAudioFitness(fixture.target, fixture.reference);
    SineAtom noOpMutation = targetAtom();
    noOpMutation.amplitude = 0.0;

    const GenerationResult result = evaluateMutations(fixture.target,
        targetFitness,
        fixture.reference,
        {noOpMutation, targetAtom()});
    require(!result.improved, "A tie or worsening mutation was accepted.");
    require(result.winningMutations.empty(),
        "A no-improvement generation reported a winner.");
    require(result.fitness == targetFitness,
        "A no-improvement generation changed fitness.");
    require(result.audio.samples == fixture.target.samples,
        "A no-improvement generation changed the parent waveform.");
}

void testOneAndZeroCandidates()
{
    const ControlledFixture fixture = createControlledFixture();
    SineMutationSettings mutationSettings;
    mutationSettings.minimumFrequencyHz = 128.0;
    mutationSettings.maximumFrequencyHz = 128.0;
    mutationSettings.minimumDurationSeconds = 0.2;
    mutationSettings.maximumDurationSeconds = 0.2;
    mutationSettings.maximumAbsoluteAmplitude = 0.0;
    std::mt19937_64 generator(5);

    EvolutionSettings oneCandidateSettings;
    oneCandidateSettings.candidatesPerGeneration = 1;
    const GenerationResult result = evolveOneGeneration(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        oneCandidateSettings,
        mutationSettings,
        generator);
    require(result.candidatesTested == 1,
        "A one-candidate generation tested the wrong count.");
    require(result.fitness <= fixture.sourceFitness,
        "Elitism failed for a one-candidate generation.");

    requireThrows([&fixture, &mutationSettings, &generator]() {
        EvolutionSettings zeroCandidateSettings;
        zeroCandidateSettings.candidatesPerGeneration = 0;
        evolveOneGeneration(fixture.source,
            fixture.sourceFitness,
            fixture.reference,
            zeroCandidateSettings,
            mutationSettings,
            generator);
    }, "Zero candidates");
}

void testGenerationReproducibilityAndElitism()
{
    const ControlledFixture fixture = createControlledFixture();
    EvolutionSettings evolutionSettings;
    evolutionSettings.candidatesPerGeneration = 20;
    SineMutationSettings mutationSettings;
    mutationSettings.minimumFrequencyHz = 100.0;
    mutationSettings.maximumFrequencyHz = 160.0;
    mutationSettings.minimumDurationSeconds = 0.2;
    mutationSettings.maximumDurationSeconds = 0.5;
    mutationSettings.maximumAbsoluteAmplitude = 0.5;
    std::mt19937_64 firstGenerator(20260911);
    std::mt19937_64 secondGenerator(20260911);

    const GenerationResult first = evolveOneGeneration(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        evolutionSettings,
        mutationSettings,
        firstGenerator);
    const GenerationResult second = evolveOneGeneration(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        evolutionSettings,
        mutationSettings,
        secondGenerator);

    require(first.fitness <= fixture.sourceFitness,
        "A generation violated elitism.");
    require(first.fitness == second.fitness
            && first.audio.samples == second.audio.samples
            && first.winningMutations.size() == second.winningMutations.size(),
        "The same seed produced different generation results.");
    for (std::size_t index = 0; index < first.winningMutations.size();
         ++index) {
        require(atomsEqual(first.winningMutations[index],
                    second.winningMutations[index]),
            "The same seed selected different winning mutations.");
    }
}

EvolutionSettings createFullRunSettings()
{
    EvolutionSettings settings;
    settings.candidatesPerGeneration = 30;
    settings.maximumGenerations = 4;
    return settings;
}

SineMutationSettings createFocusedMutationSettings()
{
    SineMutationSettings settings;
    settings.minimumFrequencyHz = 100.0;
    settings.maximumFrequencyHz = 160.0;
    settings.minimumDurationSeconds = 0.3;
    settings.maximumDurationSeconds = 0.5;
    settings.maximumAbsoluteAmplitude = 0.5;
    return settings;
}

void requireMatchingRunResults(
    const EvolutionRunResult& first, const EvolutionRunResult& second)
{
    require(first.finalAudio.samples == second.finalAudio.samples
            && first.initialFitness == second.initialFitness
            && first.finalFitness == second.finalFitness
            && first.completedGenerations == second.completedGenerations
            && first.successfulWrites == second.successfulWrites
            && first.stopReason == second.stopReason
            && first.finalGenome.atoms == second.finalGenome.atoms
            && first.guidedCandidatesTested == second.guidedCandidatesTested
            && first.randomCandidatesTested == second.randomCandidatesTested
            && first.refinementCandidatesTested
                == second.refinementCandidatesTested
            && first.history.size() == second.history.size(),
        "Same-seed full runs produced different overall results.");

    for (std::size_t index = 0; index < first.history.size(); ++index) {
        const GenerationSummary& left = first.history[index];
        const GenerationSummary& right = second.history[index];
        require(left.generation == right.generation
                && left.previousFitness == right.previousFitness
                && left.fitness == right.fitness
                && left.improvement == right.improvement
                && left.improved == right.improved
                && left.candidatesTested == right.candidatesTested
                && left.guidedCandidatesTested == right.guidedCandidatesTested
                && left.randomCandidatesTested == right.randomCandidatesTested
                && left.refinementCandidatesTested
                    == right.refinementCandidatesTested
                && left.winningMutations == right.winningMutations,
            "Same-seed full runs produced different histories.");
    }
}

void testOneGenerationFullRunCompatibility()
{
    const ControlledFixture fixture = createControlledFixture();
    EvolutionSettings settings = createFullRunSettings();
    settings.maximumGenerations = 1;
    const SineMutationSettings mutationSettings =
        createFocusedMutationSettings();
    std::mt19937_64 directGenerator(4567);
    std::mt19937_64 runGenerator(4567);

    const GenerationResult direct = evolveOneGeneration(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        settings,
        mutationSettings,
        directGenerator);
    GuidedSearchSettings randomOnlySearch;
    randomOnlySearch.guidedMutationRatio = 0.0;
    randomOnlySearch.refinementBudgetRatio = 0.0;
    StftAnalyzer analyzer(fftSize, hopSize);
    const EvolutionRunResult run = evolve(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        settings,
        mutationSettings,
        GenomeMutationSettings{},
        randomOnlySearch,
        runGenerator,
        analyzer);

    require(run.completedGenerations == 1 && run.history.size() == 2
            && run.stopReason == EvolutionStopReason::MaximumGenerations,
        "A one-generation full run has incorrect completion state.");
    require(run.finalFitness == direct.fitness
            && run.finalAudio.samples == direct.audio.samples,
        "A one-generation full run differs from evolveOneGeneration().");
}

void testFullRunDeterminismCarryForwardAndHistory()
{
    const ControlledFixture fixture = createControlledFixture();
    const EvolutionSettings settings = createFullRunSettings();
    const SineMutationSettings mutationSettings =
        createFocusedMutationSettings();
    std::mt19937_64 firstGenerator(20260911);
    std::mt19937_64 secondGenerator(20260911);

    const EvolutionRunResult first = evolve(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        settings,
        mutationSettings,
        firstGenerator);
    const EvolutionRunResult second = evolve(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        settings,
        mutationSettings,
        secondGenerator);
    requireMatchingRunResults(first, second);

    require(first.completedGenerations == settings.maximumGenerations
            && first.history.size() == settings.maximumGenerations + 1
            && first.stopReason == EvolutionStopReason::MaximumGenerations,
        "A maximum-generation run stopped incorrectly.");

    AudioBuffer reconstructed = fixture.source;
    Genome reconstructedGenome;
    std::size_t improvementCount = 0;
    for (std::size_t index = 1; index < first.history.size(); ++index) {
        const GenerationSummary& previous = first.history[index - 1];
        const GenerationSummary& current = first.history[index];
        require(current.generation == index,
            "Generation history numbering is incorrect.");
        require(current.previousFitness.has_value()
                && *current.previousFitness == previous.fitness,
            "Generation fitness history is not chained.");
        require(current.fitness <= previous.fitness
                && current.improvement == previous.fitness - current.fitness,
            "Generation fitness history is not monotonic or consistent.");
        require(current.candidatesTested <= settings.candidatesPerGeneration
                && current.candidatesTested
                    == current.guidedCandidatesTested
                        + current.randomCandidatesTested
                        + current.refinementCandidatesTested,
            "A generation recorded the wrong candidate count.");
        for (const GenomeMutation& mutation : current.winningMutations) {
            applyGenomeMutation(reconstructedGenome, reconstructed, mutation);
            ++improvementCount;
        }
    }
    require(improvementCount >= 2,
        "The carry-forward test did not produce enough improvements.");
    require(reconstructed.samples == first.finalAudio.samples,
        "The final champion does not contain all accepted mutations.");
    const AudioBuffer fullRender =
        renderGenome(fixture.source, first.finalGenome);
    for (std::size_t index = 0; index < fullRender.samples.size(); ++index)
        require(std::abs(
                    fullRender.samples[index] - first.finalAudio.samples[index])
                < 1e-12,
            "Final cached audio differs from render-from-genome output.");
}

void testStoppingConditions()
{
    const ControlledFixture fixture = createControlledFixture();

    EvolutionSettings immediateSettings;
    immediateSettings.candidatesPerGeneration = 2;
    immediateSettings.maximumGenerations = 5;
    immediateSettings.targetFitness = 0.0;
    std::mt19937_64 immediateGenerator(1);
    std::size_t observedGenerations = 0;
    const EvolutionRunResult immediate = evolve(fixture.target,
        0.0,
        fixture.reference,
        immediateSettings,
        createFocusedMutationSettings(),
        immediateGenerator,
        [&observedGenerations](const GenerationSummary&, const AudioBuffer&) {
        ++observedGenerations;
    });
    require(immediate.completedGenerations == 0
            && immediate.successfulWrites == 1 && immediate.history.size() == 1
            && observedGenerations == 1
            && immediate.stopReason
                == EvolutionStopReason::TargetFitnessReached,
        "A generation-zero target threshold did not stop immediately.");

    EvolutionSettings stagnationSettings;
    stagnationSettings.candidatesPerGeneration = 2;
    stagnationSettings.maximumGenerations = 10;
    stagnationSettings.stagnationGenerations = 3;
    SineMutationSettings noOpSettings = createFocusedMutationSettings();
    noOpSettings.maximumAbsoluteAmplitude = 0.0;
    std::mt19937_64 stagnationGenerator(2);
    const EvolutionRunResult stagnant = evolve(fixture.target,
        0.0,
        fixture.reference,
        stagnationSettings,
        noOpSettings,
        stagnationGenerator);
    require(stagnant.completedGenerations == 3
            && stagnant.stopReason == EvolutionStopReason::Stagnation
            && stagnant.finalAudio.samples == fixture.target.samples,
        "Consecutive flat generations did not trigger stagnation.");

    std::size_t stagnationCount = 0;
    stagnationCount = updateStagnationCount(stagnationCount, false);
    stagnationCount = updateStagnationCount(stagnationCount, false);
    stagnationCount = updateStagnationCount(stagnationCount, true);
    stagnationCount = updateStagnationCount(stagnationCount, false);
    require(stagnationCount == 1,
        "An improvement did not reset the stagnation counter.");

    EvolutionSettings thresholdSettings;
    thresholdSettings.maximumGenerations = 10;
    thresholdSettings.targetFitness = 0.5;
    thresholdSettings.stagnationGenerations = 1;
    const auto midRunReason =
        determineStopReason(0.4, 2, 1, 1, thresholdSettings);
    require(midRunReason == EvolutionStopReason::TargetFitnessReached,
        "Target fitness did not take precedence after a generation.");

    EvolutionSettings successfulWriteSettings = createFullRunSettings();
    successfulWriteSettings.maximumGenerations = 3;
    successfulWriteSettings.generationLimitMode =
        GenerationLimitMode::SuccessfulWrites;
    std::mt19937_64 successfulWriteGenerator(20260911);
    const EvolutionRunResult successfulWrites = evolve(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        successfulWriteSettings,
        createFocusedMutationSettings(),
        successfulWriteGenerator);
    require(successfulWrites.successfulWrites == 3
            && successfulWrites.stopReason
                == EvolutionStopReason::MaximumSuccessfulWrites,
        "The successful-write limit did not count persisted generations.");
    const std::size_t improvedGenerations =
        static_cast<std::size_t>(std::count_if(successfulWrites.history.begin(),
            successfulWrites.history.end(),
            [](const GenerationSummary& summary) { return summary.improved; }));
    require(improvedGenerations + 1 == successfulWrites.successfulWrites,
        "The successful-write count does not match improved generations plus "
        "the baseline.");
}

void testFullRunValidation()
{
    EvolutionSettings settings;
    settings.maximumGenerations = 0;
    requireThrows([&settings]() {
        determineStopReason(1.0, 0, 1, 0, settings);
    }, "Zero maximum generations");
    settings = EvolutionSettings{};
    settings.targetFitness = std::numeric_limits<double>::infinity();
    requireThrows([&settings]() {
        determineStopReason(1.0, 0, 1, 0, settings);
    }, "Infinite target fitness");
    settings = EvolutionSettings{};
    settings.stagnationGenerations = 0;
    requireThrows([&settings]() {
        determineStopReason(1.0, 0, 1, 0, settings);
    }, "Zero stagnation generations");
}

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void testMetadataOutput()
{
    const TemporaryDirectory directory;
    const ControlledFixture fixture = createControlledFixture();
    const GenerationResult generation = evaluateMutations(fixture.source,
        fixture.sourceFitness,
        fixture.reference,
        {targetAtom()});
    EvolutionSettings evolutionSettings;
    evolutionSettings.candidatesPerGeneration = 1;
    const SineMutationSettings mutationSettings;

    const RunMetadata runMetadata{
        .sourcePath = "source\"name.wav",
        .targetPath = "target\\name.wav",
        .preConversionSourcePath = "original source.mp3",
        .preConversionTargetPath = "original target.flac",
        .sampleRateHz = fixture.source.sampleRateHz,
        .sampleCount = fixture.source.samples.size(),
        .seed = 123456789,
        .evolutionSettings = evolutionSettings,
        .mutationSettings = mutationSettings,
        .genomeMutationSettings = {},
        .searchSettings = {},
        .fitnessDynamicRangeDb = 80.0,
        .fitnessMagnitudeFloor = fixture.reference.magnitudeFloor,
        .fftSize = fftSize,
        .hopSize = hopSize,
        .initialFitness = fixture.sourceFitness,
    };

    writeRunMetadata(directory.path() / "run.json",
        runMetadata,
        RunCompletion{
            .completedGenerations = 1,
            .successfulWrites = 2,
            .finalFitness = generation.fitness,
            .stopReason = EvolutionStopReason::MaximumGenerations,
        });
    writeInitialGenerationMetadata(
        directory.path() / "generation-000.json", fixture.sourceFitness);
    writeGenerationMetadata(directory.path() / "generation-001.json",
        {
            .generation = 1,
            .previousFitness = generation.previousFitness,
            .fitness = generation.fitness,
            .improvement = generation.previousFitness - generation.fitness,
            .improved = generation.improved,
            .candidatesTested = generation.candidatesTested,
            .winningMutations = {AddAtomMutation{
                {1, generation.winningMutations.front()}}},
        });

    const std::string runJson = readFile(directory.path() / "run.json");
    const std::string initialJson =
        readFile(directory.path() / "generation-000.json");
    const std::string generationJson =
        readFile(directory.path() / "generation-001.json");
    require(runJson.find("\"seed\": 123456789") != std::string::npos
            && runJson.find("source\\\"name.wav") != std::string::npos
            && runJson.find("target\\\\name.wav") != std::string::npos
            && runJson.find("\"preConversionSource\": \"original source.mp3\"")
                != std::string::npos
            && runJson.find("\"preConversionTarget\": \"original target.flac\"")
                != std::string::npos
            && runJson.find("\"status\": \"complete\"") != std::string::npos
            && runJson.find("\"stopReason\": \"maximum-generations\"")
                != std::string::npos
            && runJson.find("\"fitnessDynamicRangeDb\": 80")
                != std::string::npos
            && runJson.find("\"guidedMutationRatio\":") != std::string::npos
            && runJson.find("\"refinementBudgetRatio\":") != std::string::npos,
        "Run metadata omitted fields or failed to escape paths.");
    require(initialJson.find("\"generation\": 0") != std::string::npos
            && initialJson.find("\"mutations\": []") != std::string::npos,
        "Generation 0 metadata is incomplete.");
    require(generationJson.find("\"generation\": 1") != std::string::npos
            && generationJson.find("\"improved\": true") != std::string::npos
            && generationJson.find("\"improvement\":") != std::string::npos
            && generationJson.find("\"type\": \"add-sine\"")
                != std::string::npos,
        "Generation 1 metadata is incomplete.");

    writeMorphRunMetadata(directory.path() / "morph-run.json",
        {
            .sourcePath = "source.wav",
            .targetPath = "target.wav",
            .preConversionSourcePath = std::nullopt,
            .preConversionTargetPath = std::nullopt,
            .sampleRateHz = 48'000,
            .sampleCount = 96'000,
            .settings = {.strategy = MorphStrategy::Interpolate,
                .morphSteps = 10},
            .fftSize = 2048,
            .hopSize = 512,
        });
    writeMorphGenerationMetadata(directory.path() / "morph-generation.json",
        {.generation = 4,
            .progress = 0.4,
            .strategy = MorphStrategy::Interpolate});
    const std::string morphRunJson =
        readFile(directory.path() / "morph-run.json");
    const std::string morphGenerationJson =
        readFile(directory.path() / "morph-generation.json");
    require(
        morphRunJson.find("\"mode\": \"spectral-morph\"") != std::string::npos
            && morphRunJson.find("\"strategy\": \"interpolate\"")
                != std::string::npos
            && morphRunJson.find("\"morphSteps\": 10") != std::string::npos
            && morphRunJson.find("candidatesPerGeneration")
                == std::string::npos,
        "Spectral morph run metadata is incomplete or mode-independent.");
    require(morphGenerationJson.find("\"generation\": 4") != std::string::npos
            && morphGenerationJson.find("\"progress\": 0.40000000000000002")
                != std::string::npos
            && morphGenerationJson.find("\"mode\": \"spectral-morph\"")
                != std::string::npos,
        "Spectral morph generation metadata is incomplete.");

    writeMorphRunMetadata(directory.path() / "replace-run.json",
        {
            .sourcePath = "source.wav",
            .targetPath = "target.wav",
            .preConversionSourcePath = std::nullopt,
            .preConversionTargetPath = std::nullopt,
            .sampleRateHz = 48'000,
            .sampleCount = 96'000,
            .settings =
                {
                    .strategy = MorphStrategy::Replace,
                    .morphSteps = 10,
                    .replacementOrder = ReplacementOrder::Error,
                    .replacementDynamicRangeDb = 80.0,
                },
            .fftSize = 2048,
            .hopSize = 512,
        });
    writeMorphGenerationMetadata(directory.path() / "replace-generation.json",
        {
            .generation = 4,
            .progress = 0.4,
            .strategy = MorphStrategy::Replace,
            .replacementOrder = ReplacementOrder::Error,
            .cellsReplaced = 82'340,
            .totalCells = 205'850,
        });
    const std::string replaceRunJson =
        readFile(directory.path() / "replace-run.json");
    const std::string replaceGenerationJson =
        readFile(directory.path() / "replace-generation.json");
    require(
        replaceRunJson.find("\"strategy\": \"replace\"") != std::string::npos
            && replaceRunJson.find("\"replacementOrder\": \"error\"")
                != std::string::npos
            && replaceRunJson.find("\"replacementDynamicRangeDb\": 80")
                != std::string::npos,
        "Replacement run metadata is incomplete.");
    require(replaceGenerationJson.find("\"cellsReplaced\": 82340")
                != std::string::npos
            && replaceGenerationJson.find("\"totalCells\": 205850")
                != std::string::npos
            && replaceGenerationJson.find("\"replacementOrder\": \"error\"")
                != std::string::npos,
        "Replacement generation metadata is incomplete.");

    writeMorphRunMetadata(directory.path() / "random-run.json",
        {
            .sourcePath = "source.wav",
            .targetPath = "target.wav",
            .preConversionSourcePath = std::nullopt,
            .preConversionTargetPath = std::nullopt,
            .sampleRateHz = 48'000,
            .sampleCount = 96'000,
            .settings =
                {
                    .strategy = MorphStrategy::Replace,
                    .morphSteps = 10,
                    .replacementOrder = ReplacementOrder::Random,
                    .replacementDynamicRangeDb = 80.0,
                    .replacementSeed = 123456789,
                },
            .fftSize = 2048,
            .hopSize = 512,
        });
    const std::string randomRunJson =
        readFile(directory.path() / "random-run.json");
    require(randomRunJson.find("\"replacementOrder\": \"random\"")
                != std::string::npos
            && randomRunJson.find("\"seed\": 123456789") != std::string::npos,
        "Random replacement metadata omitted its ordering seed.");

    writeMorphRunMetadata(directory.path() / "polar-run.json",
        {
            .sourcePath = "source.wav",
            .targetPath = "target.wav",
            .preConversionSourcePath = std::nullopt,
            .preConversionTargetPath = std::nullopt,
            .sampleRateHz = 48'000,
            .sampleCount = 96'000,
            .settings =
                {
                    .strategy = MorphStrategy::PolarInterpolate,
                    .morphSteps = 10,
                    .interpolationDynamicRangeDb = 80.0,
                },
            .fftSize = 2048,
            .hopSize = 512,
        });
    const std::string polarRunJson =
        readFile(directory.path() / "polar-run.json");
    require(polarRunJson.find("\"strategy\": \"polar-interpolate\"")
                != std::string::npos
            && polarRunJson.find("\"interpolationDynamicRangeDb\": 80")
                != std::string::npos
            && polarRunJson.find("replacementOrder") == std::string::npos,
        "Polar interpolation metadata is incomplete or contains replacement "
        "fields.");

    const Genome genome{{GenomeAtom{77, targetAtom()}}};
    writeGenomeMetadata(directory.path() / "genome.json", genome);
    const std::string genomeJson = readFile(directory.path() / "genome.json");
    require(genomeJson.find("\"atomId\": 77") != std::string::npos
            && genomeJson.find("\"frequencyHz\": 128") != std::string::npos,
        "Final genome metadata is incomplete.");
}

void testPersistedRunOutputCountsAndFinalization()
{
    const TemporaryDirectory directory;
    const ControlledFixture fixture = createControlledFixture();
    EvolutionSettings settings;
    settings.candidatesPerGeneration = 1;
    settings.maximumGenerations = 3;
    SineMutationSettings mutationSettings = createFocusedMutationSettings();
    mutationSettings.maximumAbsoluteAmplitude = 0.0;
    const RunMetadata runMetadata{
        .sourcePath = "source.wav",
        .targetPath = "target.wav",
        .preConversionSourcePath = std::nullopt,
        .preConversionTargetPath = std::nullopt,
        .sampleRateHz = fixture.target.sampleRateHz,
        .sampleCount = fixture.target.samples.size(),
        .seed = 99,
        .evolutionSettings = settings,
        .mutationSettings = mutationSettings,
        .genomeMutationSettings = {},
        .searchSettings = {},
        .fitnessDynamicRangeDb = 80.0,
        .fitnessMagnitudeFloor = fixture.reference.magnitudeFloor,
        .fftSize = fftSize,
        .hopSize = hopSize,
        .initialFitness = 0.0,
    };
    writeRunMetadata(directory.path() / "run.json", runMetadata);

    std::mt19937_64 generator(99);
    const GenerationObserver observer = [&directory](
                                            const GenerationSummary& summary,
                                            const AudioBuffer& audio) {
        const auto wavPath = directory.path()
            / formatGenerationFilename(summary.generation, ".wav");
        const auto jsonPath = directory.path()
            / formatGenerationFilename(summary.generation, ".json");
        if (summary.generation == 0) {
            saveAudio(wavPath.string(), audio);
            writeInitialGenerationMetadata(jsonPath, summary.fitness);
            return;
        }
        if (!summary.improved)
            return;
        saveAudio(wavPath.string(), audio);
        writeGenerationMetadata(jsonPath, summary);
    };
    const EvolutionRunResult result = evolve(fixture.target,
        0.0,
        fixture.reference,
        settings,
        mutationSettings,
        generator,
        observer);
    writeRunMetadata(directory.path() / "run.json",
        runMetadata,
        RunCompletion{
            .completedGenerations = result.completedGenerations,
            .successfulWrites = result.successfulWrites,
            .finalFitness = result.finalFitness,
            .stopReason = result.stopReason,
        });

    std::size_t wavCount = 0;
    std::size_t generationJsonCount = 0;
    for (const auto& entry :
        std::filesystem::directory_iterator(directory.path())) {
        if (entry.path().extension() == ".wav") {
            ++wavCount;
        } else if (entry.path().extension() == ".json"
            && entry.path().filename() != "run.json") {
            ++generationJsonCount;
        }
    }
    require(wavCount == 1 && generationJsonCount == 1,
        "Flat generations unexpectedly produced output files.");
    require(std::filesystem::exists(directory.path() / "generation-000.wav")
            && !std::filesystem::exists(directory.path() / "generation-001.wav")
            && !std::filesystem::exists(
                directory.path() / "generation-003.json"),
        "Initial or skipped-generation persistence is incorrect.");
    require(formatGenerationFilename(17, ".wav") == "generation-017.wav"
            && formatGenerationFilename(100, ".json") == "generation-100.json",
        "Generation filename formatting is incorrect.");

    const std::string runJson = readFile(directory.path() / "run.json");
    require(runJson.find("\"completedGenerations\": 3") != std::string::npos
            && runJson.find("\"successfulWrites\": 1") != std::string::npos
            && runJson.find("\"generationLimitMode\": \"attempts\"")
                != std::string::npos
            && runJson.find("\"finalFitness\": 0") != std::string::npos
            && runJson.find("\"stopReason\": \"maximum-generations\"")
                != std::string::npos
            && runJson.find("\"status\": \"complete\"") != std::string::npos,
        "Finalized run metadata is incomplete.");
}

}  // namespace

int main()
{
    try {
        testSameSeedProducesSameAtoms();
        testDifferentSeedsProduceDifferentSequences();
        testMutationBoundsAndAmplitudeSigns();
        testShortAudioAndNyquist();
        testMutationValidation();
        testControlledImprovementAndBestSelection();
        testRangesAndMultipleWinnerSelection();
        testEditableGenomeOperations();
        testIncrementalGenomeEvaluationMatchesFullAnalysis();
        testGuidedProposalLocationAndDeterminism();
        testErrorWeightedAtomSelectionScores();
        testGuidedGenomeEditsFollowResidual();
        testSearchBudgetAndGuidanceStatistics();
        testBoundedAmplitudeAndPhaseSearch();
        testNoImprovementAndTiePreserveParent();
        testOneAndZeroCandidates();
        testGenerationReproducibilityAndElitism();
        testOneGenerationFullRunCompatibility();
        testFullRunDeterminismCarryForwardAndHistory();
        testStoppingConditions();
        testFullRunValidation();
        testMetadataOutput();
        testPersistedRunOutputCountsAndFinalization();
        std::cout << "All evolution tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Evolution test failure: " << error.what() << '\n';
        return 1;
    }
}
