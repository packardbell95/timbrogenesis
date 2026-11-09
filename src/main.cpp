#include "audio.hpp"
#include "evolution.hpp"
#include "fitness.hpp"
#include "metadata.hpp"
#include "mutation.hpp"
#include "spectral_morph.hpp"
#include "spectrum.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class ApplicationMode {
    Evolution,
    SpectralMorph,
};

constexpr std::size_t defaultFftSize = 2048;
constexpr std::size_t defaultHopSize = 512;

struct CommandLineOptions {
    std::string sourcePath;
    std::string targetPath;
    std::optional<std::string> preConversionSourcePath;
    std::optional<std::string> preConversionTargetPath;
    std::optional<std::filesystem::path> outputDirectory;
    std::optional<RandomSeed> seed;
    EvolutionSettings evolutionSettings;
    GuidedSearchSettings searchSettings;
    double fitnessDynamicRangeDb = 80.0;
    ApplicationMode mode = ApplicationMode::Evolution;
    SpectralMorphSettings morphSettings;
};

std::uint64_t parseUnsignedInteger(
    std::string_view value, std::string_view optionName)
{
    std::uint64_t parsedValue = 0;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsedValue);
    if (value.empty() || result.ec != std::errc{}
        || result.ptr != value.data() + value.size()) {
        throw std::runtime_error("Invalid value for " + std::string(optionName)
            + ": '" + std::string(value) + "'.");
    }
    return parsedValue;
}

std::size_t parsePositiveSize(
    std::string_view value, std::string_view optionName)
{
    const std::uint64_t parsedValue = parseUnsignedInteger(value, optionName);
    if (parsedValue == 0) {
        throw std::runtime_error(
            std::string(optionName) + " must be greater than zero.");
    }
    if (parsedValue > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(optionName) + " is too large.");
    }
    return static_cast<std::size_t>(parsedValue);
}

double parseNonNegativeDouble(
    std::string_view value, std::string_view optionName)
{
    double parsedValue = 0.0;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsedValue);
    if (value.empty() || result.ec != std::errc{}
        || result.ptr != value.data() + value.size()
        || !std::isfinite(parsedValue) || parsedValue < 0.0) {
        throw std::runtime_error("Invalid value for " + std::string(optionName)
            + ": '" + std::string(value) + "'.");
    }
    return parsedValue;
}

double parsePositiveDouble(std::string_view value, std::string_view optionName)
{
    const double parsed = parseNonNegativeDouble(value, optionName);
    if (parsed == 0.0)
        throw std::runtime_error(
            std::string(optionName) + " must be greater than zero.");
    return parsed;
}

double parseRatio(std::string_view value, std::string_view optionName)
{
    const double parsed = parseNonNegativeDouble(value, optionName);
    if (parsed > 1.0)
        throw std::runtime_error(
            std::string(optionName) + " must not exceed one.");
    return parsed;
}

CommandLineOptions parseCommandLine(int argc, char* argv[])
{
    CommandLineOptions options;
    std::vector<std::string> positionalArguments;
    bool candidateCountSpecified = false;
    bool generationLimitSpecified = false;
    bool modeSpecified = false;
    bool morphStrategySpecified = false;
    bool morphStepsSpecified = false;
    bool replacementOrderSpecified = false;
    std::optional<std::string> evolutionOnlyOption;
    std::optional<std::string> morphOnlyOption;

    for (int argumentIndex = 1; argumentIndex < argc; ++argumentIndex) {
        const std::string_view argument = argv[argumentIndex];
        const bool hasValue = argument == "--seed" || argument == "--mode"
            || argument == "--morph-strategy" || argument == "--morph-steps"
            || argument == "--replacement-order" || argument == "--candidates"
            || argument == "--candidates-per-generation"
            || argument == "--generations"
            || argument == "--maximum-generations"
            || argument == "--maximum-attempts"
            || argument == "--maximum-successful-writes"
            || argument == "--max-accepted-mutations"
            || argument == "--fitness-dynamic-range"
            || argument == "--guided-mutation-ratio"
            || argument == "--refinement-ratio"
            || argument == "--target-fitness"
            || argument == "--stagnation-generations"
            || argument == "--pre-conversion-source"
            || argument == "--pre-conversion-target" || argument == "--output";

        if (hasValue) {
            if (argumentIndex + 1 >= argc) {
                throw std::runtime_error(
                    "Missing value for " + std::string(argument) + ".");
            }
            const std::string_view value = argv[++argumentIndex];

            if (argument == "--mode") {
                if (modeSpecified)
                    throw std::runtime_error(
                        "Application mode may only be specified once.");
                if (value == "evolution")
                    options.mode = ApplicationMode::Evolution;
                else if (value == "spectral-morph")
                    options.mode = ApplicationMode::SpectralMorph;
                else
                    throw std::runtime_error("Invalid value for --mode: '"
                        + std::string(value) + "'.");
                modeSpecified = true;
            } else if (argument == "--morph-strategy") {
                if (morphStrategySpecified)
                    throw std::runtime_error(
                        "Morph strategy may only be specified once.");
                if (value == "interpolate")
                    options.morphSettings.strategy = MorphStrategy::Interpolate;
                else if (value == "polar-interpolate")
                    options.morphSettings.strategy =
                        MorphStrategy::PolarInterpolate;
                else if (value == "magnitude-source-phase")
                    options.morphSettings.strategy =
                        MorphStrategy::MagnitudeSourcePhase;
                else if (value == "magnitude-target-phase")
                    options.morphSettings.strategy =
                        MorphStrategy::MagnitudeTargetPhase;
                else if (value == "replace")
                    options.morphSettings.strategy = MorphStrategy::Replace;
                else
                    throw std::runtime_error(
                        "Invalid value for --morph-strategy: '"
                        + std::string(value) + "'.");
                morphStrategySpecified = true;
                morphOnlyOption = std::string(argument);
            } else if (argument == "--morph-steps") {
                if (morphStepsSpecified)
                    throw std::runtime_error(
                        "Morph step count may only be specified once.");
                const std::size_t steps = parsePositiveSize(value, argument);
                if (steps == std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("--morph-steps is too large.");
                options.morphSettings.morphSteps = steps;
                morphStepsSpecified = true;
                if (!morphOnlyOption)
                    morphOnlyOption = std::string(argument);
            } else if (argument == "--replacement-order") {
                if (replacementOrderSpecified)
                    throw std::runtime_error(
                        "Replacement order may only be specified once.");
                if (value == "error")
                    options.morphSettings.replacementOrder =
                        ReplacementOrder::Error;
                else if (value == "target-energy")
                    options.morphSettings.replacementOrder =
                        ReplacementOrder::TargetEnergy;
                else if (value == "source-energy")
                    options.morphSettings.replacementOrder =
                        ReplacementOrder::SourceEnergy;
                else if (value == "low-to-high")
                    options.morphSettings.replacementOrder =
                        ReplacementOrder::LowToHigh;
                else if (value == "high-to-low")
                    options.morphSettings.replacementOrder =
                        ReplacementOrder::HighToLow;
                else if (value == "early-to-late")
                    options.morphSettings.replacementOrder =
                        ReplacementOrder::EarlyToLate;
                else if (value == "late-to-early")
                    options.morphSettings.replacementOrder =
                        ReplacementOrder::LateToEarly;
                else if (value == "random")
                    options.morphSettings.replacementOrder =
                        ReplacementOrder::Random;
                else
                    throw std::runtime_error(
                        "Invalid value for --replacement-order: '"
                        + std::string(value) + "'.");
                replacementOrderSpecified = true;
                if (!morphOnlyOption)
                    morphOnlyOption = std::string(argument);
            } else if (argument == "--seed") {
                options.seed = parseUnsignedInteger(value, argument);
            } else if (argument == "--candidates"
                || argument == "--candidates-per-generation") {
                if (candidateCountSpecified)
                    throw std::runtime_error("Candidates per generation may "
                                             "only be specified once.");
                options.evolutionSettings.candidatesPerGeneration =
                    parsePositiveSize(value, argument);
                candidateCountSpecified = true;
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--generations"
                || argument == "--maximum-generations"
                || argument == "--maximum-attempts") {
                if (generationLimitSpecified)
                    throw std::runtime_error(
                        "The generation limit may only be specified once.");
                options.evolutionSettings.maximumGenerations =
                    parsePositiveSize(value, argument);
                options.evolutionSettings.generationLimitMode =
                    GenerationLimitMode::Attempts;
                generationLimitSpecified = true;
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--maximum-successful-writes") {
                if (generationLimitSpecified)
                    throw std::runtime_error(
                        "The generation limit may only be specified once.");
                options.evolutionSettings.maximumGenerations =
                    parsePositiveSize(value, argument);
                options.evolutionSettings.generationLimitMode =
                    GenerationLimitMode::SuccessfulWrites;
                generationLimitSpecified = true;
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--max-accepted-mutations") {
                options.evolutionSettings
                    .maximumAcceptedMutationsPerGeneration =
                    parsePositiveSize(value, argument);
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--fitness-dynamic-range") {
                options.fitnessDynamicRangeDb =
                    parsePositiveDouble(value, argument);
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--guided-mutation-ratio") {
                options.searchSettings.guidedMutationRatio =
                    parseRatio(value, argument);
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--refinement-ratio") {
                const double ratio = parseRatio(value, argument);
                if (ratio >= 1.0)
                    throw std::runtime_error(
                        "--refinement-ratio must be less than one.");
                options.searchSettings.refinementBudgetRatio = ratio;
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--target-fitness") {
                options.evolutionSettings.targetFitness =
                    parseNonNegativeDouble(value, argument);
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--stagnation-generations") {
                options.evolutionSettings.stagnationGenerations =
                    parsePositiveSize(value, argument);
                if (!evolutionOnlyOption)
                    evolutionOnlyOption = std::string(argument);
            } else if (argument == "--pre-conversion-source") {
                options.preConversionSourcePath = value;
            } else if (argument == "--pre-conversion-target") {
                options.preConversionTargetPath = value;
            } else {
                if (options.outputDirectory.has_value()) {
                    throw std::runtime_error(
                        "Output directory may only be specified once.");
                }
                options.outputDirectory = std::filesystem::path(value);
            }
        } else if (argument.starts_with("--")) {
            throw std::runtime_error(
                "Unknown option: " + std::string(argument) + ".");
        } else {
            positionalArguments.emplace_back(argument);
        }
    }

    if (positionalArguments.size() < 2 || positionalArguments.size() > 3) {
        throw std::runtime_error(
            "Usage: timbrogenesis <source.wav> <target.wav> "
            "[output-directory] [--output PATH] "
            "[--mode evolution|spectral-morph] "
            "[--morph-strategy interpolate|polar-interpolate|"
            "magnitude-source-phase|magnitude-target-phase|replace] "
            "[--morph-steps N] "
            "[--replacement-order "
            "error|target-energy|source-energy|low-to-high|"
            "high-to-low|early-to-late|late-to-early|random] [--seed N] "
            "[--candidates-per-generation N] [--maximum-generations N] "
            "[--maximum-attempts N] [--maximum-successful-writes N] "
            "[--max-accepted-mutations N] "
            "[--fitness-dynamic-range DB] [--guided-mutation-ratio X] "
            "[--refinement-ratio X] "
            "[--target-fitness X] [--stagnation-generations N] "
            "[--pre-conversion-source PATH] "
            "[--pre-conversion-target PATH]");
    }

    options.sourcePath = positionalArguments[0];
    options.targetPath = positionalArguments[1];
    if (positionalArguments.size() == 3) {
        if (options.outputDirectory.has_value()) {
            throw std::runtime_error(
                "Output directory may only be specified once.");
        }
        options.outputDirectory = positionalArguments[2];
    }
    if (options.mode == ApplicationMode::SpectralMorph && evolutionOnlyOption) {
        throw std::runtime_error(
            *evolutionOnlyOption + " is only valid in evolution mode.");
    }
    if (options.mode == ApplicationMode::Evolution && morphOnlyOption) {
        throw std::runtime_error(
            *morphOnlyOption + " is only valid in spectral-morph mode.");
    }
    if (replacementOrderSpecified
        && options.morphSettings.strategy != MorphStrategy::Replace) {
        throw std::runtime_error(
            "--replacement-order is only valid with --morph-strategy replace.");
    }
    if (options.mode == ApplicationMode::SpectralMorph && options.seed
        && (options.morphSettings.strategy != MorphStrategy::Replace
            || options.morphSettings.replacementOrder
                != ReplacementOrder::Random)) {
        throw std::runtime_error("--seed is only valid for random replacement "
                                 "order in spectral-morph mode.");
    }
    return options;
}

std::filesystem::path createDefaultRunDirectory()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
    if (localtime_r(&currentTime, &localTime) == nullptr) {
        throw std::runtime_error("Unable to create a timestamped run name.");
    }

    std::ostringstream timestamp;
    timestamp << std::put_time(&localTime, "%Y-%m-%d_%H%M%S");
    const std::filesystem::path basePath = "output";
    std::filesystem::path candidate = basePath / timestamp.str();
    for (std::size_t suffix = 1; std::filesystem::exists(candidate); ++suffix) {
        candidate = basePath / (timestamp.str() + "_" + std::to_string(suffix));
    }
    return candidate;
}

void prepareRunDirectory(const std::filesystem::path& runDirectory)
{
    if (std::filesystem::exists(runDirectory)) {
        if (!std::filesystem::is_directory(runDirectory)) {
            throw std::runtime_error(
                "Output path exists and is not a directory: '"
                + runDirectory.string() + "'.");
        }
        if (!std::filesystem::is_empty(runDirectory)) {
            throw std::runtime_error("Output directory is not empty: '"
                + runDirectory.string() + "'.");
        }
        return;
    }
    std::filesystem::create_directories(runDirectory);
}

void printGenerationProgress(const GenerationSummary& summary,
    const EvolutionSettings& settings,
    std::size_t successfulWrites,
    std::size_t stagnationCount)
{
    std::cout << "Generation " << summary.generation;
    if (settings.generationLimitMode == GenerationLimitMode::Attempts)
        std::cout << '/' << settings.maximumGenerations;
    std::cout << '\n';
    if (summary.improved) {
        std::cout << "Fitness: " << *summary.previousFitness << " -> "
                  << summary.fitness << '\n'
                  << "Improvement: " << summary.improvement << '\n';
    } else {
        std::cout << "Fitness: " << summary.fitness << '\n'
                  << "No improving candidate found.\n";
    }
    std::cout << "Candidates: " << summary.candidatesTested << '\n'
              << "Guided/random/refined: " << summary.guidedCandidatesTested
              << '/' << summary.randomCandidatesTested << '/'
              << summary.refinementCandidatesTested << '\n'
              << "Accepted mutations: " << summary.winningMutations.size()
              << '\n'
              << "Successful writes: " << successfulWrites;
    if (settings.generationLimitMode == GenerationLimitMode::SuccessfulWrites)
        std::cout << '/' << settings.maximumGenerations;
    std::cout << '\n' << "Stagnation: " << stagnationCount << "\n\n";
}

std::string_view humanStopReason(EvolutionStopReason reason)
{
    switch (reason) {
    case EvolutionStopReason::MaximumGenerations:
        return "maximum generations reached";
    case EvolutionStopReason::MaximumSuccessfulWrites:
        return "maximum successful writes reached";
    case EvolutionStopReason::TargetFitnessReached:
        return "target fitness reached";
    case EvolutionStopReason::Stagnation:
        return "stagnation";
    }
    return "unknown";
}

void runSpectralMorphApplication(const CommandLineOptions& options,
    const AudioBuffer& source,
    const AudioBuffer& target,
    const std::filesystem::path& runDirectory)
{
    ComplexStftAnalyzer analyzer(defaultFftSize, defaultHopSize);
    ComplexStftSynthesizer synthesizer(defaultFftSize);
    SpectralMorphSettings morphSettings = options.morphSettings;
    if (morphSettings.strategy == MorphStrategy::Replace
        && morphSettings.replacementOrder == ReplacementOrder::Random) {
        morphSettings.replacementSeed =
            options.seed.value_or(createRandomSeed());
    }
    const MorphGenerationObserver observer =
        [&runDirectory](
            const MorphGeneration& generation, const AudioBuffer& audio) {
        const auto wavPath = runDirectory
            / formatGenerationFilename(generation.generation, ".wav");
        const auto jsonPath = runDirectory
            / formatGenerationFilename(generation.generation, ".json");
        saveAudio(wavPath.string(), audio);
        writeMorphGenerationMetadata(jsonPath, generation);
        std::cout << "Generation " << generation.generation << " progress "
                  << std::fixed << std::setprecision(1)
                  << generation.progress * 100.0 << '%';
        if (generation.strategy == MorphStrategy::Replace)
            std::cout << " (" << generation.cellsReplaced << '/'
                      << generation.totalCells << " cells replaced)";
        std::cout << '\n';
    };

    std::cout << "Validated compatible mono WAV files: "
              << source.samples.size() << " samples at " << source.sampleRateHz
              << " Hz.\n"
              << "Spectral morph: "
              << morphStrategyValue(morphSettings.strategy) << ", "
              << morphSettings.morphSteps << " intervals ("
              << morphSettings.morphSteps + 1 << " generations)";
    if (morphSettings.strategy == MorphStrategy::Replace)
        std::cout << ", "
                  << replacementOrderValue(morphSettings.replacementOrder)
                  << " replacement order";
    if (morphSettings.replacementOrder == ReplacementOrder::Random)
        std::cout << ", seed " << morphSettings.replacementSeed;
    std::cout << ".\n";
    runSpectralMorph(
        source, target, morphSettings, analyzer, synthesizer, observer);
    writeMorphRunMetadata(runDirectory / "run.json",
        {
            .sourcePath = options.sourcePath,
            .targetPath = options.targetPath,
            .preConversionSourcePath = options.preConversionSourcePath,
            .preConversionTargetPath = options.preConversionTargetPath,
            .sampleRateHz = source.sampleRateHz,
            .sampleCount = source.samples.size(),
            .settings = morphSettings,
            .fftSize = defaultFftSize,
            .hopSize = defaultHopSize,
        });
    std::cout << "Spectral morph complete.\nOutput: " << runDirectory.string()
              << '\n';
}

}  // namespace

int main(int argc, char* argv[])
{
    try {
        const CommandLineOptions options = parseCommandLine(argc, argv);
        const AudioBuffer source = loadAudio(options.sourcePath);
        const AudioBuffer target = loadAudio(options.targetPath);
        validateCompatibleAudio(source, target);

        const std::filesystem::path runDirectory =
            options.outputDirectory.has_value() ? *options.outputDirectory
                                                : createDefaultRunDirectory();
        prepareRunDirectory(runDirectory);

        if (options.mode == ApplicationMode::SpectralMorph) {
            runSpectralMorphApplication(options, source, target, runDirectory);
            return 0;
        }

        const SineMutationSettings mutationSettings;
        const GenomeMutationSettings genomeMutationSettings;
        validateSineMutationSettings(source, mutationSettings);

        constexpr std::size_t fftSize = defaultFftSize;
        constexpr std::size_t hopSize = defaultHopSize;
        StftAnalyzer analyzer(fftSize, hopSize);
        const Spectrogram sourceSpectrogram = analyzer.calculate(source);
        const Spectrogram targetSpectrogram = analyzer.calculate(target);
        const FitnessReference targetFitnessReference = createFitnessReference(
            targetSpectrogram, options.fitnessDynamicRangeDb);
        const double targetFitness =
            calculateSpectralFitness(targetSpectrogram, targetFitnessReference);
        const double sourceFitness =
            calculateSpectralFitness(sourceSpectrogram, targetFitnessReference);

        const RandomSeed seed =
            options.seed.has_value() ? *options.seed : createRandomSeed();
        std::mt19937_64 randomGenerator(seed);

        const RunMetadata runMetadata{
            .sourcePath = options.sourcePath,
            .targetPath = options.targetPath,
            .preConversionSourcePath = options.preConversionSourcePath,
            .preConversionTargetPath = options.preConversionTargetPath,
            .sampleRateHz = source.sampleRateHz,
            .sampleCount = source.samples.size(),
            .seed = seed,
            .evolutionSettings = options.evolutionSettings,
            .mutationSettings = mutationSettings,
            .genomeMutationSettings = genomeMutationSettings,
            .searchSettings = options.searchSettings,
            .fitnessDynamicRangeDb = options.fitnessDynamicRangeDb,
            .fitnessMagnitudeFloor = targetFitnessReference.magnitudeFloor,
            .fftSize = fftSize,
            .hopSize = hopSize,
            .initialFitness = sourceFitness,
        };
        writeRunMetadata(runDirectory / "run.json", runMetadata);

        std::size_t stagnationCount = 0;
        std::size_t successfulWrites = 1;
        const GenerationObserver observer =
            [&runDirectory, &stagnationCount, &successfulWrites, &options](
                const GenerationSummary& summary, const AudioBuffer& audio) {
            const auto wavPath = runDirectory
                / formatGenerationFilename(summary.generation, ".wav");
            const auto jsonPath = runDirectory
                / formatGenerationFilename(summary.generation, ".json");

            if (summary.generation == 0) {
                saveAudio(wavPath.string(), audio);
                writeInitialGenerationMetadata(jsonPath, summary.fitness);
                return;
            }

            stagnationCount =
                updateStagnationCount(stagnationCount, summary.improved);
            if (summary.improved)
                ++successfulWrites;
            printGenerationProgress(summary,
                options.evolutionSettings,
                successfulWrites,
                stagnationCount);

            if (!summary.improved) {
                return;
            }

            saveAudio(wavPath.string(), audio);
            writeGenerationMetadata(jsonPath, summary);
        };

        std::cout << std::fixed << std::setprecision(10)
                  << "Validated compatible mono WAV files: "
                  << source.samples.size() << " samples at "
                  << source.sampleRateHz << " Hz.\n"
                  << "STFT: " << sourceSpectrogram.frameCount << " frames, "
                  << sourceSpectrogram.binCount << " bins (FFT " << fftSize
                  << ", hop " << hopSize << ").\n"
                  << "Target vs target fitness: " << targetFitness << '\n'
                  << "Seed: " << seed << "\n\n"
                  << "Generation 0\n"
                  << "Fitness: " << sourceFitness << "\n\n";

        const EvolutionRunResult result = evolve(source,
            sourceFitness,
            targetFitnessReference,
            options.evolutionSettings,
            mutationSettings,
            genomeMutationSettings,
            options.searchSettings,
            randomGenerator,
            analyzer,
            observer);
        writeGenomeMetadata(runDirectory / "genome.json", result.finalGenome);
        writeRunMetadata(runDirectory / "run.json",
            runMetadata,
            RunCompletion{
                .completedGenerations = result.completedGenerations,
                .successfulWrites = result.successfulWrites,
                .finalFitness = result.finalFitness,
                .stopReason = result.stopReason,
                .guidedCandidatesTested = result.guidedCandidatesTested,
                .randomCandidatesTested = result.randomCandidatesTested,
                .refinementCandidatesTested = result.refinementCandidatesTested,
                .guidedWinnersAccepted = result.guidedWinnersAccepted,
                .randomWinnersAccepted = result.randomWinnersAccepted,
                .refinementWinnersAccepted = result.refinementWinnersAccepted,
            });

        std::cout << "Evolution complete.\n"
                  << "Generations completed: " << result.completedGenerations
                  << '\n'
                  << "Successful writes: " << result.successfulWrites << '\n'
                  << "Initial fitness: " << result.initialFitness << '\n'
                  << "Final fitness: " << result.finalFitness << '\n'
                  << "Stop reason: " << humanStopReason(result.stopReason)
                  << '\n'
                  << "Output: " << runDirectory.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
