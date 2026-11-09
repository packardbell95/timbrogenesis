#include "metadata.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace {

void writeJsonString(std::ostream& output, std::string_view value)
{
    constexpr char hex[] = "0123456789abcdef";
    output << '"';
    for (const char raw : value) {
        const auto character = static_cast<unsigned char>(raw);
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U)
                output << "\\u00" << hex[character >> 4U]
                       << hex[character & 0x0fU];
            else
                output << static_cast<char>(character);
        }
    }
    output << '"';
}

std::ofstream openMetadataFile(const std::filesystem::path& path)
{
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error(
            "Unable to create metadata file: '" + path.string() + "'.");
    output << std::setprecision(17);
    return output;
}

void finishMetadataFile(
    std::ofstream& output, const std::filesystem::path& path)
{
    output.flush();
    if (!output)
        throw std::runtime_error(
            "Unable to write metadata file: '" + path.string() + "'.");
}

void writeAtomFields(
    std::ostream& output, const SineAtom& atom, std::string_view indent)
{
    output << indent << "\"startSeconds\": " << atom.startSeconds << ",\n"
           << indent << "\"durationSeconds\": " << atom.durationSeconds << ",\n"
           << indent << "\"frequencyHz\": " << atom.frequencyHz << ",\n"
           << indent << "\"amplitude\": " << atom.amplitude << ",\n"
           << indent << "\"phaseRadians\": " << atom.phaseRadians;
}

void writeMutation(std::ostream& output, const GenomeMutation& mutation)
{
    std::visit([&](const auto& operation) {
        using T = std::decay_t<decltype(operation)>;
        output << "{\n";
        if constexpr (std::is_same_v<T, AddAtomMutation>) {
            output << "      \"type\": \"add-sine\",\n"
                   << "      \"atomId\": " << operation.atom.id << ",\n"
                   << "      \"atom\": {\n";
            writeAtomFields(output, operation.atom.atom, "        ");
            output << "\n      }\n";
        } else if constexpr (std::is_same_v<T, RemoveAtomMutation>) {
            output << "      \"type\": \"remove-sine\",\n"
                   << "      \"atomId\": " << operation.atomId << '\n';
        } else if constexpr (std::is_same_v<T, MoveAtomMutation>) {
            output << "      \"type\": \"move-sine\",\n      \"atomId\": "
                   << operation.atomId << ",\n      \"previousStartSeconds\": "
                   << operation.previousStartSeconds
                   << ",\n      \"startSeconds\": " << operation.startSeconds
                   << '\n';
        } else if constexpr (std::is_same_v<T, ChangeFrequencyMutation>) {
            output << "      \"type\": \"change-frequency\",\n      "
                      "\"atomId\": "
                   << operation.atomId << ",\n      \"previousFrequencyHz\": "
                   << operation.previousFrequencyHz
                   << ",\n      \"frequencyHz\": " << operation.frequencyHz
                   << '\n';
        } else if constexpr (std::is_same_v<T, ChangeAmplitudeMutation>) {
            output << "      \"type\": \"change-amplitude\",\n      "
                      "\"atomId\": "
                   << operation.atomId << ",\n      \"previousAmplitude\": "
                   << operation.previousAmplitude
                   << ",\n      \"amplitude\": " << operation.amplitude << '\n';
        } else if constexpr (std::is_same_v<T, ChangeDurationMutation>) {
            output << "      \"type\": \"change-duration\",\n      "
                      "\"atomId\": "
                   << operation.atomId
                   << ",\n      \"previousDurationSeconds\": "
                   << operation.previousDurationSeconds
                   << ",\n      \"durationSeconds\": "
                   << operation.durationSeconds << '\n';
        } else {
            output << "      \"type\": \"change-phase\",\n      \"atomId\": "
                   << operation.atomId << ",\n      \"previousPhaseRadians\": "
                   << operation.previousPhaseRadians
                   << ",\n      \"phaseRadians\": " << operation.phaseRadians
                   << '\n';
        }
        output << "    }";
    }, mutation);
}

void writeOptionalDouble(
    std::ostream& output, const std::optional<double>& value)
{
    if (value)
        output << *value;
    else
        output << "null";
}
void writeOptionalSize(
    std::ostream& output, const std::optional<std::size_t>& value)
{
    if (value)
        output << *value;
    else
        output << "null";
}

}  // namespace

std::string formatGenerationFilename(
    std::size_t generation, std::string_view extension)
{
    std::ostringstream name;
    name << "generation-" << std::setfill('0') << std::setw(3) << generation
         << extension;
    return name.str();
}

void writeRunMetadata(const std::filesystem::path& path,
    const RunMetadata& metadata,
    const std::optional<RunCompletion>& completion)
{
    std::ofstream output = openMetadataFile(path);
    output << "{\n  \"mode\": \"evolution\",\n  \"source\": ";
    writeJsonString(output, metadata.sourcePath);
    output << ",\n  \"target\": ";
    writeJsonString(output, metadata.targetPath);
    output << ",\n  \"preConversionSource\": ";
    if (metadata.preConversionSourcePath)
        writeJsonString(output, *metadata.preConversionSourcePath);
    else
        output << "null";
    output << ",\n  \"preConversionTarget\": ";
    if (metadata.preConversionTargetPath)
        writeJsonString(output, *metadata.preConversionTargetPath);
    else
        output << "null";
    output << ",\n  \"sampleRateHz\": " << metadata.sampleRateHz
           << ",\n  \"sampleCount\": " << metadata.sampleCount
           << ",\n  \"seed\": " << metadata.seed
           << ",\n  \"maximumGenerations\": "
           << metadata.evolutionSettings.maximumGenerations
           << ",\n  \"generationLimitMode\": ";
    writeJsonString(output,
        generationLimitModeValue(
            metadata.evolutionSettings.generationLimitMode));
    output << ",\n  \"successfulWrites\": ";
    if (completion)
        output << completion->successfulWrites;
    else
        output << "null";
    output << ",\n  \"completedGenerations\": ";
    if (completion)
        output << completion->completedGenerations;
    else
        output << "null";
    output << ",\n  \"candidatesPerGeneration\": "
           << metadata.evolutionSettings.candidatesPerGeneration
           << ",\n  \"maximumAcceptedMutationsPerGeneration\": "
           << metadata.evolutionSettings.maximumAcceptedMutationsPerGeneration
           << ",\n  \"fitnessDynamicRangeDb\": "
           << metadata.fitnessDynamicRangeDb
           << ",\n  \"fitnessMagnitudeFloor\": "
           << metadata.fitnessMagnitudeFloor << ",\n  \"guidedMutationRatio\": "
           << metadata.searchSettings.guidedMutationRatio
           << ",\n  \"maximumHotspots\": "
           << metadata.searchSettings.hotspots.maximumHotspots
           << ",\n  \"hotspotSuppressionFrames\": "
           << metadata.searchSettings.hotspots.suppressionFrames
           << ",\n  \"hotspotSuppressionBins\": "
           << metadata.searchSettings.hotspots.suppressionBins
           << ",\n  \"minimumHotspotResidual\": "
           << metadata.searchSettings.hotspots.minimumResidualMagnitude
           << ",\n  \"refinementBudgetRatio\": "
           << metadata.searchSettings.refinementBudgetRatio
           << ",\n  \"refinementParentCount\": "
           << metadata.searchSettings.refinementParentCount
           << ",\n  \"atomSelectionExplorationWeight\": "
           << metadata.searchSettings.atomSelectionExplorationWeight
           << ",\n  \"guidedFrequencyRadiusBins\": "
           << metadata.searchSettings.guidedFrequencyRadiusBins
           << ",\n  \"guidedTimeRadiusHops\": "
           << metadata.searchSettings.guidedTimeRadiusHops
           << ",\n  \"durationPersistenceThresholdRatio\": "
           << metadata.searchSettings.durationPersistenceThresholdRatio
           << ",\n  \"refinementFrequencyRadiusBins\": "
           << metadata.searchSettings.refinementFrequencyRadiusBins
           << ",\n  \"refinementTimeRadiusHops\": "
           << metadata.searchSettings.refinementTimeRadiusHops
           << ",\n  \"refinementDurationRatio\": "
           << metadata.searchSettings.refinementDurationRatio
           << ",\n  \"fftSize\": " << metadata.fftSize
           << ",\n  \"hopSize\": " << metadata.hopSize
           << ",\n  \"fitnessType\": \"log-magnitude-stft-mse\",\n"
           << "  \"mutationType\": \"editable-sine-genome\",\n"
           << "  \"initialFitness\": " << metadata.initialFitness
           << ",\n  \"finalFitness\": ";
    if (completion)
        output << completion->finalFitness;
    else
        output << "null";
    output << ",\n  \"targetFitness\": ";
    writeOptionalDouble(output, metadata.evolutionSettings.targetFitness);
    output << ",\n  \"stagnationGenerations\": ";
    writeOptionalSize(output, metadata.evolutionSettings.stagnationGenerations);
    output << ",\n  \"stopReason\": ";
    if (completion)
        writeJsonString(
            output, evolutionStopReasonValue(completion->stopReason));
    else
        output << "null";
    output << ",\n  \"status\": "
           << (completion ? "\"complete\"" : "\"running\"")
           << ",\n  \"searchStatistics\": {\n"
           << "    \"guidedCandidatesTested\": "
           << (completion ? completion->guidedCandidatesTested : 0) << ",\n"
           << "    \"randomCandidatesTested\": "
           << (completion ? completion->randomCandidatesTested : 0) << ",\n"
           << "    \"refinementCandidatesTested\": "
           << (completion ? completion->refinementCandidatesTested : 0) << ",\n"
           << "    \"guidedWinnersAccepted\": "
           << (completion ? completion->guidedWinnersAccepted : 0) << ",\n"
           << "    \"randomWinnersAccepted\": "
           << (completion ? completion->randomWinnersAccepted : 0) << ",\n"
           << "    \"refinementWinnersAccepted\": "
           << (completion ? completion->refinementWinnersAccepted : 0) << "\n"
           << "  }"
           << ",\n  \"sineMutation\": {\n"
           << "    \"minimumFrequencyHz\": "
           << metadata.mutationSettings.minimumFrequencyHz << ",\n"
           << "    \"maximumFrequencyHz\": "
           << metadata.mutationSettings.maximumFrequencyHz << ",\n"
           << "    \"minimumDurationSeconds\": "
           << metadata.mutationSettings.minimumDurationSeconds << ",\n"
           << "    \"maximumDurationSeconds\": "
           << metadata.mutationSettings.maximumDurationSeconds << ",\n"
           << "    \"maximumAbsoluteAmplitude\": "
           << metadata.mutationSettings.maximumAbsoluteAmplitude << "\n"
           << "  },\n  \"genomeMutation\": {\n"
           << "    \"addWeight\": " << metadata.genomeMutationSettings.addWeight
           << ",\n"
           << "    \"removeWeight\": "
           << metadata.genomeMutationSettings.removeWeight << ",\n"
           << "    \"moveWeight\": "
           << metadata.genomeMutationSettings.moveWeight << ",\n"
           << "    \"frequencyWeight\": "
           << metadata.genomeMutationSettings.frequencyWeight << ",\n"
           << "    \"amplitudeWeight\": "
           << metadata.genomeMutationSettings.amplitudeWeight << ",\n"
           << "    \"durationWeight\": "
           << metadata.genomeMutationSettings.durationWeight << ",\n"
           << "    \"phaseWeight\": "
           << metadata.genomeMutationSettings.phaseWeight << ",\n"
           << "    \"maximumMoveSeconds\": "
           << metadata.genomeMutationSettings.maximumMoveSeconds << ",\n"
           << "    \"maximumFrequencyRatio\": "
           << metadata.genomeMutationSettings.maximumFrequencyRatio << ",\n"
           << "    \"maximumAmplitudeDelta\": "
           << metadata.genomeMutationSettings.maximumAmplitudeDelta << ",\n"
           << "    \"maximumDurationDeltaSeconds\": "
           << metadata.genomeMutationSettings.maximumDurationDeltaSeconds
           << ",\n"
           << "    \"maximumPhaseDeltaRadians\": "
           << metadata.genomeMutationSettings.maximumPhaseDeltaRadians << "\n"
           << "  }\n}\n";
    finishMetadataFile(output, path);
}

void writeMorphRunMetadata(
    const std::filesystem::path& path, const MorphRunMetadata& metadata)
{
    std::ofstream output = openMetadataFile(path);
    output << "{\n  \"mode\": \"spectral-morph\",\n  \"strategy\": ";
    writeJsonString(output, morphStrategyValue(metadata.settings.strategy));
    output << ",\n  \"morphSteps\": " << metadata.settings.morphSteps;
    if (metadata.settings.strategy == MorphStrategy::PolarInterpolate
        || metadata.settings.strategy == MorphStrategy::MagnitudeSourcePhase
        || metadata.settings.strategy == MorphStrategy::MagnitudeTargetPhase) {
        output << ",\n  \"interpolationDynamicRangeDb\": "
               << metadata.settings.interpolationDynamicRangeDb;
    }
    if (metadata.settings.strategy == MorphStrategy::Replace) {
        output << ",\n  \"replacementOrder\": ";
        writeJsonString(
            output, replacementOrderValue(metadata.settings.replacementOrder));
        output << ",\n  \"replacementDynamicRangeDb\": "
               << metadata.settings.replacementDynamicRangeDb;
        if (metadata.settings.replacementOrder == ReplacementOrder::Random)
            output << ",\n  \"seed\": " << metadata.settings.replacementSeed;
    }
    output << ",\n  \"fftSize\": " << metadata.fftSize
           << ",\n  \"hopSize\": " << metadata.hopSize << ",\n  \"source\": ";
    writeJsonString(output, metadata.sourcePath);
    output << ",\n  \"target\": ";
    writeJsonString(output, metadata.targetPath);
    output << ",\n  \"preConversionSource\": ";
    if (metadata.preConversionSourcePath)
        writeJsonString(output, *metadata.preConversionSourcePath);
    else
        output << "null";
    output << ",\n  \"preConversionTarget\": ";
    if (metadata.preConversionTargetPath)
        writeJsonString(output, *metadata.preConversionTargetPath);
    else
        output << "null";
    output << ",\n  \"sampleRateHz\": " << metadata.sampleRateHz
           << ",\n  \"sampleCount\": " << metadata.sampleCount
           << ",\n  \"status\": \"complete\"\n}\n";
    finishMetadataFile(output, path);
}

void writeMorphGenerationMetadata(
    const std::filesystem::path& path, const MorphGeneration& generation)
{
    std::ofstream output = openMetadataFile(path);
    output << "{\n  \"generation\": " << generation.generation
           << ",\n  \"mode\": \"spectral-morph\",\n"
           << "  \"strategy\": ";
    writeJsonString(output, morphStrategyValue(generation.strategy));
    if (generation.strategy == MorphStrategy::Replace) {
        output << ",\n  \"replacementOrder\": ";
        writeJsonString(
            output, replacementOrderValue(generation.replacementOrder));
    }
    output << ",\n  \"progress\": " << generation.progress;
    if (generation.strategy == MorphStrategy::Replace) {
        output << ",\n  \"cellsReplaced\": " << generation.cellsReplaced
               << ",\n  \"totalCells\": " << generation.totalCells;
    }
    output << "\n}\n";
    finishMetadataFile(output, path);
}

void writeInitialGenerationMetadata(
    const std::filesystem::path& path, double fitness)
{
    std::ofstream output = openMetadataFile(path);
    output << "{\n  \"generation\": 0,\n  \"fitness\": " << fitness
           << ",\n  \"improved\": false,\n  \"candidatesTested\": 0,\n"
           << "  \"mutations\": []\n}\n";
    finishMetadataFile(output, path);
}

void writeGenerationMetadata(
    const std::filesystem::path& path, const GenerationSummary& summary)
{
    if (summary.generation == 0 || !summary.previousFitness)
        throw std::invalid_argument(
            "Evolved generation metadata requires a previous fitness.");
    std::ofstream output = openMetadataFile(path);
    output
        << "{\n  \"generation\": " << summary.generation
        << ",\n  \"previousFitness\": " << *summary.previousFitness
        << ",\n  \"fitness\": " << summary.fitness
        << ",\n  \"improvement\": " << summary.improvement
        << ",\n  \"improved\": " << (summary.improved ? "true" : "false")
        << ",\n  \"candidatesTested\": " << summary.candidatesTested
        << ",\n  \"guidedCandidatesTested\": " << summary.guidedCandidatesTested
        << ",\n  \"randomCandidatesTested\": " << summary.randomCandidatesTested
        << ",\n  \"refinementCandidatesTested\": "
        << summary.refinementCandidatesTested
        << ",\n  \"guidedWinnersAccepted\": " << summary.guidedWinnersAccepted
        << ",\n  \"randomWinnersAccepted\": " << summary.randomWinnersAccepted
        << ",\n  \"refinementWinnersAccepted\": "
        << summary.refinementWinnersAccepted << ",\n  \"mutations\": [";
    if (!summary.winningMutations.empty())
        output << '\n';
    for (std::size_t index = 0; index < summary.winningMutations.size();
         ++index) {
        output << "    ";
        writeMutation(output, summary.winningMutations[index]);
        output << (index + 1 == summary.winningMutations.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    finishMetadataFile(output, path);
}

void writeGenomeMetadata(
    const std::filesystem::path& path, const Genome& genome)
{
    std::ofstream output = openMetadataFile(path);
    output << "{\n  \"atoms\": [";
    if (!genome.atoms.empty())
        output << '\n';
    for (std::size_t index = 0; index < genome.atoms.size(); ++index) {
        const GenomeAtom& atom = genome.atoms[index];
        output << "    {\n      \"atomId\": " << atom.id << ",\n";
        writeAtomFields(output, atom.atom, "      ");
        output << "\n    }"
               << (index + 1 == genome.atoms.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    finishMetadataFile(output, path);
}
