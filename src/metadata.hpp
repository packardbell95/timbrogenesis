#pragma once

#include "evolution.hpp"
#include "mutation.hpp"
#include "spectral_morph.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

struct RunMetadata {
    std::string sourcePath;
    std::string targetPath;
    std::optional<std::string> preConversionSourcePath;
    std::optional<std::string> preConversionTargetPath;
    std::uint32_t sampleRateHz = 0;
    std::size_t sampleCount = 0;
    RandomSeed seed = 0;
    EvolutionSettings evolutionSettings;
    SineMutationSettings mutationSettings;
    GenomeMutationSettings genomeMutationSettings;
    GuidedSearchSettings searchSettings;
    double fitnessDynamicRangeDb = 80.0;
    double fitnessMagnitudeFloor = 1e-12;
    std::size_t fftSize = 0;
    std::size_t hopSize = 0;
    double initialFitness = 0.0;
};

struct RunCompletion {
    std::size_t completedGenerations = 0;
    std::size_t successfulWrites = 0;
    double finalFitness = 0.0;
    EvolutionStopReason stopReason = EvolutionStopReason::MaximumGenerations;
    std::size_t guidedCandidatesTested = 0;
    std::size_t randomCandidatesTested = 0;
    std::size_t refinementCandidatesTested = 0;
    std::size_t guidedWinnersAccepted = 0;
    std::size_t randomWinnersAccepted = 0;
    std::size_t refinementWinnersAccepted = 0;
};

struct MorphRunMetadata {
    std::string sourcePath;
    std::string targetPath;
    std::optional<std::string> preConversionSourcePath;
    std::optional<std::string> preConversionTargetPath;
    std::uint32_t sampleRateHz = 0;
    std::size_t sampleCount = 0;
    SpectralMorphSettings settings;
    std::size_t fftSize = 0;
    std::size_t hopSize = 0;
};

std::string formatGenerationFilename(
    std::size_t generation, std::string_view extension);

void writeRunMetadata(const std::filesystem::path& path,
    const RunMetadata& metadata,
    const std::optional<RunCompletion>& completion = std::nullopt);

void writeMorphRunMetadata(
    const std::filesystem::path& path, const MorphRunMetadata& metadata);

void writeMorphGenerationMetadata(
    const std::filesystem::path& path, const MorphGeneration& generation);

void writeInitialGenerationMetadata(
    const std::filesystem::path& path, double fitness);

void writeGenerationMetadata(
    const std::filesystem::path& path, const GenerationSummary& summary);

void writeGenomeMetadata(
    const std::filesystem::path& path, const Genome& genome);
