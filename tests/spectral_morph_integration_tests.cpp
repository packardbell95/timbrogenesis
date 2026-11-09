#include "audio.hpp"
#include "metadata.hpp"
#include "spectral_morph.hpp"
#include "spectrum.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory()
    {
        const auto suffix =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
            / ("spectral-morph-integration-" + std::to_string(suffix));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Could not read " + path.string());
    return {std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
}

AudioBuffer makeSource()
{
    AudioBuffer audio;
    audio.sampleRateHz = 8'000;
    audio.samples.resize(257);
    for (std::size_t index = 0; index < audio.samples.size(); ++index) {
        const int value = static_cast<int>(index % 9) - 4;
        audio.samples[index] = static_cast<double>(value) / 8.0;
    }
    return audio;
}

AudioBuffer makeTarget()
{
    AudioBuffer audio;
    audio.sampleRateHz = 8'000;
    audio.samples.resize(257);
    for (std::size_t index = 0; index < audio.samples.size(); ++index) {
        const int value = 3 - static_cast<int>(index % 7);
        audio.samples[index] = static_cast<double>(value) / 4.0;
    }
    return audio;
}

void requireSameAudio(const AudioBuffer& actual,
    const AudioBuffer& expected,
    const std::string& label)
{
    require(actual.sampleRateHz == expected.sampleRateHz,
        label + " sample rate changed.");
    require(actual.samples == expected.samples, label + " samples changed.");
}

void testTenStepPersistedMorph()
{
    const TemporaryDirectory directory;
    const AudioBuffer source = makeSource();
    const AudioBuffer target = makeTarget();
    const SpectralMorphSettings settings{
        .strategy = MorphStrategy::Interpolate,
        .morphSteps = 10,
    };
    constexpr std::size_t fftSize = 64;
    constexpr std::size_t hopSize = 16;
    ComplexStftAnalyzer analyzer(fftSize, hopSize);
    ComplexStftSynthesizer synthesizer(fftSize);

    runSpectralMorph(source,
        target,
        settings,
        analyzer,
        synthesizer,
        [&directory](
            const MorphGeneration& generation, const AudioBuffer& audio) {
        const auto wavPath = directory.path()
            / formatGenerationFilename(generation.generation, ".wav");
        const auto jsonPath = directory.path()
            / formatGenerationFilename(generation.generation, ".json");
        saveAudio(wavPath.string(), audio);
        writeMorphGenerationMetadata(jsonPath, generation);
    });

    writeMorphRunMetadata(directory.path() / "run.json",
        {
            .sourcePath = "source.wav",
            .targetPath = "target.wav",
            .preConversionSourcePath = std::nullopt,
            .preConversionTargetPath = std::nullopt,
            .sampleRateHz = source.sampleRateHz,
            .sampleCount = source.samples.size(),
            .settings = settings,
            .fftSize = fftSize,
            .hopSize = hopSize,
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
    require(wavCount == 11, "Ten steps did not produce eleven WAV files.");
    require(generationJsonCount == 11,
        "Ten steps did not produce eleven generation JSON files.");

    for (std::size_t generation = 0; generation <= 10; ++generation) {
        const auto wavPath =
            directory.path() / formatGenerationFilename(generation, ".wav");
        const auto jsonPath =
            directory.path() / formatGenerationFilename(generation, ".json");
        require(std::filesystem::exists(wavPath),
            "A numbered morph WAV is missing.");
        require(std::filesystem::exists(jsonPath),
            "A numbered morph metadata file is missing.");

        const AudioBuffer audio = loadAudio(wavPath.string());
        require(audio.sampleRateHz == source.sampleRateHz,
            "A morph output has the wrong sample rate.");
        require(audio.samples.size() == source.samples.size(),
            "A morph output has the wrong sample count.");
        for (double sample : audio.samples) {
            require(std::isfinite(sample),
                "A morph output contains a non-finite sample.");
        }

        const std::string generationJson = readFile(jsonPath);
        require(generationJson.find("\"mode\": \"spectral-morph\"")
                != std::string::npos,
            "Generation metadata omitted the mode.");
        require(
            generationJson.find("\"generation\": " + std::to_string(generation))
                != std::string::npos,
            "Generation metadata has the wrong generation number.");
    }

    requireSameAudio(
        loadAudio((directory.path() / "generation-000.wav").string()),
        source,
        "Source endpoint");
    requireSameAudio(
        loadAudio((directory.path() / "generation-010.wav").string()),
        target,
        "Target endpoint");

    const std::string runJson = readFile(directory.path() / "run.json");
    require(runJson.find("\"mode\": \"spectral-morph\"") != std::string::npos
            && runJson.find("\"strategy\": \"interpolate\"")
                != std::string::npos
            && runJson.find("\"morphSteps\": 10") != std::string::npos
            && runJson.find("\"fftSize\": 64") != std::string::npos
            && runJson.find("\"hopSize\": 16") != std::string::npos
            && runJson.find("\"sampleRateHz\": 8000") != std::string::npos
            && runJson.find("\"sampleCount\": 257") != std::string::npos
            && runJson.find("\"status\": \"complete\"") != std::string::npos,
        "Run metadata is incomplete.");
    require(runJson.find("candidatesPerGeneration") == std::string::npos
            && runJson.find("mutationType") == std::string::npos,
        "Spectral morph metadata contains evolution-only fields.");
}

}  // namespace

int main()
{
    try {
        testTenStepPersistedMorph();
        std::cout << "All spectral morph integration tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Spectral morph integration test failure: " << error.what()
                  << '\n';
        return 1;
    }
}
