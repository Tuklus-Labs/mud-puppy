#include "checkpoint_scan.hpp"

#include <phos/phos.h>

#include <sys/stat.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace mp_studio {

namespace fs = std::filesystem;

// Get mtime of a path as a Unix timestamp (C++17-safe, no file_clock::to_sys).
static long path_mtime(const fs::path& p) {
    struct stat st{};
    if (::stat(p.c_str(), &st) == 0) {
        return static_cast<long>(st.st_mtime);
    }
    return 0;
}

// Read trainer_state.json and extract step/loss/save_time if present.
static nlohmann::json read_trainer_state(const fs::path& checkpoint_dir) {
    auto state_path = checkpoint_dir / "trainer_state.json";
    if (!fs::exists(state_path)) return nlohmann::json::object();

    try {
        std::ifstream f(state_path);
        auto state = nlohmann::json::parse(f);

        nlohmann::json result;
        result["step"] = state.value("global_step", 0);

        // Last loss from log_history.
        if (state.contains("log_history") && state["log_history"].is_array() &&
            !state["log_history"].empty()) {
            const auto& last = state["log_history"].back();
            if (last.contains("loss")) result["loss"] = last["loss"];
            if (last.contains("eval_loss")) result["eval_loss"] = last["eval_loss"];
        }
        if (state.contains("best_model_checkpoint")) {
            result["best"] = (state["best_model_checkpoint"].get<std::string>() ==
                              checkpoint_dir.string());
        }

        return result;
    } catch (const std::exception& e) {
        PHOS_LOG_WARN("checkpoint_scan: failed to parse trainer_state.json in {}: {}",
                      checkpoint_dir.string(), e.what());
        return nlohmann::json::object();
    }
}

nlohmann::json scan_checkpoints(const std::string& output_dir) {
    nlohmann::json result = nlohmann::json::array();
    fs::path root(output_dir);

    if (!fs::exists(root)) {
        PHOS_LOG_WARN("checkpoint_scan: output_dir does not exist: {}", output_dir);
        return result;
    }

    // Enumerate checkpoint-* directories.
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) continue;
        const auto& name = entry.path().filename().string();
        if (name.rfind("checkpoint-", 0) != 0) continue;

        nlohmann::json item;
        item["path"] = entry.path().string();
        item["name"] = name;
        item["type"] = "checkpoint";
        item["mtime"] = path_mtime(entry.path());

        // Merge trainer_state data.
        auto state = read_trainer_state(entry.path());
        item.update(state);

        result.push_back(std::move(item));
    }

    // Also enumerate lora_library/ if it exists inside output_dir.
    fs::path lora_lib = root / "lora_library";
    if (fs::exists(lora_lib) && fs::is_directory(lora_lib)) {
        for (const auto& entry : fs::directory_iterator(lora_lib)) {
            if (!entry.is_directory()) continue;
            nlohmann::json item;
            item["path"] = entry.path().string();
            item["name"] = entry.path().filename().string();
            item["type"] = "lora_adapter";
            item["mtime"] = path_mtime(entry.path());
            result.push_back(std::move(item));
        }
    }

    PHOS_LOG_INFO("checkpoint_scan: output_dir={} found {} items", output_dir, result.size());
    return result;
}

}  // namespace mp_studio
