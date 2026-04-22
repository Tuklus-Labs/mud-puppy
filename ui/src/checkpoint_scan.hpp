#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace mp_studio {

// Enumerate checkpoints in output_dir (checkpoint-* subdirs) and LoRA adapters.
// Returns JSON array of checkpoint descriptors.
nlohmann::json scan_checkpoints(const std::string& output_dir);

}  // namespace mp_studio
