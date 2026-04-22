#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace mp_studio {

// Read first n lines of a JSONL file and detect its format.
// Returns { "format": string, "rows": [...] }
nlohmann::json dataset_peek(const std::string& path, int n = 5);

}  // namespace mp_studio
