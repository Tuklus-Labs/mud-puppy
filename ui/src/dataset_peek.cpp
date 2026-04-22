#include "dataset_peek.hpp"

#include <phos/phos.h>

#include <fstream>
#include <stdexcept>
#include <string>

namespace mp_studio {

// Classify format based on the keys present in the first row.
static std::string detect_format(const nlohmann::json& first_row) {
    if (!first_row.is_object()) return "unknown";

    if (first_row.contains("messages")) return "messages";
    if (first_row.contains("prompt") && first_row.contains("chosen") &&
        first_row.contains("rejected")) return "preference";
    if (first_row.contains("prompt") && first_row.contains("completion")) return "prompt-completion";
    if (first_row.contains("instruction") && first_row.contains("response")) return "instruction-response";
    if (first_row.contains("text")) return "text";
    if (first_row.contains("prompt")) return "prompt-only";
    return "unknown";
}

nlohmann::json dataset_peek(const std::string& path, int n) {
    if (n <= 0) n = 5;

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open dataset file: " + path);
    }

    nlohmann::json rows = nlohmann::json::array();
    std::string line;
    int count = 0;
    std::string format = "unknown";

    while (count < n && std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            auto row = nlohmann::json::parse(line);
            if (count == 0) {
                format = detect_format(row);
            }
            rows.push_back(std::move(row));
            ++count;
        } catch (const nlohmann::json::exception& e) {
            PHOS_LOG_WARN("dataset_peek: JSON parse error at line {}: {}", count + 1, e.what());
        }
    }

    PHOS_LOG_INFO("dataset_peek: path={} format={} rows_returned={}", path, format, rows.size());

    return {{"format", format}, {"rows", rows}};
}

}  // namespace mp_studio
