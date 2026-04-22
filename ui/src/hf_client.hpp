#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace mp_studio {

// HfClient wraps the Hugging Face Hub REST API.
// Uses libcurl for HTTP. Thread-safe (each call creates its own CURL handle).
class HfClient {
public:
    HfClient();

    // Search for models matching query.
    // GET https://huggingface.co/api/models?search=<q>&limit=20
    // Returns JSON array of {id, downloads, likes, tags, pipeline_tag}.
    nlohmann::json search_models(const std::string& query);

private:
    std::string hf_token_;  // optional, from HF_TOKEN env

    static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata);
};

}  // namespace mp_studio
