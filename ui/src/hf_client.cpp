#include "hf_client.hpp"

#include <phos/phos.h>

#include <curl/curl.h>

#include <cstring>
#include <stdexcept>
#include <sstream>

namespace mp_studio {

HfClient::HfClient() {
    const char* tok = ::getenv("HF_TOKEN");
    if (tok) hf_token_ = tok;
}

size_t HfClient::write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = reinterpret_cast<std::string*>(userdata);
    buf->append(ptr, size * nmemb);
    return size * nmemb;
}

nlohmann::json HfClient::search_models(const std::string& query) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("curl_easy_init failed");
    }

    // URL-encode the query.
    char* escaped = curl_easy_escape(curl, query.c_str(), static_cast<int>(query.size()));
    std::string url = "https://huggingface.co/api/models?search=";
    url += escaped;
    url += "&limit=20&sort=downloads&direction=-1";
    curl_free(escaped);

    std::string response_body;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "mud-puppy-studio/0.4.0");

    // Add HF auth if token present.
    struct curl_slist* headers = nullptr;
    std::string auth_header;
    if (!hf_token_.empty()) {
        auth_header = "Authorization: Bearer " + hf_token_;
        headers = curl_slist_append(headers, auth_header.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }

    CURLcode res = curl_easy_perform(curl);

    if (headers) curl_slist_free_all(headers);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl error: ") + curl_easy_strerror(res));
    }

    if (http_code != 200) {
        throw std::runtime_error("HF API returned HTTP " + std::to_string(http_code));
    }

    auto full = nlohmann::json::parse(response_body);
    if (!full.is_array()) {
        throw std::runtime_error("HF API response is not an array");
    }

    // Project each model to only the fields we need.
    nlohmann::json result = nlohmann::json::array();
    for (const auto& m : full) {
        nlohmann::json item;
        item["id"]            = m.value("modelId", m.value("id", ""));
        item["downloads"]     = m.value("downloads", 0);
        item["likes"]         = m.value("likes", 0);
        item["pipeline_tag"]  = m.value("pipeline_tag", "");
        item["tags"]          = m.value("tags", nlohmann::json::array());
        result.push_back(std::move(item));
    }

    PHOS_LOG_INFO("HfClient: search '{}' returned {} models", query, result.size());
    return result;
}

}  // namespace mp_studio
