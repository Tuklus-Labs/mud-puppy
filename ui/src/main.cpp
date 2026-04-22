#include <phos/phos.h>

#include "training_manager.hpp"
#include "ws_bridge.hpp"
#include "hf_client.hpp"
#include "dataset_peek.hpp"
#include "checkpoint_scan.hpp"

#include <curl/curl.h>

#include <csignal>
#include <filesystem>
#include <unistd.h>

// Global pointers for SIGTERM handler.
static mp_studio::TrainingManager* g_tm = nullptr;
static mp_studio::WsBridge* g_ws = nullptr;

static void handle_sigterm(int) {
    // Clean shutdown: quit the GTK main loop which triggers destructors.
    // g_idle_add is async-signal-safe enough for this purpose.
    g_idle_add([](gpointer) -> gboolean {
        GApplication* gapp = g_application_get_default();
        if (gapp) g_application_quit(gapp);
        return G_SOURCE_REMOVE;
    }, nullptr);
}

int main(int /*argc*/, char** argv) {
    // Phos resolves asset_dir and manifest paths relative to the CWD.
    // To make the binary runnable from any directory, chdir to the
    // directory containing argv[0] (where manifest.toml and web/dist/
    // are staged by CMake).
    try {
        std::filesystem::path exe_path = std::filesystem::canonical(argv[0]);
        std::filesystem::current_path(exe_path.parent_path());
    } catch (const std::exception& exc) {
        // Non-fatal: fall back to whatever CWD the user supplied.
        PHOS_LOG_WARN("Could not chdir to binary directory: {}", exc.what());
    }

    // Ignore SIGPIPE (broken pipe from sidecar stderr).
    signal(SIGPIPE, SIG_IGN);

    // Install SIGTERM/SIGINT for clean shutdown of child processes.
    signal(SIGTERM, handle_sigterm);
    signal(SIGINT,  handle_sigterm);

    // libcurl requires one global init before any easy handle is created.
    if (curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
        PHOS_LOG_ERROR("curl_global_init failed");
        return 1;
    }

    auto manifest = phos::resolve_manifest();
    PHOS_LOG_INFO("mud-puppy-studio starting, manifest={}", manifest);

    phos::App app(manifest);
    auto& win = app.window(app.config().window);

    // Enable debug logging when devtools are on.
    if (app.config().window.devtools) {
        phos::log::set_level(phos::log::Level::Debug);
    }

    // Instantiate subsystems. WsBridge is constructed first so its ref
    // can be bound into the TrainingManager port-announcement callback.
    mp_studio::WsBridge ws(win);
    mp_studio::TrainingManager tm(win);
    mp_studio::HfClient hf;

    // When a child sidecar announces its bound monitor port (via the
    // stdout marker), connect the WsBridge. This replaces the older
    // pre-allocate-port-and-pass-it dance which had a TOCTOU race.
    tm.set_port_ready_callback(
        [&ws](const std::string& run_id, int port) {
            ws.connect(run_id, port);
        });

    // Store for SIGTERM handler (signal-safe pointers).
    g_tm = &tm;
    g_ws = &ws;

    // --- IPC handlers ---
    // Only handlers declared in manifest.toml are registered here.

    // run.start: spawn a training sidecar. The monitor port is not
    // known at return time; WsBridge is connected asynchronously via
    // the port_ready_cb once the child announces its bound port.
    win.handle("run.start", phos::safe_handler([&](const phos::Json& req) -> phos::Json {
        // Accept either { config: {...} } or the config object directly.
        const auto& config = req.contains("config") ? req["config"] : req;
        auto handle = tm.start(config);
        return {
            {"run_id", handle.run_id},
            {"pid", static_cast<int>(handle.pid)}
        };
    }));

    // run.stop: send SIGTERM to sidecar and disconnect WsBridge.
    win.handle("run.stop", phos::safe_handler([&](const phos::Json& req) -> phos::Json {
        std::string run_id = req.at("run_id").get<std::string>();
        ws.disconnect(run_id);
        bool ok = tm.stop(run_id);
        return phos::make_ok({{"stopped", ok}});
    }));

    // run.list: return all active runs as a JSON array.
    win.handle("run.list", phos::safe_handler([&](const phos::Json&) -> phos::Json {
        return tm.list();
    }));

    // hf.search_models: search HF Hub by query string.
    win.handle("hf.search_models", phos::safe_handler([&](const phos::Json& req) -> phos::Json {
        std::string query = req.at("query").get<std::string>();
        return hf.search_models(query);
    }));

    // checkpoint.list: enumerate checkpoints and LoRA adapters in output_dir.
    win.handle("checkpoint.list", phos::safe_handler([](const phos::Json& req) -> phos::Json {
        std::string output_dir = req.at("output_dir").get<std::string>();
        return mp_studio::scan_checkpoints(output_dir);
    }));

    // dataset.preview: read first N lines of a JSONL file and detect format.
    win.handle("dataset.preview", phos::safe_handler([](const phos::Json& req) -> phos::Json {
        std::string path = req.at("path").get<std::string>();
        int n = req.value("n", 5);
        return mp_studio::dataset_peek(path, n);
    }));

    int rc = app.run();

    // libcurl cleanup mirrors the init at startup.
    curl_global_cleanup();
    return rc;
}
