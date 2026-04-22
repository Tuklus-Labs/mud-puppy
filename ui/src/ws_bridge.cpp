#include "ws_bridge.hpp"

#include <phos/phos.h>

#include <chrono>
#include <set>
#include <stdexcept>
#include <vector>

namespace mp_studio {

// Context passed through libsoup async callbacks.
//
// Lifetime rule: a single WsConnCtx travels through the chain
// attempt_connect -> on_ws_connected -> (signals). It is freed either:
//   - in attempt_connect / on_ws_connected when stopping is set or retries
//     exhaust (no further schedule), or
//   - in on_ws_closed (terminal state after a successful upgrade).
struct WsConnCtx {
    WsBridge* bridge;
    WsBridge::RunConn* conn;
};

WsBridge::WsBridge(phos::Window& win) : win_(win) {}

WsBridge::~WsBridge() {
    // Snapshot run ids under lock, then disconnect each (which joins).
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ids.reserve(conns_.size());
        for (auto& [id, _] : conns_) ids.push_back(id);
    }
    for (const auto& id : ids) disconnect(id);
}

void WsBridge::connect(const std::string& run_id, int port) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (conns_.count(run_id)) return;
    }

    auto conn = std::make_unique<RunConn>();
    conn->run_id = run_id;
    conn->port = port;
    conn->ctx = g_main_context_new();
    conn->loop = g_main_loop_new(conn->ctx, FALSE);
    conn->bridge = this;

    RunConn* raw = conn.get();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        conns_.emplace(run_id, std::move(conn));
    }

    // Start the GMainLoop on a dedicated thread. All mutations of raw->session,
    // raw->ws_conn, raw->retry_count happen ON this thread.
    raw->loop_thread = std::thread([this, raw]() { run_loop(raw); });
}

void WsBridge::disconnect(const std::string& run_id) {
    // Take ownership of the RunConn out of the map first. After this point
    // no other code path can reach it via conns_.
    std::unique_ptr<RunConn> conn;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = conns_.find(run_id);
        if (it == conns_.end()) return;
        conn = std::move(it->second);
        conns_.erase(it);
    }

    // Flag the loop so retry timers / pending callbacks see stopping=true.
    conn->stopping.store(true);

    RunConn* raw = conn.get();

    // Schedule close work ON the loop thread (the sole mutator of ws_conn /
    // session). g_main_context_invoke queues when called from outside the
    // context, runs inline when called from inside it.
    g_main_context_invoke(raw->ctx, [](gpointer data) -> gboolean {
        auto* c = static_cast<RunConn*>(data);
        if (c->ws_conn) {
            soup_websocket_connection_close(c->ws_conn,
                                            SOUP_WEBSOCKET_CLOSE_NORMAL, nullptr);
            g_object_unref(c->ws_conn);
            c->ws_conn = nullptr;
        }
        if (c->session) {
            g_object_unref(c->session);
            c->session = nullptr;
        }
        g_main_loop_quit(c->loop);
        return G_SOURCE_REMOVE;
    }, raw);

    // Join the loop thread. After join() returns no callbacks can fire.
    if (raw->loop_thread.joinable()) {
        raw->loop_thread.join();
    }

    g_main_loop_unref(raw->loop);
    g_main_context_unref(raw->ctx);
    // unique_ptr frees the RunConn here.
}

// ---------------------------------------------------------------------------
// Loop thread
// ---------------------------------------------------------------------------

void WsBridge::run_loop(RunConn* conn) {
    g_main_context_push_thread_default(conn->ctx);

    conn->session = soup_session_new();

    // Kick off the first connection attempt after a 500ms delay so the
    // sidecar has time to bind its port.
    auto* ctx_data = new WsConnCtx{this, conn};
    GSource* timer = g_timeout_source_new(500);
    g_source_set_callback(timer, &WsBridge::attempt_connect, ctx_data, nullptr);
    g_source_attach(timer, conn->ctx);
    g_source_unref(timer);

    g_main_loop_run(conn->loop);

    g_main_context_pop_thread_default(conn->ctx);
}

gboolean WsBridge::attempt_connect(gpointer user_data) {
    // Runs on the loop thread.
    auto* cd = static_cast<WsConnCtx*>(user_data);
    if (cd->conn->stopping.load()) {
        delete cd;
        return G_SOURCE_REMOVE;
    }

    if (cd->conn->retry_count >= WsBridge::MAX_RETRIES) {
        PHOS_LOG_WARN("WsBridge: retries exhausted for run_id={} ({} attempts)",
                      cd->conn->run_id, cd->conn->retry_count);
        cd->bridge->emit_retry_exhausted(cd->conn->run_id);
        delete cd;
        return G_SOURCE_REMOVE;
    }
    cd->conn->retry_count++;

    std::string uri_str = "ws://127.0.0.1:" + std::to_string(cd->conn->port) + "/ws";
    SoupMessage* msg = soup_message_new("GET", uri_str.c_str());
    if (!msg) {
        PHOS_LOG_WARN("WsBridge: invalid URI {}", uri_str);
        delete cd;
        return G_SOURCE_REMOVE;
    }

    soup_session_websocket_connect_async(
        cd->conn->session, msg, nullptr, nullptr,
        G_PRIORITY_DEFAULT, nullptr,
        &WsBridge::on_ws_connected, cd);

    g_object_unref(msg);
    return G_SOURCE_REMOVE;
}

void WsBridge::on_ws_connected(GObject* src, GAsyncResult* res, gpointer user_data) {
    // Runs on the loop thread.
    auto* cd = static_cast<WsConnCtx*>(user_data);

    GError* err = nullptr;
    SoupWebsocketConnection* ws =
        soup_session_websocket_connect_finish(SOUP_SESSION(src), res, &err);

    if (cd->conn->stopping.load()) {
        if (ws) g_object_unref(ws);
        if (err) g_error_free(err);
        delete cd;
        return;
    }

    if (err) {
        PHOS_LOG_WARN("WsBridge: connect attempt {} failed for run_id={}: {}",
                      cd->conn->retry_count, cd->conn->run_id, err->message);
        g_error_free(err);

        // Schedule a retry after 2s via attempt_connect (which enforces the cap).
        GSource* timer = g_timeout_source_new(2000);
        g_source_set_callback(timer, &WsBridge::attempt_connect, cd, nullptr);
        g_source_attach(timer, cd->conn->ctx);
        g_source_unref(timer);
        return;
    }

    PHOS_LOG_INFO("WsBridge: connected to run_id={}", cd->conn->run_id);
    cd->conn->ws_conn = ws;

    g_signal_connect(ws, "message", G_CALLBACK(&WsBridge::on_ws_message), cd);
    g_signal_connect(ws, "closed",  G_CALLBACK(&WsBridge::on_ws_closed),  cd);
    // cd ownership transfers to signal callbacks; freed in on_ws_closed.
}

void WsBridge::on_ws_message(SoupWebsocketConnection* /*ws*/,
                              SoupWebsocketDataType type,
                              GBytes* message, gpointer user_data) {
    if (type != SOUP_WEBSOCKET_DATA_TEXT) return;

    auto* cd = static_cast<WsConnCtx*>(user_data);

    gsize len = 0;
    const gchar* data = static_cast<const gchar*>(g_bytes_get_data(message, &len));
    if (!data || len == 0) return;

    std::string text(data, len);
    try {
        auto frame = nlohmann::json::parse(text);
        cd->bridge->forward_frame(cd->conn->run_id, frame);
    } catch (const nlohmann::json::exception& e) {
        PHOS_LOG_WARN("WsBridge: JSON parse error for run_id={}: {}",
                      cd->conn->run_id, e.what());
    }
}

void WsBridge::on_ws_closed(SoupWebsocketConnection* /*ws*/, gpointer user_data) {
    // Runs on the loop thread.
    auto* cd = static_cast<WsConnCtx*>(user_data);
    PHOS_LOG_INFO("WsBridge: WS closed for run_id={}", cd->conn->run_id);
    // Do NOT unref ws_conn here -- disconnect() owns that unref (via its
    // g_main_context_invoke callback). Just free our ctx.
    delete cd;
}

void WsBridge::forward_frame(const std::string& run_id, const nlohmann::json& frame) {
    static const std::set<std::string> known_events = {
        "metrics", "gpu", "stream_stats", "memory_stats", "log_line", "run_complete"
    };

    std::string event_type;
    if (frame.contains("type") && frame["type"].is_string()) {
        event_type = frame["type"].get<std::string>();
    }
    if (event_type.empty() || !known_events.count(event_type)) {
        event_type = "monitor";
    }

    auto payload = frame;
    if (!payload.contains("run_id")) {
        payload["run_id"] = run_id;
    }

    win_.emit(event_type, std::move(payload));
}

void WsBridge::emit_retry_exhausted(const std::string& run_id) {
    win_.emit("run_complete", {
        {"run_id", run_id},
        {"exit_code", -1},
        {"error", "WsBridge: retries exhausted; could not connect to monitor"}
    });
}

}  // namespace mp_studio
