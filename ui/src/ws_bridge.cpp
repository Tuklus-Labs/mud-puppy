#include "ws_bridge.hpp"

#include <phos/phos.h>

#include <chrono>
#include <set>
#include <stdexcept>

namespace mp_studio {

// Context passed through libsoup async callbacks.
struct WsConnCtx {
    WsBridge* bridge;
    WsBridge::RunConn* conn;
};

WsBridge::WsBridge(phos::Window& win) : win_(win) {}

WsBridge::~WsBridge() {
    // Disconnect all runs.
    std::vector<std::string> ids;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (auto& [id, _] : conns_) ids.push_back(id);
    }
    for (const auto& id : ids) disconnect(id);
}

void WsBridge::connect(const std::string& run_id, int port) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (conns_.count(run_id)) return;  // already connected

    auto* conn = new RunConn();
    conn->run_id = run_id;
    conn->port = port;
    conn->ctx = g_main_context_new();
    conn->loop = g_main_loop_new(conn->ctx, FALSE);
    conn->session = nullptr;
    conn->ws_conn = nullptr;

    conns_[run_id] = conn;

    // Start the GMainLoop on a dedicated thread.
    conn->loop_thread = std::thread([this, conn]() {
        run_loop(conn);
    });
}

void WsBridge::disconnect(const std::string& run_id) {
    RunConn* conn = nullptr;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = conns_.find(run_id);
        if (it == conns_.end()) return;
        conn = it->second;
        conns_.erase(it);
    }

    conn->stopping.store(true);

    // Close WS connection if open.
    if (conn->ws_conn) {
        soup_websocket_connection_close(conn->ws_conn, SOUP_WEBSOCKET_CLOSE_NORMAL, nullptr);
        g_object_unref(conn->ws_conn);
        conn->ws_conn = nullptr;
    }

    // Stop the GMainLoop.
    g_main_loop_quit(conn->loop);

    if (conn->loop_thread.joinable()) {
        conn->loop_thread.join();
    }

    if (conn->session) g_object_unref(conn->session);
    g_main_loop_unref(conn->loop);
    g_main_context_unref(conn->ctx);
    delete conn;
}

void WsBridge::run_loop(RunConn* conn) {
    g_main_context_push_thread_default(conn->ctx);

    conn->session = soup_session_new();

    // Schedule the first connection attempt after 500ms (sidecar may be starting).
    auto* ctx_data = new WsConnCtx{this, conn};

    GSource* timer = g_timeout_source_new(500);
    g_source_set_callback(timer, [](gpointer data) -> gboolean {
        auto* cd = reinterpret_cast<WsConnCtx*>(data);
        if (cd->conn->stopping.load()) { delete cd; return G_SOURCE_REMOVE; }

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
            WsBridge::on_ws_connected, cd);
        g_object_unref(msg);
        return G_SOURCE_REMOVE;
    }, ctx_data, nullptr);
    g_source_attach(timer, conn->ctx);
    g_source_unref(timer);

    // Run until disconnect() quits the loop.
    g_main_loop_run(conn->loop);

    g_main_context_pop_thread_default(conn->ctx);
}

void WsBridge::on_ws_connected(GObject* src, GAsyncResult* res, gpointer user_data) {
    auto* cd = reinterpret_cast<WsConnCtx*>(user_data);
    if (cd->conn->stopping.load()) { delete cd; return; }

    GError* err = nullptr;
    SoupWebsocketConnection* ws =
        soup_session_websocket_connect_finish(SOUP_SESSION(src), res, &err);

    if (err) {
        PHOS_LOG_WARN("WsBridge: connect failed for run_id={}: {}", cd->conn->run_id, err->message);
        g_error_free(err);

        if (!cd->conn->stopping.load()) {
            // Retry after 2 seconds.
            GSource* timer = g_timeout_source_new(2000);
            g_source_set_callback(timer, [](gpointer data) -> gboolean {
                auto* cd2 = reinterpret_cast<WsConnCtx*>(data);
                if (cd2->conn->stopping.load()) { delete cd2; return G_SOURCE_REMOVE; }
                std::string uri_str = "ws://127.0.0.1:" + std::to_string(cd2->conn->port) + "/ws";
                SoupMessage* msg = soup_message_new("GET", uri_str.c_str());
                if (msg) {
                    soup_session_websocket_connect_async(
                        cd2->conn->session, msg, nullptr, nullptr,
                        G_PRIORITY_DEFAULT, nullptr,
                        WsBridge::on_ws_connected, cd2);
                    g_object_unref(msg);
                } else {
                    delete cd2;
                }
                return G_SOURCE_REMOVE;
            }, cd, nullptr);
            g_source_attach(timer, cd->conn->ctx);
            g_source_unref(timer);
        } else {
            delete cd;
        }
        return;
    }

    PHOS_LOG_INFO("WsBridge: connected to run_id={}", cd->conn->run_id);
    cd->conn->ws_conn = ws;

    g_signal_connect(ws, "message", G_CALLBACK(on_ws_message), cd);
    g_signal_connect(ws, "closed",  G_CALLBACK(on_ws_closed),  cd);
    // cd ownership transferred to signal callbacks; freed in on_ws_closed.
}

void WsBridge::on_ws_message(SoupWebsocketConnection* /*ws*/,
                              SoupWebsocketDataType type,
                              GBytes* message, gpointer user_data) {
    if (type != SOUP_WEBSOCKET_DATA_TEXT) return;

    auto* cd = reinterpret_cast<WsConnCtx*>(user_data);

    gsize len = 0;
    const gchar* data = reinterpret_cast<const gchar*>(g_bytes_get_data(message, &len));
    if (!data || len == 0) return;

    std::string text(data, len);

    try {
        auto frame = nlohmann::json::parse(text);
        cd->bridge->forward_frame(cd->conn->run_id, frame);
    } catch (const nlohmann::json::exception& e) {
        PHOS_LOG_WARN("WsBridge: JSON parse error for run_id={}: {}", cd->conn->run_id, e.what());
    }
}

void WsBridge::on_ws_closed(SoupWebsocketConnection* ws, gpointer user_data) {
    auto* cd = reinterpret_cast<WsConnCtx*>(user_data);
    PHOS_LOG_INFO("WsBridge: WS closed for run_id={}", cd->conn->run_id);
    g_object_unref(ws);
    cd->conn->ws_conn = nullptr;
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

}  // namespace mp_studio
