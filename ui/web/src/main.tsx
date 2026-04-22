/**
 * Entry point for mud-puppy-studio web frontend.
 *
 * Root ErrorBoundary wraps the entire app per Forge lesson:
 * a render throw without it leaves <div id="root" /> blank.
 */
import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./ErrorBoundary";
import "./styles/tokens.css";
import "./styles/index.css";

const container = document.getElementById("root");
if (!container) {
  throw new Error("Root element #root not found in DOM");
}

createRoot(container).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
