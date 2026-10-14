// Researcher admin page (admin.html): sessions, per-session image order and data, and the planned
// image order of the fixed seed. Not linked from the participant experiment.
//
// Production build: data comes from admin_api.php after a password login.
// Vite dev server (no PHP): data comes from the sessions saved in this browser's localStorage.
//
// Routes: #/ (sessions) · #/session/<id> · #/preview · #/settings

import "./styles.css";
import "./visualization/plot.css";
import "./admin/admin.css";
import { h } from "./components/dom.js";
import { experimentConfig } from "./config/experimentConfig.js";
import { applySettings, settingsFromConfig } from "./config/experimentSettings.js";
import { AdminAuthError, LocalAdminSource, ServerAdminSource } from "./admin/dataSource.js";
import { renderSessionDetail } from "./admin/sessionDetail.js";
import { createSessionList } from "./admin/sessionList.js";
import { createSequencePreview } from "./admin/sequencePreview.js";
import { createSettingsView } from "./admin/settingsView.js";
import { loadPca } from "./scientific/pcaLoader.js";

const facesDirectory = experimentConfig.assets.facesDirectory;

function loginView(source, { notConfigured }) {
  return new Promise((resolve) => {
    const password = h("input", { type: "password", id: "admin-password", autocomplete: "current-password", required: true, disabled: notConfigured });
    const error = h("p", { class: "form-error", role: "alert" });
    const submit = h("button", { type: "submit", disabled: notConfigured }, "Sign in");
    const form = h(
      "form",
      {
        class: "login-card",
        onsubmit: async (event) => {
          event.preventDefault();
          submit.disabled = true;
          error.textContent = "";
          try {
            await source.login(password.value);
            resolve();
          } catch (e) {
            error.textContent = e.message;
            password.select();
          } finally {
            submit.disabled = false;
          }
        },
      },
      h("span", { class: "login-icon", "aria-hidden": "true" }),
      h("h1", { text: "Researcher sign-in" }),
      h("p", { class: "muted", text: "This page shows the collected study data. It is for the research team only." }),
      notConfigured
        ? h("p", { class: "warning", text: "No admin password is set on the server yet. See the README section “Admin page” to create one." })
        : null,
      h("label", { for: "admin-password", text: "Password" }),
      password,
      error,
      submit,
    );
    document.getElementById("admin").replaceChildren(h("div", { class: "login-wrap" }, form));
    password.focus();
  });
}

async function ensureLoggedIn(source) {
  if (!source.requiresLogin) return;
  const status = await source.status();
  if (!status.authenticated) await loginView(source, { notConfigured: !status.configured });
}

async function boot() {
  document.body.classList.add("admin");
  const root = document.getElementById("admin");
  root.replaceChildren(h("div", { class: "loading" }, h("span", { class: "spinner" }), h("p", { text: "Loading…" })));

  const source = import.meta.env.DEV ? new LocalAdminSource({ prefix: experimentConfig.storage.localStoragePrefix }) : new ServerAdminSource();
  let pca = null;
  const pcaReady = loadPca(experimentConfig)
    .then((value) => (pca = value))
    .catch(() => null); // only the plot and the planned order need it

  try {
    await ensureLoggedIn(source);
  } catch (error) {
    root.replaceChildren(h("div", { class: "login-wrap" }, h("div", { class: "login-card" }, h("h1", { text: "Admin unavailable" }), h("p", { class: "form-error", text: error.message }))));
    return;
  }

  // ---------------------------------------------------------------- shell
  const main = h("main", { class: "admin-main" });
  const banner = h("div", { class: "banner", role: "alert", hidden: true });
  const nav = {
    sessions: h("a", { href: "#/" }, "Sessions"),
    preview: h("a", { href: "#/preview" }, "Planned image order"),
    settings: h("a", { href: "#/settings" }, "Settings"),
  };
  const signOut = source.requiresLogin
    ? h(
        "button",
        {
          type: "button",
          class: "secondary small",
          onclick: async () => {
            await source.logout().catch(() => null);
            window.location.reload();
          },
        },
        "Sign out",
      )
    : null;
  root.replaceChildren(
    h(
      "header",
      { class: "topbar" },
      h("div", { class: "brand" }, h("strong", { text: "Drift experiment" }), h("span", { class: "muted", text: " · Admin" })),
      h("nav", {}, nav.sessions, nav.preview, nav.settings),
      h("span", { class: "source-label", text: source.label }),
      signOut,
    ),
    banner,
    main,
  );

  const showError = async (error) => {
    if (error instanceof AdminAuthError) {
      await loginView(source, { notConfigured: false });
      window.location.reload();
      return;
    }
    banner.textContent = error.message;
    banner.hidden = false;
  };

  const list = createSessionList({ source, onOpen: (id) => (window.location.hash = `#/session/${id}`), onError: showError });
  // The planned order follows the saved experiment settings (what the next session will use).
  let preview = null;
  let previewShown = false;
  const usePlannedConfig = (savedSettings) => {
    preview = createSequencePreview({ getPca: () => pca, config: applySettings(experimentConfig, savedSettings?.values), facesDirectory });
    previewShown = false;
  };
  usePlannedConfig(await source.getSettings().catch(() => null));
  const settingsView = createSettingsView({
    source,
    defaults: settingsFromConfig(experimentConfig),
    getPca: () => pca,
    onSaved: usePlannedConfig,
    onError: showError,
  });

  async function route() {
    banner.hidden = true;
    const hash = window.location.hash || "#/";
    const sessionMatch = hash.match(/^#\/session\/([\w-]+)$/);
    nav.sessions.classList.toggle("active", !hash.startsWith("#/preview") && !hash.startsWith("#/settings"));
    nav.preview.classList.toggle("active", hash.startsWith("#/preview"));
    nav.settings.classList.toggle("active", hash.startsWith("#/settings"));
    window.scrollTo(0, 0);

    if (sessionMatch) {
      main.replaceChildren(h("div", { class: "loading" }, h("span", { class: "spinner" }), h("p", { text: "Loading session…" })));
      await pcaReady;
      try {
        main.replaceChildren(
          await renderSessionDetail({
            source,
            sessionId: sessionMatch[1],
            pca,
            config: experimentConfig,
            facesDirectory,
            onBack: () => (window.location.hash = "#/"),
            onRefresh: () => route(),
          }),
        );
      } catch (error) {
        main.replaceChildren();
        showError(error);
      }
      return;
    }
    if (hash.startsWith("#/settings")) {
      main.replaceChildren(settingsView.element);
      await pcaReady;
      await settingsView.load(experimentConfig);
      return;
    }
    if (hash.startsWith("#/preview")) {
      main.replaceChildren(preview.element);
      if (!previewShown) {
        await pcaReady;
        preview.show();
        previewShown = true;
      }
      return;
    }
    main.replaceChildren(list.element);
    list.refresh();
  }

  window.addEventListener("hashchange", route);
  route();
}

boot();
