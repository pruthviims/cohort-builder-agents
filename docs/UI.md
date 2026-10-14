# Analytics workspace (web UI)

The workspace is a Streamlit application (`src/cohort_builder/ui/`) that runs **on top of the HTTP API**.
It holds no business logic and no database connection: every read and every action goes through the API.
The same governance as the CLI and API therefore applies unchanged:

- bearer-token authentication, roles and tenant scope;
- deterministic validation and compilation;
- human approval;
- execution gates;
- small-cell suppression;
- the audit log.

## What it does

| Page | What it uses | Notes |
|---|---|---|
| Dashboard | `/cohorts`, `/status`, `/datasets`, `/runs`, `/audit` (admin) | Counts come from API responses. A panel your role cannot read is left out, not shown as zero. |
| AI Cohort Builder | `/cohorts/ask`, `/cohorts/{id}/refine`, `/cohorts`, `/cohorts/validate`, `/cohorts/{id}/versions` | Conversation on the left and the definition on the right, for the same draft. |
| Cohort Library | `/cohorts`, `/cohorts/{id}` | Search, status and dataset filters. Open a definition in the builder, validation, review or explorer. |
| Validate & Preview | `/cohorts/validate`, `/cohorts/{id}/generations` | A dry run on the active dataset, with suppressed attrition. It is not an execution. |
| Review & Approve | `/cohorts/{id}`, `/sql`, `/reviews`, `/review`, `/execute` | Approval covers one exact version. Rejections need a comment. Execution is blocked until the server reports the version approved. |
| SQL & Ontology Explorer | `/cohorts/{id}/sql`, `/concepts/{id}`, `/concepts/search`, `/status` | SQL preview from the compiler, with download. Nothing is executed here. |
| Run History & Replay | `/runs`, `/runs/{id}`, `/runs/{id}/replay`, `/versions` | Lists the dependencies that changed since a run. |
| Dataset Management | `/datasets`, `/datasets/csv/profile`, `/datasets/{name}/ingest` (admin) | Imports run a dry run first and need an explicit confirmation. |
| Cohort Comparison | `/cohorts/{id}/diff/{other}`, `/cohorts/{id}/generations`, `/cohorts/compare` | Shows field differences between versions. Overlap is shown only for executed versions on the same dataset (suppressed). |
| System Health | `/health`, `/status`, `/versions`, `/audit` (admin) | The server runs each check when the page loads. |

### How the conversation works

- **First message:** the existing agent pipeline (intent parser → concept resolver → validator → critic)
  turns it into a definition, saved as a draft (`POST /cohorts/ask`).
- **Change requests** (*"change the HbA1c threshold to 9%"*) go to `POST /cohorts/{id}/refine`:
  - The server sends the same pipeline the request that produced the draft, followed by `Revision: ...`.
  - For a version that was saved from the structured editor, the server sends the version's
    deterministic explanation instead.
  - The result is a **new version** whose parent is the old one. It never inherits an approval.
  - The response lists every field that changed (`changes`), and the chat shows that list. Any change
    to an unrelated criterion is therefore visible, not silent.
- **Questions** (*"why are patients with type 1 diabetes excluded?"*) are answered from the stored
  definition, without the AI model:
  - the matching criteria, their concept sets, and the assumptions recorded when the definition was built;
  - the answer is labelled as coming from the stored definition;
  - it never adds clinical rationale that the definition does not record.
- **When the model is unavailable** (no API key, no cached answer), the chat says that no definition was
  created and shows the server's message. Opening, editing, validating, reviewing and executing
  definitions keep working. You can also start without the AI by uploading a definition in the IR JSON
  format (for example `examples/t2dm_metformin_hba1c.json`).

### How the structured editor works

- **What you can edit:** name, description, age limits, prior observation, follow-up, cohort exit, and
  for each criterion its name, occurrence and count, window, threshold, or removal.
- **What you cannot edit:** the index event and the concept sets. Change those through the conversation,
  because concept resolution belongs to the resolver.
- **Validate changes** runs the server's validation and dry run on the edited copy. Nothing is saved.
- **Save as new version** creates a child version. The server validates it, and a definition with errors
  is stored as `needs_review`.
- **Stale edits are refused:** if someone else already created a version from the one you are editing,
  saving is blocked until you open the latest version.
- **Renaming:** if you change a threshold but the criterion's name still quotes the old number, you are
  warned. Names are labels; the server checks the logic, not the wording.

## Windows setup (PowerShell)

Prerequisite: Python 3.11, 3.12 or 3.13 from python.org, with "Add to PATH" ticked during installation.

```powershell
git clone https://github.com/pruthviims/cohort-builder-agents.git
cd cohort-builder-agents
py -3.12 -m venv .venv
# if activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[api,ui]"

# 1. synthetic demo data (vocabulary + OMOP-style patients + LAAD-style claims); ~1 minute
cohort-builder init-demo

# 2. tokens: one per person/role. Each token is printed ONCE; copy it somewhere safe.
New-Item -ItemType Directory -Force config | Out-Null
cohort-builder auth issue-token --subject alice --roles author,viewer --tokens-file config\tokens.yaml
cohort-builder auth issue-token --subject rita --roles reviewer,viewer --tokens-file config\tokens.yaml
cohort-builder auth issue-token --subject eddie --roles executor,viewer --tokens-file config\tokens.yaml

# 3. terminal 1: the API
$env:CB_AUTH_TOKENS_FILE = "$PWD\config\tokens.yaml"
cohort-builder serve                      # http://127.0.0.1:8000

# 4. terminal 2 (activate the venv again): the workspace
.\.venv\Scripts\Activate.ps1
cohort-builder ui                         # http://127.0.0.1:8501 , sign in with a token
```

Keep `config\tokens.yaml` out of version control. It holds only token hashes, but it maps identities to
roles.

**Optional: AI features.** Set the key in the API's terminal before `cohort-builder serve`:
`$env:ANTHROPIC_API_KEY = "<your key>"`. Never put it in a file inside the repository. Without a key the
workspace still opens, and every deterministic workflow works.

**Linux / macOS:** the same steps, with `python3 -m venv .venv`, `source .venv/bin/activate` and
`export CB_AUTH_TOKENS_FILE=$PWD/config/tokens.yaml`.

## Configuration

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `CB_UI_API_URL` | workspace | `http://127.0.0.1:8000` | address of the API (`cohort-builder ui --api-url` overrides it) |
| `CB_UI_API_TOKEN` | workspace | none | sign in automatically with this token (single-user local use only) |
| `CB_AUTH_TOKENS_FILE` | API | none (everything returns 401) | hashed token file |
| `CB_DATASET` | API | `omop_demo` | the active dataset; restart the API to switch |
| `ANTHROPIC_API_KEY` | API | none | enables the AI model (optional) |

`cohort-builder ui` options:
- `--host` (default `127.0.0.1`, this machine only) and `--port` (default 8501);
- Streamlit's usage statistics are switched off.

## Security notes

- **The token** lives in the server-side Streamlit session. It is never written to the page, URLs, logs
  or browser storage, and error messages never echo it. Tests check this.
- **Permissions:** the UI hides actions your roles cannot use, but enforcement stays in the API. A hidden
  button is a convenience, not a control.
- **Patient-level data** is not displayed. Counts are the API's suppressed aggregates; suppressed values
  stay as text and are never charted as numbers.
- **The UI does not encrypt its own traffic.** By default it listens on 127.0.0.1 only. To serve other
  machines, put it behind a TLS-terminating reverse proxy with authentication.

## Tests

```powershell
pytest tests/test_ui_workspace.py tests/test_api_workspace.py
```

These tests drive the real pages with Streamlit's `AppTest`, against the real API and database in the
same process. Only the LLM is a scripted fake. They cover:

- sign-in, including a failed sign-in;
- multi-turn chat (create, refine, ask);
- an honest model failure;
- structured edits and stale-edit refusal;
- validation;
- self-approval refusal and rejection comments;
- execution gating;
- replay, comparison, role-restricted panels;
- secret leakage.

## Known limitations

- **Windows:** the PowerShell steps are written for Windows, but the workspace has only been run and
  tested on Linux.
- **One dataset per server:** the API serves one active dataset (`CB_DATASET`). The workspace shows it and
  lists the others, but cannot switch the server.
- **Conversation history** lives in the browser session. Leaving or reloading the page starts a new
  conversation; the versions themselves are stored on the server.
- **Refinement:** each revision re-runs the AI interpretation on the accumulated request. The diff shows
  any unrelated change, but nothing stops one from happening; review the diff.
- **Concept sets and the index event** can only be changed through the conversation.
- **No user administration:** tokens are issued with the CLI.
- **Overlap** is available only for executed versions on the same dataset.
- **Charts:** the attrition chart is drawn only when no step is suppressed.
