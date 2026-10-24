# machinekind-twin-llm

An object detector for the digital twin that asks a vision model where things are in the camera
frames. The model sits behind a CLI installed on this machine, `claude` (Claude Code) or `codex`
(Codex CLI), and answers on the subscription that CLI is logged in with. The package holds no API
key, uses no model SDK and opens no network connection itself: every question is one headless run
of the CLI as a subprocess.

The images leave the machine. They go to Anthropic or OpenAI under the terms of the subscription
the CLI is logged in with.

The package depends on `machinekind-twin-api` (the contracts), numpy and OpenCV, never on the twin.

## CLI and model selection

The panel offers a `cli` dropdown (`codex`, `claude`) and a `model` dropdown for the selected CLI.
The `(domyslne)` model choice uses that CLI's default. Switching CLI replaces an incompatible
model selection with this default. The choices checked on 2026-10-24 are:

| CLI | Model choices, besides the empty default |
|---|---|
| `codex` | `gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` |
| `claude` | `sonnet`, `opus`, `haiku`, `fable`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-5-5`, `claude-fable-5-1` |

The Codex choices come from its locally cached visible models, all accepting images, fetched on
2026-10-24 by CLI 0.163.0. The Claude choices match installed Claude Code 2.1.310 and its official
[model configuration documentation](https://code.claude.com/docs/en/model-config). Refresh these
lists when the CLIs change. Explicit model names supplied in config remain supported.

## How a question runs (`machinekind_twin_llm.cli`)

- Each call gets a directory of its own, holding the JPEGs, the prompt (`stdin.txt`), `stdout.txt`
  and `stderr.txt`. It is removed afterwards. With `MACHINEKIND_TWIN_LLM_KEEP=<dir>` set, a copy
  goes to `<dir>` first.
- The CLI runs in a session of its own. A timeout or a cancel kills the whole process group
  within 0.2 s.
- The CLI's environment has API keys, base-URL overrides and the calling Claude Code or cmux
  session's variables removed (`cli.STRIPPED_ENV`), so it falls back to its subscription login.
  `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) is kept.
- `check_login` refuses a CLI logged in with an API key and a CLI that is not logged in. Any
  other login passes.
- claude never gets `--bare`: it turns OAuth off.

## Smoke check by hand (`machinekind_twin_llm.smoke`)

```
uv run python -m machinekind_twin_llm.smoke --cli claude --executable ~/.local/bin/claude
uv run python -m machinekind_twin_llm.smoke --cli claude --executable ~/.local/bin/claude --route read
uv run python -m machinekind_twin_llm.smoke --cli codex --executable ~/.local/bin/codex --model gpt-6-astra
```

Without image arguments it draws two synthetic 640x480 photos, a red square on one and a blue
disc on the other, asks one question and checks that the model puts each object within 40 px of
where it was drawn. Pass image files to ask about your own photos instead. Run it again after a
CLI update: the claude flags it relies on are partly hidden.

`--executable ~/.local/bin/<cli>` bypasses terminal wrappers on PATH (cmux shims).

### Outcome on 2026-10-23 (macOS, Apple silicon)

claude 2.1.309, subscription login "claude.ai (max)", default model (claude-fable-5-1), `--effort low`:

| Check | Route "stream" | Route "read" |
|---|---|---|
| `--max-turns` accepted (hidden flag) | yes (3) | yes (6) |
| `--json-schema` returns `structured_output` | yes, with `--tools ""` | yes, with `--tools Read --allowedTools Read` |
| both images reached the model, both centres exact | yes | yes |
| exits on stdin EOF | yes | yes |
| API-key field of the `system`/`init` event | `apiKeySource`, value `"none"` | (no init event in `--output-format json`) |
| wall time for one question | 4.2 s (API 2.4 s, 2 turns) | 11.2 s |

- Images inline over stream-json stdin work, so route "stream" is the default. Route "read" works
  too, costs Read tool turns and takes about 2.5 times as long.
- Flags accepted together: `-p --input-format stream-json --output-format stream-json --verbose
  --tools "" --max-turns 3 --json-schema ... --no-session-persistence --safe-mode
  --strict-mcp-config --permission-prompts none --effort low`.
- The init event also reports `permissionMode: "auto"`, `tools: ["StructuredOutput"]` and no MCP
  servers. The stream carries a `rate_limit_event` with the subscription's five-hour window.

codex-cli 0.163.0, subscription login "Logged in using ChatGPT", `-m gpt-6-astra`, effort low:

| Check | Result |
|---|---|
| `exec --json -o ... --skip-git-repo-check --ephemeral --ignore-user-config --sandbox read-only --cd ... --color never --output-schema ...` accepted | yes |
| two `--image` flags reach the model, then `-- -` reads the prompt from stdin | yes, both centres exact |
| the strict schema (nullable `box` as `anyOf` with null) works as `--output-schema` | yes, answer in `-o` and in the last `agent_message` |
| `-c model_reasoning_effort` applies with `--ignore-user-config` | yes: an unknown effort is refused by the server ("Supported values are: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', and 'max'") |
| exits on stdin EOF | yes |
| wall time for one question | 9.2 s (19.8k input tokens, 0 reasoning tokens) |

- With `--ignore-user-config` codex still tried to refresh the OAuth token of an MCP server
  configured elsewhere and logged an error on stderr. It does not affect the answer. Such lines
  are left out of error messages (`cli._NOISE`).
- A failed turn arrives as an `error` event and a `turn.failed` event on stdout, with exit 1.
