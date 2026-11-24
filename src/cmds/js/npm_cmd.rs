//! Filters npm output and auto-injects the "run" subcommand when appropriate.

use crate::core::runner;
use crate::core::shell::display_args;
use crate::core::tee;
use crate::core::utils::{resolved_command, strip_ansi};
use anyhow::Result;
use regex::Regex;
use std::io::IsTerminal;
use std::sync::LazyLock;

/// Known npm subcommands that should NOT get "run" injected.
/// Shared between production code and tests to avoid drift.
///
/// A tool with a list like this also belongs in `core::tracking::SUBCOMMAND_ROUTERS`,
/// which keeps the subcommand in its telemetry label.
const NPM_SUBCOMMANDS: &[&str] = &[
    "install",
    "i",
    "ci",
    "uninstall",
    "remove",
    "rm",
    "update",
    "up",
    "list",
    "ls",
    "outdated",
    "init",
    "create",
    "publish",
    "pack",
    "link",
    "audit",
    "fund",
    "exec",
    "explain",
    "why",
    "search",
    "view",
    "info",
    "show",
    "config",
    "set",
    "get",
    "cache",
    "prune",
    "dedupe",
    "doctor",
    "help",
    "version",
    "prefix",
    "root",
    "bin",
    "bugs",
    "docs",
    "home",
    "repo",
    "ping",
    "whoami",
    "token",
    "profile",
    "team",
    "access",
    "owner",
    "deprecate",
    "dist-tag",
    "star",
    "stars",
    "login",
    "logout",
    "adduser",
    "unpublish",
    "pkg",
    "diff",
    "rebuild",
    "test",
    "t",
    "start",
    "stop",
    "restart",
    "completion",
    "edit",
    "explore",
    "find-dupes",
    "help-search",
    "hook",
    "install-ci-test",
    "install-test",
    "ll",
    "org",
    "query",
    "run-script",
    "sbom",
    "shrinkwrap",
    "unstar",
];

pub fn run(args: &[String], verbose: u8, skip_env: bool) -> Result<i32> {
    // Determine if this is "npm run <script>" or another npm subcommand (install, list, etc.)
    // Only inject "run" when args look like a script name, not a known npm subcommand.
    let first_arg = args.first().map(|s| s.as_str());
    let is_run_explicit = first_arg == Some("run");
    let is_npm_subcommand = first_arg
        .map(|a| NPM_SUBCOMMANDS.contains(&a) || a.starts_with('-'))
        .unwrap_or(false);

    let mut effective_args: Vec<String> = Vec::with_capacity(args.len() + 1);
    if is_run_explicit || is_npm_subcommand {
        effective_args.extend_from_slice(args);
    } else {
        // "rtk npm build" → "npm run build" (assume script name)
        effective_args.push("run".to_string());
        effective_args.extend_from_slice(args);
    }

    run_filtered("npm", &effective_args, verbose, skip_env, false)
}

/// Run an npx tool through the same filtered pipeline as `npm`.
///
/// Used for unrouted tools in the `Commands::Npx` fallback so that
/// `rtk npx cowsay hello` dispatches to `npx`, not `npm`. Honors `--skip-env`
/// the same way `run` does.
pub fn exec(args: &[String], verbose: u8, skip_env: bool) -> Result<i32> {
    run_filtered("npx", args, verbose, skip_env, false)
}

/// Run an arbitrary package runner (`npx`, `bunx`) under the npm output filter.
///
/// This is the light line filter, not the errors-only one: these runners host
/// tools whose entire value is their stdout, so dropping non-error lines would
/// return nothing.
pub fn exec_with(runner: &str, args: &[String], verbose: u8, skip_env: bool) -> Result<i32> {
    run_filtered(runner, args, verbose, skip_env, true)
}

/// Shared command-execution path for `run` (npm) and `exec` (npx).
///
/// Builds the resolved command, appends args, applies `SKIP_ENV_VALIDATION`,
/// emits the verbose log line, and routes through `runner::run_filtered` with
/// the npm output filter.
/// `forward_piped_stdin` sends the caller's stdin to the child when it is a
/// pipe rather than a terminal. The filtered path buffers output until exit, so
/// inheriting a terminal would let an interactive tool prompt invisibly and wait
/// forever; a pipe already holds its data and cannot do that.
///
/// `bunx` opts in, `npx` does not: npx's behaviour here predates this path and
/// is tracked on its own as rtk-ai/rtk#2431, which covers every filtered entry
/// point rather than this one.
fn run_filtered(
    name: &str,
    args: &[String],
    verbose: u8,
    skip_env: bool,
    forward_piped_stdin: bool,
) -> Result<i32> {
    let mut cmd = resolved_command(name);
    for arg in args {
        cmd.arg(arg);
    }

    if skip_env {
        cmd.env("SKIP_ENV_VALIDATION", "1");
    }

    let args_display = display_args(args);
    if verbose > 0 {
        eprintln!("Running: {} {}", name, args_display);
    }

    if name == "npm" && runner::is_watch_mode(args) {
        let mut watch_cmd = resolved_command(name);
        for arg in args {
            watch_cmd.arg(arg);
        }
        if skip_env {
            watch_cmd.env("SKIP_ENV_VALIDATION", "1");
        }
        return runner::run(
            watch_cmd,
            name,
            &args_display,
            runner::RunMode::Passthrough,
            runner::RunOptions::default(),
        );
    }

    let mut opts = runner::RunOptions::default();
    if forward_piped_stdin && !std::io::stdin().is_terminal() {
        opts = opts.inherit_stdin();
    }

    if name == "npm" {
        let kind = classify_npm_invocation(args);
        return runner::run_filtered_with_exit(
            cmd,
            name,
            &args_display,
            move |raw, exit_code| match kind {
                NpmCommandKind::Test => filter_npm_test_output(raw, exit_code),
                NpmCommandKind::Build => filter_npm_build_output(raw, exit_code),
                NpmCommandKind::General => filter_npm_general_output(raw, exit_code),
            },
            opts.tee("npm"),
        );
    }

    runner::run_filtered(cmd, name, &args_display, filter_npm_output, opts)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpmCommandKind {
    Test,
    Build,
    General,
}

fn classify_npm_invocation(args: &[String]) -> NpmCommandKind {
    match args.first().map(String::as_str) {
        Some("test" | "t") => NpmCommandKind::Test,
        Some("rebuild") => NpmCommandKind::Build,
        Some("run" | "run-script") => match args.get(1).map(String::as_str) {
            Some("test") => NpmCommandKind::Test,
            Some(script) if script == "build" || script.starts_with("build:") => {
                NpmCommandKind::Build
            }
            _ => NpmCommandKind::General,
        },
        _ => NpmCommandKind::General,
    }
}

static IMPORTANT_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(error|err!|warn|warning|deprecated|failed|fatal|exception|panic)\b")
        .unwrap()
});

static TEST_SUMMARY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        Test\s+Files|Test\s+Suites|Tests:|Snapshots:|Duration|Time:|Start\s+at|
        Ran\s+\d+\s+test|
        \b\d+\s+(?:passing|passed|failing|failed|skipped|todo|pending)\b|
        ^(?:ℹ\s+)?(?:tests|suites|pass|fail|cancelled|skipped|todo)\s+\d+|
        ^[✓✔]\s+\d+\s+test
        ",
    )
    .unwrap()
});

static PASS_NOISE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:[✓✔]\s+|PASS\s+|ok\s+|ℹ\s+ok\s+)").unwrap());

static BUILD_NOISE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        ^transforming\b|
        ^\d+\s+modules\s+transformed|
        ^[├└│▶]\s+|
        ^\d{2}:\d{2}:\d{2}\s+\[(?:build|router|content)\]|
        ^generating\s+static\s+routes|
        ^dist/\S+\s+[\d.]+\s+[kKmMgG]?B\b|
        \[built\]|
        ^asset\s+\S+
        ",
    )
    .unwrap()
});

static BUILD_SUMMARY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(built\s+in|complete!?|build\s+completed|compiled|successfully|finished\s+in|[✓✔]\s+\d+\s+modules|\btotal\b|\bpages?\s+built\b)",
    )
    .unwrap()
});

fn is_npm_boilerplate(line: &str) -> bool {
    let trimmed = line.trim_start();
    if line.starts_with('>') && line.contains('@') {
        return true;
    }
    if trimmed.starts_with("npm WARN")
        || trimmed.starts_with("npm warn")
        || trimmed.starts_with("npm notice")
    {
        return true;
    }
    if line.contains("⸩") || line.contains("⸨") || (line.contains("...") && line.len() < 10) {
        return true;
    }
    line.trim().is_empty()
}

fn clean_npm_lines(output: &str) -> (String, Vec<String>) {
    let cleaned = strip_ansi(output);
    let lines = cleaned
        .lines()
        .filter(|line| !is_npm_boilerplate(line))
        .map(str::to_string)
        .collect();
    (cleaned, lines)
}

fn cap_passing_lines_with_recall(
    kept: Vec<String>,
    max_lines: usize,
    raw_output: &str,
    tee_slug: &str,
) -> String {
    if kept.is_empty() {
        return "ok".to_string();
    }
    if kept.len() <= max_lines {
        return kept.join("\n");
    }

    let mut important = Vec::new();
    let mut non_important_tail = Vec::new();
    let total = kept.len();
    for line in &kept {
        if IMPORTANT_LINE_RE.is_match(line) {
            important.push(line.clone());
        } else {
            non_important_tail.push(line.clone());
        }
    }

    let mut out = Vec::new();
    let remaining_budget = max_lines.saturating_sub(important.len());
    let tail_take = remaining_budget.min(non_important_tail.len());
    let omitted = total.saturating_sub(important.len() + tail_take);
    if omitted > 0 {
        out.push(format!("... (+{} earlier lines omitted)", omitted));
    }
    for line in important.into_iter().take(max_lines) {
        out.push(line);
    }
    if tail_take > 0 {
        let start = non_important_tail.len() - tail_take;
        out.extend(non_important_tail.into_iter().skip(start));
    }
    if omitted > 0
        && raw_output.len() >= tee::MIN_TEE_SIZE
        && let Some(hint) = tee::force_tee_hint(raw_output, tee_slug)
    {
        out.push(hint);
    }
    out.join("\n")
}

fn cap_failing_lines(lines: Vec<String>, max_lines: usize) -> String {
    if lines.is_empty() {
        return "ok".to_string();
    }
    if lines.len() <= max_lines {
        return lines.join("\n");
    }
    let head = max_lines / 2;
    let tail = max_lines - head;
    let omitted = lines.len() - max_lines;
    let mut out = Vec::with_capacity(max_lines + 1);
    out.extend(lines.iter().take(head).cloned());
    out.push(format!("... (+{} middle lines omitted)", omitted));
    out.extend(lines.iter().skip(lines.len() - tail).cloned());
    out.join("\n")
}

/// Filter npm run output - strip ANSI, boilerplate, progress bars, npm WARN/warn/notice
fn filter_npm_output(output: &str) -> String {
    let (_, result) = clean_npm_lines(output);

    if result.is_empty() {
        "ok".to_string()
    } else {
        result.join("\n")
    }
}

fn filter_npm_test_output(output: &str, exit_code: i32) -> String {
    let (_, lines) = clean_npm_lines(output);
    if lines.is_empty() {
        return "ok".to_string();
    }

    if exit_code == 0 {
        let summary: Vec<String> = lines
            .iter()
            .filter(|line| {
                let trimmed = line.trim();
                TEST_SUMMARY_RE.is_match(trimmed) || IMPORTANT_LINE_RE.is_match(trimmed)
            })
            .cloned()
            .collect();
        if !summary.is_empty() {
            return summary.into_iter().take(20).collect::<Vec<_>>().join("\n");
        }
        return cap_passing_lines_with_recall(lines, 25, output, "npm-test");
    }

    let kept: Vec<String> = lines
        .into_iter()
        .filter(|line| {
            let trimmed = line.trim_start();
            !PASS_NOISE_RE.is_match(trimmed) || IMPORTANT_LINE_RE.is_match(trimmed)
        })
        .collect();
    cap_failing_lines(kept, 80)
}

fn filter_npm_build_output(output: &str, exit_code: i32) -> String {
    let (_, lines) = clean_npm_lines(output);
    if lines.is_empty() {
        return "ok".to_string();
    }

    if exit_code == 0 {
        let mut kept = Vec::new();
        let mut dropped = 0usize;
        for line in &lines {
            let trimmed = line.trim();
            let important =
                IMPORTANT_LINE_RE.is_match(trimmed) || BUILD_SUMMARY_RE.is_match(trimmed);
            if !important && BUILD_NOISE_RE.is_match(trimmed) {
                dropped += 1;
            } else {
                kept.push(line.clone());
            }
        }
        if kept.is_empty() {
            kept = lines;
        }
        let body = cap_passing_lines_with_recall(kept, 35, output, "npm-build");
        if dropped > 0
            && !body.contains("rtk recall")
            && output.len() >= tee::MIN_TEE_SIZE
            && let Some(hint) = tee::force_tee_hint(output, "npm-build")
        {
            return format!(
                "... (+{} build detail lines omitted)\n{}\n{}",
                dropped, body, hint
            );
        }
        return body;
    }

    let kept: Vec<String> = lines
        .into_iter()
        .filter(|line| {
            let trimmed = line.trim();
            IMPORTANT_LINE_RE.is_match(trimmed) || !BUILD_NOISE_RE.is_match(trimmed)
        })
        .collect();
    cap_failing_lines(kept, 80)
}

fn filter_npm_general_output(output: &str, exit_code: i32) -> String {
    let (_, lines) = clean_npm_lines(output);
    if lines.is_empty() {
        return "ok".to_string();
    }
    if exit_code == 0 {
        cap_passing_lines_with_recall(lines, 40, output, "npm")
    } else {
        cap_failing_lines(lines, 80)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_npm_output() {
        let output = r#"
> project@1.0.0 build
> next build

npm WARN deprecated inflight@1.0.6: This module is not supported
npm notice

   Creating an optimized production build...
   ✓ Build completed
"#;
        let result = filter_npm_output(output);
        assert!(!result.contains("npm WARN"));
        assert!(!result.contains("npm notice"));
        assert!(!result.contains("> project@"));
        assert!(result.contains("Build completed"));
    }

    #[test]
    fn test_npm_subcommand_routing() {
        // Uses the shared NPM_SUBCOMMANDS constant — no drift between prod and test
        fn needs_run_injection(args: &[&str]) -> bool {
            let first = args.first().copied();
            let is_run_explicit = first == Some("run");
            let is_subcommand = first
                .map(|a| NPM_SUBCOMMANDS.contains(&a) || a.starts_with('-'))
                .unwrap_or(false);
            !is_run_explicit && !is_subcommand
        }

        // Known subcommands should NOT get "run" injected
        for subcmd in NPM_SUBCOMMANDS {
            assert!(
                !needs_run_injection(&[subcmd]),
                "'npm {}' should NOT inject 'run'",
                subcmd
            );
        }

        // Script names SHOULD get "run" injected
        for script in &["build", "dev", "lint", "typecheck", "deploy"] {
            assert!(
                needs_run_injection(&[script]),
                "'npm {}' SHOULD inject 'run'",
                script
            );
        }

        // Flags should NOT get "run" injected
        assert!(!needs_run_injection(&["--version"]));
        assert!(!needs_run_injection(&["-h"]));

        // Explicit "run" should NOT inject another "run"
        assert!(!needs_run_injection(&["run", "build"]));
    }

    #[test]
    fn test_extended_official_subcommands_do_not_inject_run() {
        let subcommands = [
            "completion",
            "edit",
            "explore",
            "find-dupes",
            "help-search",
            "hook",
            "install-ci-test",
            "install-test",
            "ll",
            "org",
            "query",
            "run-script",
            "sbom",
            "shrinkwrap",
            "unstar",
        ];

        for subcommand in subcommands {
            assert!(
                NPM_SUBCOMMANDS.contains(&subcommand),
                "npm {subcommand} must be routed as a native subcommand"
            );
        }
    }

    #[test]
    fn test_filter_npm_output_empty() {
        let output = "\n\n\n";
        let result = filter_npm_output(output);
        assert_eq!(result, "ok");
    }

    #[test]
    fn test_filter_npm_output_strips_ansi_and_lowercase_warn() {
        let output = "\x1b[33mnpm warn\x1b[39m deprecated foo@1.0.0\nnpm notice New minor version\n\x1b[32m✓ Ready\x1b[39m\n";
        let result = filter_npm_output(output);
        assert_eq!(result, "✓ Ready");
    }

    #[test]
    fn test_classify_npm_invocation() {
        assert_eq!(
            classify_npm_invocation(&["test".into()]),
            NpmCommandKind::Test
        );
        assert_eq!(
            classify_npm_invocation(&["run".into(), "test".into()]),
            NpmCommandKind::Test
        );
        assert_eq!(
            classify_npm_invocation(&["run".into(), "build".into()]),
            NpmCommandKind::Build
        );
        assert_eq!(
            classify_npm_invocation(&["run".into(), "build:prod".into()]),
            NpmCommandKind::Build
        );
        assert_eq!(
            classify_npm_invocation(&["install".into()]),
            NpmCommandKind::General
        );
    }

    #[test]
    fn test_filter_npm_build_output_condenses_passing_and_keeps_errors() {
        let mut raw = String::from("> app@1.0.0 build\n> astro build\n");
        for i in 0..200 {
            raw.push_str(&format!(
                "\x1b[32m├─ /docs/page-{i}/index.html (+12ms)\x1b[39m\n"
            ));
            raw.push_str(&format!(
                "dist/assets/chunk-{i}.js  14.20 kB │ gzip: 4.10 kB\n"
            ));
        }
        raw.push_str("200 page(s) built in 4.12s\n");
        raw.push_str("✓ Completed in 4.15s.\n");

        let filtered = filter_npm_build_output(&raw, 0);
        assert!(filtered.contains("200 page(s) built in 4.12s"));
        assert!(filtered.contains("Completed in 4.15s."));
        assert!(
            filtered.len() * 5 < raw.len(),
            "expected >80% reduction, got {} vs {}",
            filtered.len(),
            raw.len()
        );

        let fail_raw = "> app@1.0.0 build\ntransforming (1) src/index.ts\nerror during build:\nTypeError: x is not a function\n";
        let fail_filtered = filter_npm_build_output(fail_raw, 1);
        assert!(fail_filtered.contains("error during build:"));
        assert!(fail_filtered.contains("TypeError: x is not a function"));
        assert!(!fail_filtered.contains("transforming (1)"));
    }

    #[test]
    fn test_filter_npm_test_output_condenses_pass_and_keeps_fail() {
        let mut pass_raw = String::from("> app@1.0.0 test\n> vitest run\n");
        for i in 0..60 {
            pass_raw.push_str(&format!(" ✓ src/test_{i}.test.ts (3 tests) 12ms\n"));
        }
        pass_raw.push_str(
            " Test Files  60 passed (60)\n      Tests  180 passed (180)\n   Duration  1.42s\n",
        );

        let pass_filtered = filter_npm_test_output(&pass_raw, 0);
        assert!(pass_filtered.contains("Test Files  60 passed (60)"));
        assert!(pass_filtered.contains("Tests  180 passed (180)"));
        assert!(!pass_filtered.contains("src/test_0.test.ts"));

        let fail_raw = "> app@1.0.0 test\n ✓ src/ok.test.ts (2 tests)\n ✕ src/broken.test.ts > fails\nAssertionError: expected 1 to be 2\n Test Files  1 failed | 1 passed (2)\n";
        let fail_filtered = filter_npm_test_output(fail_raw, 1);
        assert!(fail_filtered.contains("src/broken.test.ts"));
        assert!(fail_filtered.contains("AssertionError: expected 1 to be 2"));
        assert!(!fail_filtered.contains("src/ok.test.ts"));
    }
}
