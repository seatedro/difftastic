//! Library entrypoints for using difftastic inside other Rust crates.

#![allow(renamed_and_removed_lints)]
#![allow(clippy::type_complexity)]
#![allow(clippy::comparison_to_empty)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::mutable_key_type)]
#![allow(unknown_lints)]
#![allow(clippy::manual_unwrap_or_default)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::needless_as_bytes)]
#![allow(dead_code)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::todo)]
#![warn(clippy::dbg_macro)]

mod conflicts;
mod constants;
mod diff;
mod display;
mod exit_codes;
mod files;
mod hash;
mod line_parser;
mod lines;
mod options;
mod parse;
mod summary;
mod version;
mod words;

#[macro_use]
extern crate log;

use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use humansize::{format_size, FormatSizeOptions, BINARY};
use line_numbers::LineNumber;
use typed_arena::Arena;

use crate::diff::changes::ChangeMap;
use crate::diff::dijkstra::{mark_syntax, ExceededGraphLimit};
use crate::diff::sliders::fix_all_sliders;
use crate::diff::unchanged;
use crate::display::context::opposite_positions;
use crate::display::hunks::{matched_pos_to_hunks, merge_adjacent};
use crate::files::{guess_content, ProbableFileKind};
use crate::hash::{DftHashMap, DftHashSet};
use crate::lines::MaxLine;
use crate::options::{DiffOptions, DisplayOptions, FileArgument};
use crate::parse::guess_language::{guess, language_name, Language, LanguageOverride};
use crate::parse::syntax::{self, init_next_prev};
use crate::parse::tree_sitter_parser as tsp;
use crate::summary::{DiffResult, FileContent, FileFormat};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DifftasticError(String);

impl DifftasticError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for DifftasticError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DifftasticError {}

impl From<serde_json::Error> for DifftasticError {
    fn from(value: serde_json::Error) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighlightKind {
    Normal,
    Keyword,
    String,
    Comment,
    Number,
    Type,
    Function,
    Operator,
    Punctuation,
    Variable,
    Constant,
    Builtin,
    Attribute,
    Tag,
    Property,
    Namespace,
    Label,
    Preprocessor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HighlightSpan {
    pub offset: u32,
    pub length: u32,
    pub kind: HighlightKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeIntensity {
    Novel,
    NovelWord,
    UnchangedContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChangeSpan {
    pub start_col: u32,
    pub end_col: u32,
    pub highlight: HighlightKind,
    pub intensity: ChangeIntensity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticLine {
    pub lhs_line: Option<u32>,
    pub rhs_line: Option<u32>,
    pub lhs_changes: Vec<ChangeSpan>,
    pub rhs_changes: Vec<ChangeSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticChunk {
    pub lines: Vec<SemanticLine>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffStatus {
    Unchanged,
    Changed,
    Created,
    Deleted,
    Binary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticDiffResult {
    pub status: DiffStatus,
    pub language: String,
    pub line_fallback_reason: Option<String>,
    pub chunks: Vec<SemanticChunk>,
    pub aligned_lines: Vec<(Option<u32>, Option<u32>)>,
}

#[derive(Debug, Clone)]
pub struct DiffRequest<'a> {
    pub display_path: &'a str,
    pub lhs_path: Option<&'a Path>,
    pub rhs_path: Option<&'a Path>,
    pub lhs_bytes: &'a [u8],
    pub rhs_bytes: &'a [u8],
}

const SEMANTIC_CONTEXT_LINES: usize = 3;
const SEMANTIC_PARSE_ERROR_LIMIT_ENV: &str = "DFT_PARSE_ERROR_LIMIT";
const SEMANTIC_DEFAULT_PARSE_ERROR_LIMIT: usize = 100;
const TIMINGS_ENV: &str = "DFT_TIMINGS";

struct TimingLog<'a> {
    display_path: &'a str,
    start: Instant,
    last: Instant,
}

impl<'a> TimingLog<'a> {
    fn new(display_path: &'a str) -> Option<Self> {
        std::env::var_os(TIMINGS_ENV).map(|_| {
            let now = Instant::now();
            Self {
                display_path,
                start: now,
                last: now,
            }
        })
    }

    fn mark(&mut self, phase: &str) {
        let now = Instant::now();
        eprintln!(
            "difftastic timing path={:?} phase={} elapsed_ms={:.3} total_ms={:.3}",
            self.display_path,
            phase,
            duration_ms(now.duration_since(self.last)),
            duration_ms(now.duration_since(self.start)),
        );
        self.last = now;
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn mark_timing(timing: &mut Option<&mut TimingLog<'_>>, phase: &str) {
    if let Some(timing) = timing.as_mut() {
        (**timing).mark(phase);
    }
}

pub fn diff_bytes_json(request: DiffRequest<'_>) -> Result<String, DifftasticError> {
    let lhs_path =
        file_argument_for_side(request.lhs_path, request.display_path, request.lhs_bytes);
    let rhs_path =
        file_argument_for_side(request.rhs_path, request.display_path, request.rhs_bytes);
    let diff_options = DiffOptions::default();
    let display_options = DisplayOptions::default();
    let overrides = Vec::<(LanguageOverride, Vec<glob::Pattern>)>::new();
    let binary_overrides = Vec::<glob::Pattern>::new();

    let diff = diff_bytes(
        request.display_path,
        &lhs_path,
        &rhs_path,
        request.lhs_bytes,
        request.rhs_bytes,
        &display_options,
        &diff_options,
        &overrides,
        &binary_overrides,
    );
    Ok(display::json::serialize(&diff)?)
}

pub fn diff_bytes_semantic(
    request: DiffRequest<'_>,
) -> Result<SemanticDiffResult, DifftasticError> {
    let lhs_path =
        file_argument_for_side(request.lhs_path, request.display_path, request.lhs_bytes);
    let rhs_path =
        file_argument_for_side(request.rhs_path, request.display_path, request.rhs_bytes);
    let diff_options = semantic_diff_options();
    let overrides = Vec::<(LanguageOverride, Vec<glob::Pattern>)>::new();
    let binary_overrides = Vec::<glob::Pattern>::new();

    Ok(diff_bytes_semantic_impl(
        request.display_path,
        &lhs_path,
        &rhs_path,
        request.lhs_bytes,
        request.rhs_bytes,
        &diff_options,
        &overrides,
        &binary_overrides,
    ))
}

fn semantic_diff_options() -> DiffOptions {
    let mut options = DiffOptions::default();
    // Tree-sitter parse errors are common in macro-heavy real-world code such
    // as the Linux kernel. Diffy's semantic API prefers best-effort AST diffs
    // over immediately falling back to line diff for a handful of ERROR nodes.
    // Keep the upstream CLI default unchanged, and let DFT_PARSE_ERROR_LIMIT
    // override this for debugging or stricter behavior.
    options.parse_error_limit = std::env::var(SEMANTIC_PARSE_ERROR_LIMIT_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(SEMANTIC_DEFAULT_PARSE_ERROR_LIMIT);
    options
}

fn diff_bytes_semantic_impl(
    display_path: &str,
    lhs_path: &FileArgument,
    rhs_path: &FileArgument,
    lhs_bytes: &[u8],
    rhs_bytes: &[u8],
    diff_options: &DiffOptions,
    overrides: &[(LanguageOverride, Vec<glob::Pattern>)],
    binary_overrides: &[glob::Pattern],
) -> SemanticDiffResult {
    let (mut lhs_src, mut rhs_src) = match (
        guess_content(lhs_bytes, lhs_path, binary_overrides),
        guess_content(rhs_bytes, rhs_path, binary_overrides),
    ) {
        (ProbableFileKind::Binary, _) | (_, ProbableFileKind::Binary) => {
            return SemanticDiffResult {
                status: DiffStatus::Binary,
                language: FileFormat::Binary.to_string(),
                line_fallback_reason: None,
                chunks: Vec::new(),
                aligned_lines: Vec::new(),
            };
        }
        (ProbableFileKind::Text(lhs_src), ProbableFileKind::Text(rhs_src)) => (lhs_src, rhs_src),
    };

    if diff_options.strip_cr {
        lhs_src.retain(|c| c != '\r');
        rhs_src.retain(|c| c != '\r');
    }
    if !lhs_src.is_empty() && !lhs_src.ends_with('\n') {
        lhs_src.push('\n');
    }
    if !rhs_src.is_empty() && !rhs_src.ends_with('\n') {
        rhs_src.push('\n');
    }

    diff_file_content_semantic(
        display_path,
        lhs_path,
        rhs_path,
        &lhs_src,
        &rhs_src,
        diff_options,
        overrides,
    )
}

fn diff_file_content_semantic(
    display_path: &str,
    lhs_path: &FileArgument,
    rhs_path: &FileArgument,
    lhs_src: &str,
    rhs_src: &str,
    diff_options: &DiffOptions,
    overrides: &[(LanguageOverride, Vec<glob::Pattern>)],
) -> SemanticDiffResult {
    let mut timing = TimingLog::new(display_path);
    let guess_src = match rhs_path {
        FileArgument::DevNull => lhs_src,
        _ => rhs_src,
    };

    let language = guess(Path::new(display_path), guess_src, overrides);
    if let Some(timing) = timing.as_mut() {
        timing.mark("guess_language");
    }
    let guessed_format = file_format_for_language(language);
    let lhs_is_dev_null = matches!(lhs_path, FileArgument::DevNull);
    let rhs_is_dev_null = matches!(rhs_path, FileArgument::DevNull);

    // Added/deleted files do not need tree-sitter or line-diff work to tell Diffy
    // the file-level status. This avoids loading parser packs for file list rows
    // whose contents are not rendered yet.
    if lhs_is_dev_null && !rhs_is_dev_null {
        if let Some(timing) = timing.as_mut() {
            timing.mark("created");
        }
        return SemanticDiffResult {
            status: DiffStatus::Created,
            language: guessed_format.to_string(),
            line_fallback_reason: None,
            chunks: Vec::new(),
            aligned_lines: Vec::new(),
        };
    }
    if rhs_is_dev_null && !lhs_is_dev_null {
        if let Some(timing) = timing.as_mut() {
            timing.mark("deleted");
        }
        return SemanticDiffResult {
            status: DiffStatus::Deleted,
            language: guessed_format.to_string(),
            line_fallback_reason: None,
            chunks: Vec::new(),
            aligned_lines: Vec::new(),
        };
    }

    if lhs_src == rhs_src {
        if let Some(timing) = timing.as_mut() {
            timing.mark("unchanged_bytes");
        }
        return SemanticDiffResult {
            status: DiffStatus::Unchanged,
            language: guessed_format.to_string(),
            line_fallback_reason: None,
            chunks: Vec::new(),
            aligned_lines: Vec::new(),
        };
    }

    let positions = text_diff_positions(lhs_src, rhs_src, diff_options, language, timing.as_mut());
    let result = text_positions_to_semantic(
        &positions.file_format,
        lhs_src,
        rhs_src,
        &positions.lhs_positions,
        &positions.rhs_positions,
    );
    if let Some(timing) = timing.as_mut() {
        timing.mark("semantic_convert");
    }
    result
}

fn line_fallback_reason(file_format: &FileFormat) -> Option<String> {
    match file_format {
        FileFormat::TextFallback { reason } => Some(reason.clone()),
        FileFormat::SupportedLanguage(_) | FileFormat::PlainText | FileFormat::Binary => None,
    }
}

fn text_positions_to_semantic(
    file_format: &FileFormat,
    lhs_src: &str,
    rhs_src: &str,
    lhs_positions: &[syntax::MatchedPos],
    rhs_positions: &[syntax::MatchedPos],
) -> SemanticDiffResult {
    use crate::display::context::all_matched_lines_filled;
    use crate::display::side_by_side::lines_with_novel;

    let lhs_lines: Vec<&str> = lhs_src.split('\n').collect();
    let rhs_lines: Vec<&str> = rhs_src.split('\n').collect();

    let (lhs_lines_with_novel, rhs_lines_with_novel) =
        lines_with_novel(lhs_positions, rhs_positions);

    if lhs_lines_with_novel.is_empty() && rhs_lines_with_novel.is_empty() {
        return SemanticDiffResult {
            status: DiffStatus::Unchanged,
            language: file_format.to_string(),
            line_fallback_reason: line_fallback_reason(file_format),
            chunks: Vec::new(),
            aligned_lines: Vec::new(),
        };
    }

    let matched_lines =
        all_matched_lines_filled(lhs_positions, rhs_positions, &lhs_lines, &rhs_lines);
    let aligned_lines: Vec<(Option<u32>, Option<u32>)> = matched_lines
        .iter()
        .map(|(lhs, rhs)| (lhs.map(|l| l.0), rhs.map(|l| l.0)))
        .collect();
    let mut lhs_changes_by_line = semantic_changes_by_line(lhs_positions);
    let mut rhs_changes_by_line = semantic_changes_by_line(rhs_positions);
    let chunks =
        semantic_chunk_ranges(&matched_lines, &lhs_lines_with_novel, &rhs_lines_with_novel)
            .into_iter()
            .map(|(start, end)| {
                let lines = matched_lines[start..end]
                    .iter()
                    .map(|(lhs_line_num, rhs_line_num)| SemanticLine {
                        lhs_line: lhs_line_num.map(|l| l.0),
                        rhs_line: rhs_line_num.map(|l| l.0),
                        lhs_changes: lhs_line_num
                            .and_then(|ln| lhs_changes_by_line.remove(&ln))
                            .unwrap_or_default(),
                        rhs_changes: rhs_line_num
                            .and_then(|ln| rhs_changes_by_line.remove(&ln))
                            .unwrap_or_default(),
                    })
                    .collect();
                SemanticChunk { lines }
            })
            .collect::<Vec<_>>();

    if chunks.is_empty() {
        SemanticDiffResult {
            status: DiffStatus::Unchanged,
            language: file_format.to_string(),
            line_fallback_reason: line_fallback_reason(file_format),
            chunks,
            aligned_lines: Vec::new(),
        }
    } else {
        SemanticDiffResult {
            status: DiffStatus::Changed,
            language: file_format.to_string(),
            line_fallback_reason: line_fallback_reason(file_format),
            chunks,
            aligned_lines,
        }
    }
}

fn semantic_chunk_ranges(
    matched_lines: &[(Option<LineNumber>, Option<LineNumber>)],
    lhs_lines_with_novel: &DftHashSet<LineNumber>,
    rhs_lines_with_novel: &DftHashSet<LineNumber>,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::<(usize, usize)>::new();
    for (idx, (lhs_line_num, rhs_line_num)) in matched_lines.iter().enumerate() {
        let lhs_novel = lhs_line_num
            .map(|line| lhs_lines_with_novel.contains(&line))
            .unwrap_or(false);
        let rhs_novel = rhs_line_num
            .map(|line| rhs_lines_with_novel.contains(&line))
            .unwrap_or(false);
        if !lhs_novel && !rhs_novel {
            continue;
        }

        let start = idx.saturating_sub(SEMANTIC_CONTEXT_LINES);
        let end = idx
            .saturating_add(SEMANTIC_CONTEXT_LINES)
            .saturating_add(1)
            .min(matched_lines.len());
        if let Some((_, last_end)) = ranges.last_mut() {
            if start <= *last_end {
                *last_end = (*last_end).max(end);
                continue;
            }
        }
        ranges.push((start, end));
    }
    ranges
}

fn semantic_changes_by_line(
    positions: &[syntax::MatchedPos],
) -> DftHashMap<LineNumber, Vec<ChangeSpan>> {
    let mut by_line: DftHashMap<LineNumber, Vec<ChangeSpan>> = DftHashMap::default();
    for matched in positions.iter().filter(|matched| matched.kind.is_novel()) {
        let (highlight, intensity) = convert_match_kind(&matched.kind);
        by_line
            .entry(matched.pos.line)
            .or_default()
            .push(ChangeSpan {
                start_col: matched.pos.start_col,
                end_col: matched.pos.end_col,
                highlight,
                intensity,
            });
    }

    for spans in by_line.values_mut() {
        spans.sort_unstable_by_key(|span| (span.start_col, span.end_col));
    }

    by_line
}

fn convert_match_kind(kind: &syntax::MatchKind) -> (HighlightKind, ChangeIntensity) {
    use syntax::{AtomKind, MatchKind, TokenKind};

    let (highlight, intensity) = match kind {
        MatchKind::Novel { highlight } => (highlight, ChangeIntensity::Novel),
        MatchKind::NovelWord { highlight } => (highlight, ChangeIntensity::NovelWord),
        MatchKind::UnchangedPartOfNovelItem { highlight, .. } => {
            (highlight, ChangeIntensity::UnchangedContext)
        }
        MatchKind::UnchangedToken { highlight, .. } | MatchKind::Ignored { highlight, .. } => {
            (highlight, ChangeIntensity::Novel)
        }
    };

    let kind = match highlight {
        TokenKind::Delimiter => HighlightKind::Punctuation,
        TokenKind::Atom(atom) => match atom {
            AtomKind::String(syntax::StringKind::StringLiteral) => HighlightKind::String,
            AtomKind::String(syntax::StringKind::Text) => HighlightKind::Normal,
            AtomKind::Keyword => HighlightKind::Keyword,
            AtomKind::Comment => HighlightKind::Comment,
            AtomKind::Type => HighlightKind::Type,
            AtomKind::Normal => HighlightKind::Normal,
            AtomKind::TreeSitterError => HighlightKind::Preprocessor,
        },
    };

    (kind, intensity)
}

pub fn highlight_ranges_for_path(
    path: &Path,
    source: &str,
) -> Result<Vec<HighlightSpan>, DifftasticError> {
    let overrides = Vec::<(LanguageOverride, Vec<glob::Pattern>)>::new();
    let Some(language) = guess(path, source, &overrides) else {
        return Ok(Vec::new());
    };
    tsp::highlight_ranges(source, language)
}

fn file_argument_for_side(path: Option<&Path>, display_path: &str, content: &[u8]) -> FileArgument {
    match path {
        Some(path) => FileArgument::NamedPath(path.to_path_buf()),
        None if content.is_empty() => FileArgument::DevNull,
        None => FileArgument::NamedPath(PathBuf::from(display_path)),
    }
}

fn diff_bytes(
    display_path: &str,
    lhs_path: &FileArgument,
    rhs_path: &FileArgument,
    lhs_bytes: &[u8],
    rhs_bytes: &[u8],
    display_options: &DisplayOptions,
    diff_options: &DiffOptions,
    overrides: &[(LanguageOverride, Vec<glob::Pattern>)],
    binary_overrides: &[glob::Pattern],
) -> DiffResult {
    let (mut lhs_src, mut rhs_src) = match (
        guess_content(lhs_bytes, lhs_path, binary_overrides),
        guess_content(rhs_bytes, rhs_path, binary_overrides),
    ) {
        (ProbableFileKind::Binary, _) | (_, ProbableFileKind::Binary) => {
            let has_byte_changes = if lhs_bytes == rhs_bytes {
                None
            } else {
                Some((lhs_bytes.len(), rhs_bytes.len()))
            };
            return DiffResult {
                extra_info: None,
                display_path: display_path.to_owned(),
                file_format: FileFormat::Binary,
                lhs_src: FileContent::Binary,
                rhs_src: FileContent::Binary,
                lhs_positions: vec![],
                rhs_positions: vec![],
                hunks: vec![],
                has_byte_changes,
                has_syntactic_changes: false,
            };
        }
        (ProbableFileKind::Text(lhs_src), ProbableFileKind::Text(rhs_src)) => (lhs_src, rhs_src),
    };

    if diff_options.strip_cr {
        lhs_src.retain(|c| c != '\r');
        rhs_src.retain(|c| c != '\r');
    }
    if !lhs_src.is_empty() && !lhs_src.ends_with('\n') {
        lhs_src.push('\n');
    }
    if !rhs_src.is_empty() && !rhs_src.ends_with('\n') {
        rhs_src.push('\n');
    }

    diff_file_content(
        display_path,
        lhs_path,
        rhs_path,
        &lhs_src,
        &rhs_src,
        display_options,
        diff_options,
        overrides,
    )
}

fn check_only_text(
    file_format: &FileFormat,
    display_path: &str,
    lhs_src: &str,
    rhs_src: &str,
) -> DiffResult {
    let has_byte_changes = if lhs_src == rhs_src {
        None
    } else {
        Some((lhs_src.as_bytes().len(), rhs_src.as_bytes().len()))
    };

    DiffResult {
        display_path: display_path.to_owned(),
        extra_info: None,
        file_format: file_format.clone(),
        lhs_src: FileContent::Text(lhs_src.into()),
        rhs_src: FileContent::Text(rhs_src.into()),
        lhs_positions: vec![],
        rhs_positions: vec![],
        hunks: vec![],
        has_byte_changes,
        has_syntactic_changes: lhs_src != rhs_src,
    }
}

fn file_format_for_language(language: Option<Language>) -> FileFormat {
    match language {
        Some(language) => FileFormat::SupportedLanguage(language),
        None => FileFormat::PlainText,
    }
}

struct TextDiffPositions {
    file_format: FileFormat,
    lhs_positions: Vec<syntax::MatchedPos>,
    rhs_positions: Vec<syntax::MatchedPos>,
}

fn text_diff_positions(
    lhs_src: &str,
    rhs_src: &str,
    diff_options: &DiffOptions,
    language: Option<Language>,
    mut timing: Option<&mut TimingLog<'_>>,
) -> TextDiffPositions {
    let lang_config = language.and_then(|language| match tsp::from_language(language) {
        Ok(config) => Some((language, config)),
        Err(error) => {
            info!("Falling back to line diff: {error}");
            None
        }
    });
    mark_timing(&mut timing, "language_config");

    match lang_config {
        None => {
            let result = TextDiffPositions {
                file_format: FileFormat::PlainText,
                lhs_positions: line_parser::change_positions(lhs_src, rhs_src),
                rhs_positions: line_parser::change_positions(rhs_src, lhs_src),
            };
            mark_timing(&mut timing, "line_diff");
            result
        }
        Some((language, lang_config)) => {
            let arena = Arena::new();
            let tree_result = tsp::to_tree_with_limit(diff_options, &lang_config, lhs_src, rhs_src);
            mark_timing(&mut timing, "parse");
            match tree_result {
                Ok((lhs_tree, rhs_tree)) => {
                    let syntax_result = tsp::to_syntax_with_limit(
                        lhs_src,
                        rhs_src,
                        &lhs_tree,
                        &rhs_tree,
                        &arena,
                        &lang_config,
                        diff_options,
                    );
                    mark_timing(&mut timing, "syntax");

                    match syntax_result {
                        Ok((lhs, rhs)) => {
                            let mut change_map = ChangeMap::default();
                            let possibly_changed =
                                if std::env::var("DFT_DBG_KEEP_UNCHANGED").is_ok() {
                                    vec![(lhs.clone(), rhs.clone())]
                                } else {
                                    unchanged::mark_unchanged(&lhs, &rhs, &mut change_map)
                                };
                            mark_timing(&mut timing, "unchanged");

                            let mut exceeded_graph_limit = false;
                            for (lhs_section_nodes, rhs_section_nodes) in possibly_changed {
                                init_next_prev(&lhs_section_nodes);
                                init_next_prev(&rhs_section_nodes);

                                match mark_syntax(
                                    lhs_section_nodes.first().copied(),
                                    rhs_section_nodes.first().copied(),
                                    &mut change_map,
                                    diff_options.graph_limit,
                                ) {
                                    Ok(()) => {}
                                    Err(ExceededGraphLimit {}) => {
                                        exceeded_graph_limit = true;
                                        break;
                                    }
                                }
                            }
                            mark_timing(&mut timing, "dijkstra");

                            if exceeded_graph_limit {
                                let result = TextDiffPositions {
                                    file_format: FileFormat::TextFallback {
                                        reason: "exceeded DFT_GRAPH_LIMIT".into(),
                                    },
                                    lhs_positions: line_parser::change_positions(lhs_src, rhs_src),
                                    rhs_positions: line_parser::change_positions(rhs_src, lhs_src),
                                };
                                mark_timing(&mut timing, "line_diff");
                                result
                            } else {
                                fix_all_sliders(language, &lhs, &mut change_map);
                                fix_all_sliders(language, &rhs, &mut change_map);
                                mark_timing(&mut timing, "sliders");

                                let mut lhs_positions = syntax::change_positions(&lhs, &change_map);
                                let mut rhs_positions = syntax::change_positions(&rhs, &change_map);

                                if diff_options.ignore_comments {
                                    let lhs_comments =
                                        tsp::comment_positions(&lhs_tree, lhs_src, &lang_config);
                                    lhs_positions.extend(lhs_comments);

                                    let rhs_comments =
                                        tsp::comment_positions(&rhs_tree, rhs_src, &lang_config);
                                    rhs_positions.extend(rhs_comments);
                                }
                                mark_timing(&mut timing, "positions");

                                TextDiffPositions {
                                    file_format: FileFormat::SupportedLanguage(language),
                                    lhs_positions,
                                    rhs_positions,
                                }
                            }
                        }
                        Err(tsp::ExceededParseErrorLimit(error_count)) => {
                            let result = TextDiffPositions {
                                file_format: FileFormat::TextFallback {
                                    reason: format!(
                                        "{} {} parse error{}, exceeded DFT_PARSE_ERROR_LIMIT",
                                        error_count,
                                        language_name(language),
                                        if error_count == 1 { "" } else { "s" }
                                    ),
                                },
                                lhs_positions: line_parser::change_positions(lhs_src, rhs_src),
                                rhs_positions: line_parser::change_positions(rhs_src, lhs_src),
                            };
                            mark_timing(&mut timing, "line_diff");
                            result
                        }
                    }
                }
                Err(tsp::ExceededByteLimit(num_bytes)) => {
                    let format_options = FormatSizeOptions::from(BINARY).decimal_places(1);
                    let result = TextDiffPositions {
                        file_format: FileFormat::TextFallback {
                            reason: format!(
                                "{} exceeded DFT_BYTE_LIMIT",
                                &format_size(num_bytes, format_options)
                            ),
                        },
                        lhs_positions: line_parser::change_positions(lhs_src, rhs_src),
                        rhs_positions: line_parser::change_positions(rhs_src, lhs_src),
                    };
                    mark_timing(&mut timing, "line_diff");
                    result
                }
            }
        }
    }
}

fn diff_file_content(
    display_path: &str,
    _lhs_path: &FileArgument,
    rhs_path: &FileArgument,
    lhs_src: &str,
    rhs_src: &str,
    display_options: &DisplayOptions,
    diff_options: &DiffOptions,
    overrides: &[(LanguageOverride, Vec<glob::Pattern>)],
) -> DiffResult {
    let guess_src = match rhs_path {
        FileArgument::DevNull => lhs_src,
        _ => rhs_src,
    };

    let language = guess(Path::new(display_path), guess_src, overrides);

    if lhs_src == rhs_src {
        let file_format = match language {
            Some(language) => FileFormat::SupportedLanguage(language),
            None => FileFormat::PlainText,
        };

        return DiffResult {
            extra_info: None,
            display_path: display_path.to_owned(),
            file_format,
            lhs_src: FileContent::Text("".into()),
            rhs_src: FileContent::Text("".into()),
            lhs_positions: vec![],
            rhs_positions: vec![],
            hunks: vec![],
            has_byte_changes: None,
            has_syntactic_changes: false,
        };
    }

    let lang_config = language.and_then(|language| match tsp::from_language(language) {
        Ok(config) => Some((language, config)),
        Err(error) => {
            info!("Falling back to line diff: {error}");
            None
        }
    });

    let (file_format, lhs_positions, rhs_positions) = match lang_config {
        None => {
            let file_format = FileFormat::PlainText;
            if diff_options.check_only {
                return check_only_text(&file_format, display_path, lhs_src, rhs_src);
            }

            let lhs_positions = line_parser::change_positions(lhs_src, rhs_src);
            let rhs_positions = line_parser::change_positions(rhs_src, lhs_src);
            (file_format, lhs_positions, rhs_positions)
        }
        Some((language, lang_config)) => {
            let arena = Arena::new();
            match tsp::to_tree_with_limit(diff_options, &lang_config, lhs_src, rhs_src) {
                Ok((lhs_tree, rhs_tree)) => match tsp::to_syntax_with_limit(
                    lhs_src,
                    rhs_src,
                    &lhs_tree,
                    &rhs_tree,
                    &arena,
                    &lang_config,
                    diff_options,
                ) {
                    Ok((lhs, rhs)) => {
                        if diff_options.check_only {
                            let has_syntactic_changes = lhs != rhs;
                            let has_byte_changes = if lhs_src == rhs_src {
                                None
                            } else {
                                Some((lhs_src.as_bytes().len(), rhs_src.as_bytes().len()))
                            };

                            return DiffResult {
                                extra_info: None,
                                display_path: display_path.to_owned(),
                                file_format: FileFormat::SupportedLanguage(language),
                                lhs_src: FileContent::Text(lhs_src.to_owned()),
                                rhs_src: FileContent::Text(rhs_src.to_owned()),
                                lhs_positions: vec![],
                                rhs_positions: vec![],
                                hunks: vec![],
                                has_byte_changes,
                                has_syntactic_changes,
                            };
                        }

                        let mut change_map = ChangeMap::default();
                        let possibly_changed = if std::env::var("DFT_DBG_KEEP_UNCHANGED").is_ok() {
                            vec![(lhs.clone(), rhs.clone())]
                        } else {
                            unchanged::mark_unchanged(&lhs, &rhs, &mut change_map)
                        };

                        let mut exceeded_graph_limit = false;
                        for (lhs_section_nodes, rhs_section_nodes) in possibly_changed {
                            init_next_prev(&lhs_section_nodes);
                            init_next_prev(&rhs_section_nodes);

                            match mark_syntax(
                                lhs_section_nodes.first().copied(),
                                rhs_section_nodes.first().copied(),
                                &mut change_map,
                                diff_options.graph_limit,
                            ) {
                                Ok(()) => {}
                                Err(ExceededGraphLimit {}) => {
                                    exceeded_graph_limit = true;
                                    break;
                                }
                            }
                        }

                        if exceeded_graph_limit {
                            let lhs_positions = line_parser::change_positions(lhs_src, rhs_src);
                            let rhs_positions = line_parser::change_positions(rhs_src, lhs_src);
                            (
                                FileFormat::TextFallback {
                                    reason: "exceeded DFT_GRAPH_LIMIT".into(),
                                },
                                lhs_positions,
                                rhs_positions,
                            )
                        } else {
                            fix_all_sliders(language, &lhs, &mut change_map);
                            fix_all_sliders(language, &rhs, &mut change_map);

                            let mut lhs_positions = syntax::change_positions(&lhs, &change_map);
                            let mut rhs_positions = syntax::change_positions(&rhs, &change_map);

                            if diff_options.ignore_comments {
                                let lhs_comments =
                                    tsp::comment_positions(&lhs_tree, lhs_src, &lang_config);
                                lhs_positions.extend(lhs_comments);

                                let rhs_comments =
                                    tsp::comment_positions(&rhs_tree, rhs_src, &lang_config);
                                rhs_positions.extend(rhs_comments);
                            }

                            (
                                FileFormat::SupportedLanguage(language),
                                lhs_positions,
                                rhs_positions,
                            )
                        }
                    }
                    Err(tsp::ExceededParseErrorLimit(error_count)) => {
                        let file_format = FileFormat::TextFallback {
                            reason: format!(
                                "{} {} parse error{}, exceeded DFT_PARSE_ERROR_LIMIT",
                                error_count,
                                language_name(language),
                                if error_count == 1 { "" } else { "s" }
                            ),
                        };

                        if diff_options.check_only {
                            return check_only_text(&file_format, display_path, lhs_src, rhs_src);
                        }

                        let lhs_positions = line_parser::change_positions(lhs_src, rhs_src);
                        let rhs_positions = line_parser::change_positions(rhs_src, lhs_src);
                        (file_format, lhs_positions, rhs_positions)
                    }
                },
                Err(tsp::ExceededByteLimit(num_bytes)) => {
                    let format_options = FormatSizeOptions::from(BINARY).decimal_places(1);
                    let file_format = FileFormat::TextFallback {
                        reason: format!(
                            "{} exceeded DFT_BYTE_LIMIT",
                            &format_size(num_bytes, format_options)
                        ),
                    };

                    if diff_options.check_only {
                        return check_only_text(&file_format, display_path, lhs_src, rhs_src);
                    }

                    let lhs_positions = line_parser::change_positions(lhs_src, rhs_src);
                    let rhs_positions = line_parser::change_positions(rhs_src, lhs_src);
                    (file_format, lhs_positions, rhs_positions)
                }
            }
        }
    };

    let opposite_to_lhs = opposite_positions(&lhs_positions);
    let opposite_to_rhs = opposite_positions(&rhs_positions);

    let hunks = matched_pos_to_hunks(&lhs_positions, &rhs_positions);
    let hunks = merge_adjacent(
        &hunks,
        &opposite_to_lhs,
        &opposite_to_rhs,
        lhs_src.max_line(),
        rhs_src.max_line(),
        display_options.num_context_lines as usize,
    );
    let has_syntactic_changes = !hunks.is_empty();

    let has_byte_changes = if lhs_src == rhs_src {
        None
    } else {
        Some((lhs_src.as_bytes().len(), rhs_src.as_bytes().len()))
    };

    DiffResult {
        extra_info: None,
        display_path: display_path.to_owned(),
        file_format,
        lhs_src: FileContent::Text(lhs_src.to_owned()),
        rhs_src: FileContent::Text(rhs_src.to_owned()),
        lhs_positions,
        rhs_positions,
        hunks,
        has_byte_changes,
        has_syntactic_changes,
    }
}
#[cfg(test)]
mod semantic_tests {
    use super::{
        diff_bytes_semantic, semantic_diff_options, DiffRequest, DiffStatus,
        SEMANTIC_DEFAULT_PARSE_ERROR_LIMIT, SEMANTIC_PARSE_ERROR_LIMIT_ENV,
    };

    #[test]
    fn semantic_diff_allows_best_effort_parse_errors_by_default() {
        if std::env::var_os(SEMANTIC_PARSE_ERROR_LIMIT_ENV).is_none() {
            assert_eq!(
                semantic_diff_options().parse_error_limit,
                SEMANTIC_DEFAULT_PARSE_ERROR_LIMIT
            );
        }
    }

    #[test]
    fn semantic_diff_includes_context_lines_for_snippets() {
        let result = diff_bytes_semantic(DiffRequest {
            display_path: "notes.txt",
            lhs_path: Some(std::path::Path::new("notes.txt")),
            rhs_path: Some(std::path::Path::new("notes.txt")),
            lhs_bytes: b"one\ntwo\nthree\n",
            rhs_bytes: b"one\nTWO\nthree\n",
        })
        .unwrap();

        assert_eq!(result.status, DiffStatus::Changed);
        assert_eq!(result.chunks.len(), 1);
        let lines = &result.chunks[0].lines;
        assert_eq!(lines.first().and_then(|line| line.lhs_line), Some(0));
        assert!(lines.iter().any(|line| line.lhs_line == Some(2)));
        assert!(lines.iter().any(|line| line.rhs_line == Some(2)));
    }

    #[test]
    fn semantic_diff_uses_paths_for_created_and_deleted_empty_files() {
        let created = diff_bytes_semantic(DiffRequest {
            display_path: "empty.txt",
            lhs_path: None,
            rhs_path: Some(std::path::Path::new("empty.txt")),
            lhs_bytes: b"",
            rhs_bytes: b"",
        })
        .unwrap();
        assert_eq!(created.status, DiffStatus::Created);

        let deleted = diff_bytes_semantic(DiffRequest {
            display_path: "empty.txt",
            lhs_path: Some(std::path::Path::new("empty.txt")),
            rhs_path: None,
            lhs_bytes: b"",
            rhs_bytes: b"",
        })
        .unwrap();
        assert_eq!(deleted.status, DiffStatus::Deleted);
    }

    #[test]
    fn semantic_diff_returns_changed_chunks_without_display_diff_result() {
        let result = diff_bytes_semantic(DiffRequest {
            display_path: "notes.txt",
            lhs_path: Some(std::path::Path::new("notes.txt")),
            rhs_path: Some(std::path::Path::new("notes.txt")),
            lhs_bytes: b"one\ntwo\n",
            rhs_bytes: b"one\nthree\n",
        })
        .unwrap();

        assert_eq!(result.status, DiffStatus::Changed);
        assert_eq!(result.chunks.len(), 1);
        assert!(!result.aligned_lines.is_empty());
    }
}
