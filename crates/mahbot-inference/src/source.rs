// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rust source as the record digest and the synthesis cache key see it: comments
//! and test-only items removed, whitespace outside literals normalized, literals
//! byte-for-byte. One lexer, so the two cannot disagree about what a comment or a
//! test is.

/// The code of a Rust source: comments removed, `#[cfg(test)]` and
/// `#[cfg(all(test, ...))]` items cut, and whitespace outside literals normalized.
///
/// A comment is replaced by a space, so adding, removing or rewording one cannot
/// move the result, however many lines it takes. The scan is literal-aware: a
/// comment marker inside a string, raw string or byte string is code, a character
/// literal is not a lifetime, and a lifetime is not a character literal; a literal's
/// own bytes are copied verbatim. A test-only item is removed whole, while the code
/// around it is kept, so a test edit is invisible too. Whitespace outside literals is
/// collapsed and blank lines dropped, so an indentation-only edit is invisible as
/// well. A change to a token still moves the result.
pub(crate) fn rust_code(source: &[u8]) -> Vec<u8> {
    normalize_code(&strip_test_items(&strip_comments(source)))
}

/// Whether `i` starts a literal — a string, a byte string, a raw string or a
/// character literal — and the end past it, or `None` when it starts an ordinary
/// token. A lifetime's quote is a one-byte "literal": the identifier after it is
/// code.
fn skip_literal(code: &[u8], i: usize) -> Option<usize> {
    match *code.get(i)? {
        b'"' => Some(skip_string(code, i)),
        b'\'' => Some(skip_char(code, i)),
        b'r' => skip_raw_string(code, i),
        b'b' => match code.get(i + 1) {
            Some(b'"') => Some(skip_string(code, i + 1)),
            Some(b'\'') => Some(skip_char(code, i + 1)),
            Some(b'r') => skip_raw_string(code, i),
            _ => None,
        },
        _ => None,
    }
}

/// [`rust_code`]'s first half: comments out, everything else byte-for-byte.
fn strip_comments(source: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(source.len());
    let mut i = 0;
    while i < source.len() {
        if source[i] == b'/' && source.get(i + 1) == Some(&b'/') {
            i += 2;
            while i < source.len() && source[i] != b'\n' {
                i += 1;
            }
            // At EOF there is no newline to separate the comment from what precedes
            // it; a space keeps the last token from fusing with anything after.
            if i == source.len() {
                out.push(b' ');
            }
        } else if source[i] == b'/' && source.get(i + 1) == Some(&b'*') {
            i += 2;
            let mut depth = 1usize;
            while i < source.len() && depth > 0 {
                if source[i] == b'/' && source.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if source[i] == b'*' && source.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            // A comment is whitespace to the lexer; one space keeps the tokens on
            // either side of it apart. No newline is kept, so a comment's own line
            // count cannot move the result.
            out.push(b' ');
        } else if let Some(end) = skip_literal(source, i) {
            out.extend_from_slice(&source[i..end]);
            i = end;
        } else {
            out.push(source[i]);
            i += 1;
        }
    }
    out
}

/// [`rust_code`]'s second half: each `#[cfg(test)]` item removed whole. The scan
/// skips literals, so an attribute in a string is not one.
fn strip_test_items(code: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(code.len());
    let mut i = 0;
    while i < code.len() {
        if code[i] == b'#'
            && let Some(attribute) = cfg_test_end(code, i)
        {
            let end = item_end(code, attribute);
            // The removed item's newlines go with it; a space separates the code
            // on either side, and the line normalizer drops the rest.
            out.push(b' ');
            i = end;
            continue;
        }
        if let Some(end) = skip_literal(code, i) {
            out.extend_from_slice(&code[i..end]);
            i = end;
        } else {
            out.push(code[i]);
            i += 1;
        }
    }
    out
}

/// Collapse whitespace outside literals: one space between tokens on a line, no
/// leading or trailing space, and no blank lines. Literal bytes are copied whole.
fn normalize_code(code: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(code.len());
    let mut i = 0;
    let mut pending_space = false;
    let mut line_has_code = false;
    while i < code.len() {
        let byte = code[i];
        if byte == b'\n' {
            if line_has_code {
                out.push(b'\n');
            }
            line_has_code = false;
            pending_space = false;
            i += 1;
            continue;
        }
        if byte.is_ascii_whitespace() {
            if line_has_code {
                pending_space = true;
            }
            i += 1;
            continue;
        }
        let end = skip_literal(code, i).unwrap_or(i + 1);
        if pending_space {
            out.push(b' ');
            pending_space = false;
        }
        out.extend_from_slice(&code[i..end]);
        line_has_code = true;
        i = end;
    }
    if line_has_code {
        out.push(b'\n');
    }
    out
}

/// The end of the outer attribute `#[...]` starting at `i` (a `#`), or `None` when
/// it is not one. Literals inside the attribute are skipped, so a `]` in one cannot
/// close it.
fn attribute_end(code: &[u8], i: usize) -> Option<usize> {
    if code.get(i) != Some(&b'#') || code.get(i + 1) != Some(&b'[') {
        return None;
    }
    let mut j = i + 2;
    let mut depth = 1usize;
    while j < code.len() {
        if let Some(end) = skip_literal(code, j) {
            j = end;
            continue;
        }
        match code[j] {
            b'[' => {
                depth += 1;
                j += 1;
            }
            b']' => {
                depth -= 1;
                j += 1;
                if depth == 0 {
                    return Some(j);
                }
            }
            _ => j += 1,
        }
    }
    None
}

/// The end of a test-only `cfg` attribute at `i` — `cfg(test)` or
/// `cfg(all(test, ...))` — or `None` for any other attribute. `all` requires every
/// predicate, so `test` among them means the item compiles only in a test build;
/// `any(test, ...)` does not, and is left in the digest.
fn cfg_test_end(code: &[u8], i: usize) -> Option<usize> {
    let end = attribute_end(code, i)?;
    let body: Vec<u8> = code[i + 2..end - 1]
        .iter()
        .copied()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    let test_only =
        body == b"cfg(test)" || body == b"cfg(all(test))" || body.starts_with(b"cfg(all(test,");
    test_only.then_some(end)
}

/// The end of the item a test-only attribute guards, starting at `from` just past
/// the attribute: further attributes are skipped, then the item runs to its closing
/// brace (taking a `;` that follows it) or its terminating `;`/`,` at the top level.
/// A brace-terminated item's header may hold a top-level comma, which is skipped.
/// Literals are skipped so their braces and semicolons do not count. An item whose
/// end cannot be found ends at EOF.
fn item_end(code: &[u8], mut i: usize) -> usize {
    loop {
        while i < code.len() && code[i].is_ascii_whitespace() {
            i += 1;
        }
        if code.get(i) == Some(&b'#')
            && let Some(end) = attribute_end(code, i)
        {
            i = end;
            continue;
        }
        break;
    }
    // A brace-terminated item can hold a top-level comma before its body — a
    // function's generic return type or where clause, a generic `impl`/`struct`/
    // `enum`/`trait`/`union` header. The item still ends at its body, not there.
    let mut depth = 0usize;
    while i < code.len() {
        if let Some(end) = skip_literal(code, i) {
            i = end;
            continue;
        }
        match code[i] {
            b'(' | b'[' | b'{' => {
                depth += 1;
                i += 1;
            }
            b')' | b']' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                i += 1;
                if depth == 0 {
                    // A braced `use` (or a braced `const` initializer) ends at the
                    // `;` that follows; take it so no separator is left.
                    let mut after = i;
                    while after < code.len() && code[after].is_ascii_whitespace() {
                        after += 1;
                    }
                    return if code.get(after) == Some(&b';') {
                        after + 1
                    } else {
                        i
                    };
                }
            }
            b',' if depth == 0 && brace_before_semicolon(code, i) => i += 1,
            b';' | b',' if depth == 0 => return i + 1,
            _ => i += 1,
        }
    }
    code.len()
}

/// Whether the next `{` at the top level comes before the next `;` — so a comma at
/// `i` is part of a brace-terminated item's header, not the item's end. A closing
/// brace before either ends a field or a match arm: its comma is the item's end.
fn brace_before_semicolon(code: &[u8], mut i: usize) -> bool {
    let mut depth = 0usize;
    while i < code.len() {
        if let Some(end) = skip_literal(code, i) {
            i = end;
            continue;
        }
        match code[i] {
            b'(' | b'[' => {
                depth += 1;
                i += 1;
            }
            b')' | b']' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b'{' if depth == 0 => return true,
            b'}' | b';' if depth == 0 => return false,
            _ => i += 1,
        }
    }
    false
}

/// The end of the string literal starting at `i` (the opening `"`): just past the
/// closing quote, or EOF when it is unterminated.
fn skip_string(code: &[u8], mut i: usize) -> usize {
    i += 1;
    while i < code.len() {
        match code[i] {
            b'\\' => i = (i + 2).min(code.len()),
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    code.len()
}

/// The end of the character literal at `i` (the opening `'`), or `i + 1` when the
/// quote opens a lifetime instead — the identifier after a lifetime is code.
fn skip_char(code: &[u8], i: usize) -> usize {
    let mut j = i + 1;
    if j >= code.len() {
        return i + 1;
    }
    if code[j] == b'\\' {
        j += 1;
        if code.get(j) == Some(&b'u') {
            j += 1;
            while j < code.len() && code[j] != b'}' {
                j += 1;
            }
            j += 1;
        } else {
            j += 1;
        }
        while j < code.len() && code[j] != b'\'' {
            j += 1;
        }
        return (j + 1).min(code.len());
    }
    let after = j + utf8_len(code[j]);
    if code.get(after) == Some(&b'\'') {
        after + 1
    } else {
        i + 1
    }
}

/// The end of a raw string at `i` — `r"..."`, `r#"..."#`, `br"..."` and their
/// hash-run forms — or `None` when the byte is an ordinary identifier's `r`/`b`.
fn skip_raw_string(code: &[u8], i: usize) -> Option<usize> {
    let mut j = match code[i] {
        b'r' => i + 1,
        b'b' if code.get(i + 1) == Some(&b'r') => i + 2,
        _ => return None,
    };
    let mut hashes = 0;
    while code.get(j) == Some(&b'#') {
        hashes += 1;
        j += 1;
    }
    if code.get(j) != Some(&b'"') {
        return None;
    }
    j += 1;
    loop {
        if j >= code.len() {
            return Some(code.len());
        }
        if code[j] == b'"' {
            let mut k = j + 1;
            let mut seen = 0;
            while seen < hashes && code.get(k) == Some(&b'#') {
                seen += 1;
                k += 1;
            }
            if seen == hashes {
                return Some(k);
            }
        }
        j += 1;
    }
}

/// The bytes one UTF-8 character starting with `first` occupies.
fn utf8_len(first: u8) -> usize {
    match first {
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A comment edit cannot move the stripped code, whatever form the comment takes,
    /// while a comment marker inside a literal is code and a lifetime is not a
    /// character literal.
    #[test]
    fn a_comment_is_stripped_and_a_literal_is_not() {
        let source = br##"
fn a<'a>(x: &'a str) -> char {
    // a line comment
    /* a block /* nested */ comment */
    let s = "// not a comment";
    let b = b"/* not a comment */";
    let r = r#"/* not a comment */"#;
    let c = '/';
    let escaped = '\'';
    let _ = (s, b, r, c, escaped);
    'a'
}
"##;
        let text = String::from_utf8(rust_code(source)).expect("UTF-8");
        for kept in [
            "\"// not a comment\"",
            "b\"/* not a comment */\"",
            "r#\"/* not a comment */\"#",
            "let c = '/';",
            "let escaped = '\\'';",
            "fn a<'a>(x: &'a str)",
        ] {
            assert!(text.contains(kept), "`{kept}` was stripped:\n{text}");
        }
        for stripped in ["a line comment", "a block", "nested */ comment"] {
            assert!(!text.contains(stripped), "`{stripped}` survived:\n{text}");
        }

        let one = b"fn a() {}\n// one\n";
        let two = b"fn a() {}\n// two\n";
        assert_eq!(
            rust_code(one),
            rust_code(two),
            "a comment edit moves the code"
        );
        assert_eq!(
            rust_code(b"fn a() {}\n// one\nfn b() {}\n"),
            rust_code(b"fn a() {}\n// two\n// more\nfn b() {}\n"),
            "a comment's line count moves the code"
        );
        assert_eq!(
            rust_code(b"fn a() {}\n/* one */\nfn b() {}\n"),
            rust_code(b"fn a() {}\n/* one\n   two\n   three */\nfn b() {}\n"),
            "a block comment's line count moves the code"
        );
    }

    /// A `#[cfg(test)]` item is removed whole and the code on either side of it is
    /// kept; an attribute in a string is not an attribute. A `cfg(all(test, ...))`
    /// item is test code whole.
    #[test]
    fn a_test_item_is_cut_and_the_code_around_it_is_kept() {
        let source = b"fn before() {}\n\
                       #[cfg(test)]\n\
                       mod tests {\n    fn t() {}\n}\n\
                       #[cfg(test)]\n\
                       use std::fmt;\n\
                       fn after() {}\n\
                       let s = \"#[cfg(test)]\";\n";
        let text = String::from_utf8(rust_code(source)).expect("UTF-8");
        for kept in ["fn before()", "fn after()", "\"#[cfg(test)]\""] {
            assert!(text.contains(kept), "`{kept}` was removed:\n{text}");
        }
        for cut in ["mod tests", "fn t()", "use std::fmt"] {
            assert!(!text.contains(cut), "`{cut}` survived:\n{text}");
        }

        let one = b"fn a() {}\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn b() {}\n";
        let two =
            b"fn a() {}\n#[cfg(test)]\nmod tests {\n    fn t() {}\n    fn u() {}\n}\nfn b() {}\n";
        assert_eq!(rust_code(one), rust_code(two), "a test edit moves the code");

        let guarded = b"fn before() {}\n\
                        #[cfg(all(test, feature = \"std\"))]\n\
                        fn helper() {}\n\
                        fn after() {}\n";
        let text = String::from_utf8(rust_code(guarded)).expect("UTF-8");
        assert!(
            text.contains("fn before()") && text.contains("fn after()"),
            "{text}"
        );
        assert!(!text.contains("fn helper()"), "{text}");

        // A generic return type's comma is inside the function's header, not its end.
        let generic = b"fn before() {}\n\
                        #[cfg(test)]\n\
                        fn helper(a: u8) -> Result<Vec<u8>, String> { Ok(vec![a]) }\n\
                        fn after() {}\n";
        let text = String::from_utf8(rust_code(generic)).expect("UTF-8");
        assert!(
            text.contains("fn before()") && text.contains("fn after()"),
            "{text}"
        );
        assert!(!text.contains("fn helper"), "{text}");
        assert!(!text.contains("String>"), "{text}");

        // A braced `use` takes its terminating `;` with it, leaving no stray token.
        let braced = b"fn before() {}\n\
                       #[cfg(test)]\n\
                       use std::{fmt, io};\n\
                       fn after() {}\n";
        let text = String::from_utf8(rust_code(braced)).expect("UTF-8");
        assert!(
            text.contains("fn before()") && text.contains("fn after()"),
            "{text}"
        );
        assert!(!text.contains("fmt") && !text.contains(';'), "{text}");

        // A braced `const` initializer takes its terminating `;` too.
        let konst = b"fn before() {}\n\
                      #[cfg(test)]\n\
                      const X: S = S { a: 1 };\n\
                      fn after() {}\n";
        let text = String::from_utf8(rust_code(konst)).expect("UTF-8");
        assert!(
            text.contains("fn before()") && text.contains("fn after()"),
            "{text}"
        );
        assert!(!text.contains("const X") && !text.contains(';'), "{text}");

        // A field or a match arm ends at its comma, not at the enclosing brace.
        let field = b"struct S {\n    #[cfg(test)]\n    a: u32,\n    b: u32,\n}\n";
        let text = String::from_utf8(rust_code(field)).expect("UTF-8");
        assert!(text.contains("b: u32"), "{text}");
        assert!(!text.contains("a: u32"), "{text}");
        let arm = b"fn f(x: u8) -> u8 {\n    match x {\n        #[cfg(test)]\n        0 => 1,\n        _ => 2,\n    }\n}\n";
        let text = String::from_utf8(rust_code(arm)).expect("UTF-8");
        assert!(text.contains("_ => 2"), "{text}");
        assert!(!text.contains("0 => 1"), "{text}");
    }
}
