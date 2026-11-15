// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! A deferring open leaves a metadata array's bytes in the file: what
//! [`GgufFile::skipped`] hands back must equal what a materialising open would
//! have parsed, and a declared length the file cannot back must still be
//! refused rather than trusted.
use rlx_gguf::{GGUF_MAGIC, GgufFile, GgufWriter, MetaValue};

const TOKENS: [&str; 4] = ["a", "bc", "d\u{00e9}", "long token"];

fn sample() -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.set_arch("llama");
    w.set_meta(
        "tokenizer.ggml.tokens",
        MetaValue::Array(
            TOKENS
                .iter()
                .map(|t| MetaValue::String((*t).into()))
                .collect(),
        ),
    );
    w.set_meta(
        "tokenizer.ggml.token_type",
        MetaValue::Array(vec![
            MetaValue::I32(1),
            MetaValue::I32(3),
            MetaValue::I32(1),
            MetaValue::I32(5),
        ]),
    );
    // Written after the arrays: a skipped payload must leave the stream where a
    // materialised one does, or the next key is read out of the wrong bytes.
    w.set_meta("general.name", MetaValue::String("after".into()));
    let mut out = std::io::Cursor::new(Vec::new());
    w.write(&mut out).unwrap();
    out.into_inner()
}

/// A header this crate's own writer would not produce: one array named `x`, of
/// element type `inner`, with a declared count and a payload of the test's
/// choosing. Padded to the default alignment so the data segment is in the file.
fn hostile(inner: u32, count: u64, payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&0u64.to_le_bytes()); // tensors
    b.extend_from_slice(&1u64.to_le_bytes()); // kv
    b.extend_from_slice(&1u64.to_le_bytes()); // key length
    b.push(b'x');
    b.extend_from_slice(&9u32.to_le_bytes()); // array
    b.extend_from_slice(&inner.to_le_bytes());
    b.extend_from_slice(&count.to_le_bytes());
    b.extend_from_slice(payload);
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b
}

fn open_skipping(bytes: &[u8], skip: &[&str]) -> (tempfile::TempPath, GgufFile) {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut f, bytes).unwrap();
    let path = f.into_temp_path();
    let file = GgufFile::from_path_mmap_skipping(&path, skip).expect("the file opens");
    (path, file)
}

/// Why a deferring open of `bytes` refuses the file.
fn refusal(bytes: &[u8]) -> String {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut f, bytes).unwrap();
    let path = f.into_temp_path();
    match GgufFile::from_path_mmap_skipping(&path, &["x"]) {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!("the file is refused"),
    }
}

#[test]
fn a_skipped_array_reads_what_a_materialised_one_holds() {
    let bytes = sample();
    let materialised = GgufFile::from_bytes(bytes.clone()).expect("the same file materialised");
    let (_path, deferred) = open_skipping(
        &bytes,
        &["tokenizer.ggml.tokens", "tokenizer.ggml.token_type"],
    );

    for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type"] {
        assert!(
            !deferred.metadata.contains_key(key),
            "`{key}` must not be materialised by a deferring open"
        );
        assert!(deferred.skipped(key).is_some(), "`{key}` is in the file");
    }

    let tokens = deferred.skipped("tokenizer.ggml.tokens").unwrap();
    let read: Vec<&str> = tokens.strings().unwrap().map(Result::unwrap).collect();
    assert_eq!(read, TOKENS);
    assert_eq!(tokens.count(), TOKENS.len());

    let kinds = deferred.skipped("tokenizer.ggml.token_type").unwrap();
    let read: Vec<u32> = kinds.u32s().unwrap().map(Result::unwrap).collect();
    assert_eq!(read, vec![1, 3, 1, 5]);

    // The rest of the header still parses, the key after the arrays included: a
    // deferred value is not a lost one and does not shift the ones beside it.
    for key in ["general.architecture", "general.name"] {
        assert_eq!(
            materialised.metadata.get(key).and_then(MetaValue::as_str),
            deferred.metadata.get(key).and_then(MetaValue::as_str),
            "{key}"
        );
    }
}

/// The element type is still checked: a table written as one thing cannot be read
/// as another, exactly as `MetaValue::as_str` / `as_u32` refuse it. The check is
/// the reader's, not an element's, so an empty array of the wrong type is refused
/// too rather than read as an empty one.
#[test]
fn a_mistyped_element_is_refused() {
    let (_path, file) = open_skipping(&hostile(4, 1, &4u32.to_le_bytes()), &["x"]);
    let err = format!(
        "{:#}",
        file.skipped("x")
            .unwrap()
            .strings()
            .err()
            .expect("a u32 is not a string")
    );
    assert!(err.contains("not a string"), "{err}");

    for count in [2, 0] {
        let (_path, file) = open_skipping(&hostile(0, count, &[1, 2]), &["x"]);
        let err = format!(
            "{:#}",
            file.skipped("x")
                .unwrap()
                .u32s()
                .err()
                .expect("a u8 array is not a u32 array")
        );
        assert!(err.contains("not a u32"), "{err}");
    }
}

#[test]
fn a_skipped_array_the_file_cannot_back_is_refused() {
    // A string array: the payload is walked string by string, so a count past
    // the end is caught by the walk.
    let err = refusal(&hostile(8, 4, &1_000u64.to_le_bytes()));
    assert!(err.contains("truncated") || err.contains("ends"), "{err}");

    // A fixed-width array: the walk reads its elements one at a time, so a count
    // past the end is an EOF once the file's bytes — here the alignment padding,
    // which covers three of the four bytes of the fourth element — run out.
    let err = refusal(&hostile(4, u64::MAX, &[]));
    assert!(err.contains("-element array of type 4"), "{err}");
}

#[test]
fn a_negative_element_is_refused_rather_than_wrapped() {
    let b = hostile(
        5,
        2,
        &(-1i32)
            .to_le_bytes()
            .into_iter()
            .chain(7i32.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let (_path, file) = open_skipping(&b, &["x"]);
    let kinds = file.skipped("x").expect("the array is in the file");
    let mut it = kinds.u32s().expect("an integer array");
    assert!(
        it.next().expect("one element").is_err(),
        "a negative element is not a u32"
    );
    assert_eq!(it.next().expect("the second element").unwrap(), 7);
}
