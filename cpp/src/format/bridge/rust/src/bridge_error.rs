// Copyright 2025 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Shared error classification for the Rust cxx bridges.
//!
//! The cxx boundary can only carry an error as a display string
//! (`rust::Error::what()`), which used to destroy the typed error the Rust
//! side already had (`lance::Error` distinguishes not-found / corruption /
//! retryable contention; the C++ side then guessed a blanket classification).
//! To keep the classification across the string-only channel, an error code is
//! embedded into the message with a marker prefix that the C++ side parses and
//! strips (see cpp `bridge_error.h`), the same mechanism the vortex bridge
//! established in `filesystem_c.rs`.
//!
//! Code space carried by the marker:
//! * LOON / ExtendStatusCode values (`ffi_error_code.h`): 12 = file-not-found,
//!   101-112 = AWS/transient/txn extend codes. The C++ side rebuilds the
//!   matching `ExtendStatusDetail` (or an ENOENT detail for 12).
//! * Bridge-private values (>= 1000, never cross the C ABI): the C++ side
//!   converts them straight into an arrow StatusCode and they cease to exist.
//!
//! Classification discipline ("producer owns classification", conservative):
//! only signals the producer positively identifies are tagged; everything else
//! stays untagged and lands in the consumer's non-retriable fallback bucket.
//! Never invent retriability.

use lance::Error as LanceError;

/// Must stay byte-identical to the vortex marker in `filesystem_c.rs` and the
/// parser constant in cpp `bridge_error.cpp` — one marker, one parser.
pub const BRIDGE_ERRCODE_MARKER: &str = "__LOON_VORTEX_FFI_ERRCODE__=";

/// Mirrors LOON_FILE_NOT_FOUND in `ffi_error_code.h`.
pub const LOON_FILE_NOT_FOUND: i32 = 12;
/// Mirror of the ExtendStatusCode transient tags (`ffi_error_code.h` 101-112).
pub const LOON_AWS_ERROR_PRECONDITION_FAILED: i32 = 103;
pub const LOON_AWS_ERROR_ACCESS_DENIED: i32 = 105;
pub const LOON_TRANSIENT_THROTTLING: i32 = 109;

/// Bridge-private codes (>= 1000): decoded by cpp `bridge_error.cpp` into an
/// arrow StatusCode, never forwarded as an FFI error code.
pub const BRIDGE_ERRCODE_DATA_CORRUPT: i32 = 1001;
pub const BRIDGE_ERRCODE_NOT_SUPPORTED: i32 = 1002;

/// Error type used by the cxx bridge functions. cxx renders it with `Display`
/// into `rust::Error::what()`; the marker survives that trip.
#[derive(Debug)]
pub struct BridgeError {
    pub code: Option<i32>,
    pub msg: String,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.code {
            Some(code) => write!(f, "{BRIDGE_ERRCODE_MARKER}{code}; {}", self.msg),
            None => write!(f, "{}", self.msg),
        }
    }
}

impl std::error::Error for BridgeError {}

/// Alias used to switch a whole bridge impl module to classified errors: the
/// `?` operator converts `lance::Error` (and `ArrowError`) via the `From`
/// impls below.
pub type BridgeResult<T> = std::result::Result<T, BridgeError>;

/// Walk the `source()` chain looking for a typed [`object_store::Error`].
/// Lance boxes the object_store error directly today, but other layers can
/// sit in between, and a single-level `downcast_ref` on the top of the chain
/// would silently classify nothing.
fn find_object_store_error<'a>(
    root: &'a (dyn std::error::Error + 'static),
) -> Option<&'a object_store::Error> {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(root);
    while let Some(err) = current {
        if let Some(os_err) = err.downcast_ref::<object_store::Error>() {
            return Some(os_err);
        }
        current = err.source();
    }
    None
}

/// Positively identify rate-limiting/throttling from rendered error text.
/// Matches only the fixed strings object_store's retry layer and the S3/GCS
/// throttle responses actually emit ("status code: 429", "SlowDown", ...);
/// anything else returns `None` (never invent retriability).
fn classify_throttling_message(msg: &str) -> Option<i32> {
    let msg = msg.to_ascii_lowercase();
    let throttled = msg.contains("status code: 429")
        || msg.contains("status code: 503")
        || msg.contains("too many requests")
        || msg.contains("service unavailable")
        || msg.contains("slowdown")
        || msg.contains("throttl");
    throttled.then_some(LOON_TRANSIENT_THROTTLING)
}

/// Classify a `lance::Error` into a marker code. `None` = not positively
/// identified -> stays untagged -> conservative non-retriable fallback on the
/// consumer side.
pub fn classify_lance_error(e: &LanceError) -> Option<i32> {
    match e {
        // The object/dataset/index/ref/version is gone. Retrying hits the same
        // store and fails identically; consumers can distinguish "data
        // missing" from a generic storage failure.
        LanceError::NotFound { .. }
        | LanceError::DatasetNotFound { .. }
        | LanceError::IndexNotFound { .. }
        | LanceError::RefNotFound { .. }
        | LanceError::VersionNotFound { .. }
        | LanceError::FieldNotFound { .. } => Some(LOON_FILE_NOT_FOUND),
        // Permanent data problems: retrying re-reads the same bytes.
        LanceError::CorruptFile { .. }
        | LanceError::SchemaMismatch { .. }
        | LanceError::Schema { .. } => Some(BRIDGE_ERRCODE_DATA_CORRUPT),
        LanceError::NotSupported { .. } => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
        // Lance itself declares these retryable: the failed attempt is spent,
        // but a fresh attempt (new commit round) can succeed. This is the
        // producer's own classification, not invented here.
        LanceError::RetryableCommitConflict { .. } | LanceError::TooMuchWriteContention { .. } => {
            Some(LOON_TRANSIENT_THROTTLING)
        }
        // IO wraps the underlying object_store error as a boxed source;
        // downcast to recover the typed variant. The downcast walks the whole
        // source() chain: lance may box the object_store error behind extra
        // layers, and a single-level downcast_ref would silently miss it.
        LanceError::IO { source, .. } => match find_object_store_error(source.as_ref()) {
            Some(object_store::Error::NotFound { .. }) => Some(LOON_FILE_NOT_FOUND),
            Some(
                object_store::Error::PermissionDenied { .. }
                | object_store::Error::Unauthenticated { .. },
            ) => Some(LOON_AWS_ERROR_ACCESS_DENIED),
            Some(object_store::Error::Precondition { .. }) => {
                Some(LOON_AWS_ERROR_PRECONDITION_FAILED)
            }
            Some(
                object_store::Error::NotSupported { .. }
                | object_store::Error::NotImplemented { .. },
            ) => Some(BRIDGE_ERRCODE_NOT_SUPPORTED),
            // object_store's retry layer folds retry-exhausted 429/503 (and
            // S3 `SlowDown` bodies) into Generic. Its RetryError source type
            // is pub(crate) — no typed downcast is possible — but throttling
            // is still a recoverable signal for the caller's longer-horizon
            // retry policy, so recover it from the rendered text.
            Some(object_store::Error::Generic { source, .. }) => {
                classify_throttling_message(&source.to_string())
            }
            // Other typed variants carry no positive transient/permanent
            // signal, so stay untagged (conservative).
            Some(_) => None,
            // No typed object_store error anywhere in the chain — either an
            // opaque wrapper, or an object_store version skew between this
            // crate and lance making every downcast fail. Fall back to the
            // rendered text so a genuine rate-limit keeps its retryable tag
            // instead of silently degrading to the non-retriable bucket.
            None => classify_throttling_message(&source.to_string()),
        },
        // InvalidInput deliberately NOT tagged as caller input: the strings we
        // feed lance are mostly assembled by this library itself, so blaming
        // the caller would misroute retries (see the 2007/2020/2021
        // demotions). Left untagged pending a producer-site audit.
        _ => None,
    }
}

impl From<LanceError> for BridgeError {
    fn from(e: LanceError) -> Self {
        BridgeError {
            code: classify_lance_error(&e),
            msg: e.to_string(),
        }
    }
}

impl From<arrow58::error::ArrowError> for BridgeError {
    fn from(e: arrow58::error::ArrowError) -> Self {
        BridgeError {
            code: None,
            msg: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generic_os_error(msg: &str) -> object_store::Error {
        object_store::Error::Generic {
            store: "S3",
            source: msg.to_string().into(),
        }
    }

    #[test]
    fn retry_exhausted_429_is_tagged_transient_throttling() {
        let e = LanceError::from(generic_os_error(
            "Error performing GET https://bucket/key in 90s, after 10 retries, \
             max_retries: 10, retry_timeout: 180s  - Server returned non-2xx \
             status code: 429 Too Many Requests: rate exceeded",
        ));
        assert_eq!(classify_lance_error(&e), Some(LOON_TRANSIENT_THROTTLING));
    }

    #[test]
    fn s3_slowdown_body_is_tagged_transient_throttling() {
        let e = LanceError::from(generic_os_error(
            "Server returned error response: <?xml version=\"1.0\"?><Error>\
             <Code>SlowDown</Code><Message>Please reduce your request rate.</Message></Error>",
        ));
        assert_eq!(classify_lance_error(&e), Some(LOON_TRANSIENT_THROTTLING));
    }

    #[test]
    fn generic_without_throttle_signal_stays_untagged() {
        let e = LanceError::from(generic_os_error("connection reset by peer"));
        assert_eq!(classify_lance_error(&e), None);
    }

    #[test]
    fn typed_not_found_found_through_wrapping_layers() {
        #[derive(Debug)]
        struct Wrapper(object_store::Error);
        impl std::fmt::Display for Wrapper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "wrapped: {}", self.0)
            }
        }
        impl std::error::Error for Wrapper {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let os_err = object_store::Error::NotFound {
            path: "bucket/key".into(),
            source: "object gone".to_string().into(),
        };
        let e = LanceError::io_source(Box::new(Wrapper(os_err)));
        assert_eq!(classify_lance_error(&e), Some(LOON_FILE_NOT_FOUND));
    }

    #[test]
    fn downcast_failure_falls_back_to_throttle_text() {
        // Source is not an object_store::Error at all (models an opaque
        // wrapper or a version-skewed object_store type): the rendered text
        // still carries the rate-limit evidence.
        let e = LanceError::io_source(
            "Server returned non-2xx status code: 503 Service Unavailable"
                .to_string()
                .into(),
        );
        assert_eq!(classify_lance_error(&e), Some(LOON_TRANSIENT_THROTTLING));
    }

    #[test]
    fn downcast_failure_without_signal_stays_untagged() {
        let e = LanceError::io_source("disk quota exceeded".to_string().into());
        assert_eq!(classify_lance_error(&e), None);
    }
}
