//! Rust port of Hikari, reference commit 90aef085a53d6048cbd19de7f4c9c53a6f3c8960.
//!
//! Board physics, movement, evaluation and the retained layered DAG are translated
//! from the original implementation.

pub mod board;
pub mod eval;
#[cfg(feature = "fast-registration")]
mod graph_storage;
pub mod movegen;
#[cfg(feature = "profile-search")]
pub mod profile_clock;
pub mod protocol;
pub mod search;
pub mod ultra;

pub const REFERENCE_COMMIT: &str = "90aef085a53d6048cbd19de7f4c9c53a6f3c8960";
