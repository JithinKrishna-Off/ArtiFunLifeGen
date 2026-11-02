//! FlyBrain neural runtime: CSR connectome storage + deterministic LIF
//! stepping + source-independent MaleCNS import pipeline.
//!
//! Module map:
//! - [`connectome`]   CSR sparse graph + side tables + structural identity
//! - [`state`]        per-organism LIF state + delayed-spike ring
//! - [`runtime`]      `FlyBrainRuntime` deterministic 1 ms stepping
//! - [`sensory`]      documented sensory channels + rate/linear encoders
//! - [`motor`]        bounded motor decoding from output-cell rates
//! - [`fixtures`]     SYNTHETIC test circuit (never presented as real data)
//! - [`connectome_format`]    normalized TSV / sidecar schema (source-independent)
//! - [`connectome_import`]    streaming TSV -> compact index -> CSR import
//! - [`malecns_bins`]         binary-array flat package importer (external
//!                            malecns166k dir; + Queen subnetwork extraction)
//! - [`connectome_binary`]    versioned artifact with CRC + identity hash
//! - [`subnetwork`]   region / cell-type filters and mapping suggestions
//! - [`provenance`]   dataset facts + field-level provenance tags

pub mod connectome;
pub mod connectome_binary;
pub mod connectome_format;
pub mod connectome_import;
pub mod fixtures;
pub mod malecns_bins;
pub mod motor;
pub mod provenance;
pub mod queen_adapter;
pub mod queen_types;
pub mod runtime;
pub mod sensory;
pub mod state;
pub mod subnetwork;
