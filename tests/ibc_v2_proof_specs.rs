//! IBC v2 phase-0 spike: can a Penumbra (cnidarium/JMT) ICS23 proof be
//! verified by a client that hardcodes the Cosmos proof specs?
//!
//! Context: `cosmos/solidity-ibc-eureka`'s membership program
//! (`packages/tendermint-light-client/membership/src/lib.rs`) calls
//! `MerkleProof::verify_membership(&ProofSpecs::cosmos(), ..)`, i.e.
//! IAVL + simple-merkle, with no way to supply specs from the client state.
//! Penumbra's IBC state lives in a JMT substore with two identical JMT specs.
//!
//! This test shows, on a real cnidarium proof:
//!   1. membership and non-membership verify under Penumbra's JMT specs;
//!   2. the same proofs fail under the Cosmos (IAVL + tendermint) specs.
//!
//! Conclusion: the proof-spec source must change on the Eureka side (make
//! specs part of the client state / program input). It cannot be worked
//! around by cnidarium emitting a different proof: the hash prefixes, the
//! empty-child placeholder and `prehash_key_before_comparison` are properties
//! of the JMT itself.

use cnidarium::{StateDelta, StateWrite, Storage};
use ibc_types::core::commitment::{MerklePath, MerkleRoot};
use once_cell::sync::Lazy;

/// Penumbra's `IBC_PROOF_SPECS` (vendored in
/// `penumbra/crates/core/component/ibc/src/prefix.rs`): two JMT specs, one
/// per layer of cnidarium's substore proof.
static PENUMBRA_PROOF_SPECS: Lazy<Vec<ics23::ProofSpec>> =
    Lazy::new(|| vec![cnidarium::ics23_spec(), cnidarium::ics23_spec()]);

/// What `ibc_core_commitment_types::specs::ProofSpecs::cosmos()` expands to,
/// and what the Eureka membership program hardcodes.
static COSMOS_PROOF_SPECS: Lazy<Vec<ics23::ProofSpec>> =
    Lazy::new(|| vec![ics23::iavl_spec(), ics23::tendermint_spec()]);

#[tokio::test]
async fn penumbra_jmt_proofs_verify_under_jmt_specs_but_not_cosmos_specs() -> anyhow::Result<()> {
    let tmpdir = tempfile::tempdir()?;
    // Penumbra keeps IBC state in the `ibc-data` substore (ibc/src/prefix.rs).
    let storage = Storage::load(tmpdir.keep(), vec!["ibc-data".to_string()]).await?;

    let mut delta = StateDelta::new(storage.latest_snapshot());
    // An IBC v2 packet commitment path shape: commitments/{clientId}/{sequence}.
    let key = "ibc-data/commitments/07-tendermint-0/1".to_string();
    let value = [0xABu8; 32].to_vec();
    delta.put_raw(key.clone(), value.clone());
    storage.commit(delta).await?;

    let snapshot = storage.latest_snapshot();
    let root = MerkleRoot {
        hash: snapshot.root_hash().await?.0.to_vec(),
    };

    // --- membership ---------------------------------------------------------
    let (got, proof) = snapshot.get_with_proof(key.clone().into_bytes()).await?;
    assert_eq!(got.as_deref(), Some(value.as_slice()));
    let path = MerklePath {
        key_path: vec![
            "ibc-data".to_string(),
            "commitments/07-tendermint-0/1".to_string(),
        ],
    };

    proof
        .verify_membership(&PENUMBRA_PROOF_SPECS, root.clone(), path.clone(), value.clone(), 0)
        .expect("membership verifies under Penumbra's JMT specs");

    let cosmos_result =
        proof.verify_membership(&COSMOS_PROOF_SPECS, root.clone(), path.clone(), value.clone(), 0);
    assert!(
        cosmos_result.is_err(),
        "a JMT membership proof must NOT verify under the hardcoded Cosmos (IAVL+tendermint) specs"
    );

    // --- non-membership (needed for IBC v2 timeouts) --------------------------
    let absent = "ibc-data/receipts/07-tendermint-0/99".to_string();
    let (got, nex_proof) = snapshot.get_with_proof(absent.into_bytes()).await?;
    assert!(got.is_none());
    let absent_path = MerklePath {
        key_path: vec![
            "ibc-data".to_string(),
            "receipts/07-tendermint-0/99".to_string(),
        ],
    };

    nex_proof
        .verify_non_membership(&PENUMBRA_PROOF_SPECS, root.clone(), absent_path.clone())
        .expect("non-membership verifies under Penumbra's JMT specs");

    let cosmos_result = nex_proof.verify_non_membership(&COSMOS_PROOF_SPECS, root, absent_path);
    assert!(
        cosmos_result.is_err(),
        "a JMT non-membership proof must NOT verify under the hardcoded Cosmos specs"
    );

    // Shape facts the Eureka client state / program would have to carry.
    assert_eq!(PENUMBRA_PROOF_SPECS.len(), 2, "two-layer proof: substore + root store");
    let spec = &PENUMBRA_PROOF_SPECS[0];
    assert!(spec.prehash_key_before_comparison);
    assert_eq!(spec.leaf_spec.as_ref().unwrap().prefix, b"JMT::LeafNode");
    assert_eq!(
        spec.inner_spec.as_ref().unwrap().empty_child,
        b"SPARSE_MERKLE_PLACEHOLDER_HASH__"
    );
    Ok(())
}
