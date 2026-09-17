//! Byte-keyed verifiable state: IBC v2 commitment paths are
//! `clientId || 0x01 || u64_be(sequence)` and stop being valid UTF-8 at
//! sequence 128. They must be stored under exactly those bytes so that a
//! counterparty recomputing the path can verify membership against our root.

use cnidarium::{StateDelta, StateRead, StateWrite, Storage};
use futures::StreamExt;
use ibc_types::core::commitment::MerkleProof;

fn packet_commitment_key(client_id: &str, seq: u64) -> Vec<u8> {
    let mut k = client_id.as_bytes().to_vec();
    k.push(0x01);
    k.extend_from_slice(&seq.to_be_bytes());
    k
}

fn full_key(substore_key: &[u8]) -> Vec<u8> {
    [b"ibc-data/".as_slice(), substore_key].concat()
}

#[tokio::test]
async fn non_utf8_keys_round_trip_prove_and_do_not_break_string_streams() -> anyhow::Result<()> {
    let tmpdir = tempfile::tempdir()?;
    let storage = Storage::load(tmpdir.keep(), vec!["ibc-data".to_string()]).await?;
    let specs = vec![cnidarium::ics23_spec(), cnidarium::ics23_spec()];

    // Sequence 200: the big-endian bytes contain 0xC8, which is not UTF-8.
    let sub_key = packet_commitment_key("07-tendermint-0", 200);
    assert!(std::str::from_utf8(&sub_key).is_err());
    let key = full_key(&sub_key);
    let value = [0xABu8; 32].to_vec();

    let mut delta = StateDelta::new(storage.latest_snapshot());
    delta.put_raw_bytes(key.clone(), value.clone());
    // A string key beside it, so the string prefix streams have something to yield.
    delta.put_raw("ibc-data/clients/07-tendermint-0/clientState".to_string(), b"cs".to_vec());
    // A byte key that happens to be valid UTF-8 must be readable through both APIs.
    delta.put_raw_bytes(b"ibc-data/v2/counterparty/07-tendermint-0".to_vec(), b"cp".to_vec());

    // Reads through the delta before commit (cache hit paths).
    assert_eq!(delta.get_raw_bytes(&key).await?.as_deref(), Some(value.as_slice()));
    assert_eq!(
        delta.get_raw("ibc-data/v2/counterparty/07-tendermint-0").await?.as_deref(),
        Some(b"cp".as_slice())
    );
    // Fork: the byte write must be visible through a layer, not only the leaf.
    let child = delta.fork();
    assert_eq!(child.get_raw_bytes(&key).await?.as_deref(), Some(value.as_slice()));
    drop(child);
    storage.commit(delta).await?;

    let snapshot = storage.latest_snapshot();
    assert_eq!(snapshot.get_raw_bytes(&key).await?.as_deref(), Some(value.as_slice()));
    assert_eq!(
        snapshot.get_raw_bytes(b"ibc-data/v2/counterparty/07-tendermint-0").await?.as_deref(),
        Some(b"cp".as_slice())
    );

    // String prefix streams over the same substore must not panic and must
    // skip the non-UTF-8 key.
    let keys: Vec<String> = snapshot
        .prefix_keys("ibc-data/")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<anyhow::Result<_>>()?;
    assert_eq!(
        keys,
        vec![
            "ibc-data/clients/07-tendermint-0/clientState".to_string(),
            "ibc-data/v2/counterparty/07-tendermint-0".to_string()
        ]
    );
    let kvs: Vec<(String, Vec<u8>)> = snapshot
        .prefix_raw("ibc-data/")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<anyhow::Result<_>>()?;
    assert_eq!(kvs.len(), 2);

    // Membership proof under the raw path bytes, as a counterparty computes it.
    // `ibc_types::MerklePath` is string-typed, so verify through ics23 on
    // the raw path bytes, exactly as ibc-go and the Eureka membership
    // program do.
    let root = snapshot.root_hash().await?.0.to_vec();
    let (got, proof) = snapshot.get_with_proof(key.clone()).await?;
    assert_eq!(got.as_deref(), Some(value.as_slice()));
    verify_membership_raw(&proof, &specs, &root, &[b"ibc-data".to_vec(), sub_key.clone()], &value)?;

    // Non-membership for the next sequence (timeout proof shape).
    let absent_sub = packet_commitment_key("07-tendermint-0", 202);
    let (got, nex) = snapshot.get_with_proof(full_key(&absent_sub)).await?;
    assert!(got.is_none());
    verify_non_membership_raw(&nex, &specs, &root, &[b"ibc-data".to_vec(), absent_sub])?;

    // Delete by bytes, then the key must be provably absent.
    let mut delta = StateDelta::new(storage.latest_snapshot());
    delta.delete_bytes(key.clone());
    storage.commit(delta).await?;
    let snapshot = storage.latest_snapshot();
    assert!(snapshot.get_raw_bytes(&key).await?.is_none());
    let root = snapshot.root_hash().await?.0.to_vec();
    let (got, nex) = snapshot.get_with_proof(key.clone()).await?;
    assert!(got.is_none());
    verify_non_membership_raw(&nex, &specs, &root, &[b"ibc-data".to_vec(), sub_key])?;
    Ok(())
}

/// ICS23 chained verification over raw byte paths (what `ibc-go` and the
/// Eureka membership program do), without ibc-types' string-typed `MerklePath`.
fn verify_membership_raw(
    proof: &MerkleProof,
    specs: &[ics23::ProofSpec],
    root: &[u8],
    keys: &[Vec<u8>],
    value: &[u8],
) -> anyhow::Result<()> {
    anyhow::ensure!(proof.proofs.len() == specs.len() && keys.len() == specs.len());
    let mut subroot: Vec<u8> = Vec::new();
    let mut val = value.to_vec();
    // Innermost proof first: keys[last] proves under the substore root.
    for ((p, spec), key) in proof.proofs.iter().zip(specs).zip(keys.iter().rev()) {
        let ex = match &p.proof {
            Some(ics23::commitment_proof::Proof::Exist(ex)) => ex,
            _ => anyhow::bail!("expected existence proof"),
        };
        subroot = ics23::calculate_existence_root::<ics23::HostFunctionsManager>(ex)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        anyhow::ensure!(
            ics23::verify_membership::<ics23::HostFunctionsManager>(p, spec, &subroot, key, &val),
            "membership failed at key {:?}",
            key
        );
        val = subroot.clone();
    }
    anyhow::ensure!(subroot == root, "root mismatch");
    Ok(())
}

fn verify_non_membership_raw(
    proof: &MerkleProof,
    specs: &[ics23::ProofSpec],
    root: &[u8],
    keys: &[Vec<u8>],
) -> anyhow::Result<()> {
    anyhow::ensure!(proof.proofs.len() == 2 && specs.len() == 2 && keys.len() == 2);
    // Layer 0: non-existence of the leaf key under the substore root, whose
    // value is recovered from one of the neighbouring existence proofs.
    let nex = match &proof.proofs[0].proof {
        Some(ics23::commitment_proof::Proof::Nonexist(nex)) => nex,
        _ => anyhow::bail!("expected non-existence proof at layer 0"),
    };
    let neighbour = nex
        .left
        .as_ref()
        .or(nex.right.as_ref())
        .ok_or_else(|| anyhow::anyhow!("non-existence proof has no neighbour"))?;
    let subroot = ics23::calculate_existence_root::<ics23::HostFunctionsManager>(neighbour)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    anyhow::ensure!(
        ics23::verify_non_membership::<ics23::HostFunctionsManager>(
            &proof.proofs[0],
            &specs[0],
            &subroot,
            &keys[1]
        ),
        "non-membership failed"
    );
    // Layer 1: the substore root exists under the main root.
    anyhow::ensure!(
        ics23::verify_membership::<ics23::HostFunctionsManager>(
            &proof.proofs[1],
            &specs[1],
            &root.to_vec(),
            &keys[0],
            &subroot
        ),
        "substore root membership failed"
    );
    Ok(())
}
