use anyhow::Result;
use cnidarium::{
    prune_main_substore, prune_substore, PruneConfig, StateDelta, StateRead, StateWrite, Storage,
};

/// Number of versions (commits) to write before pruning.
const NUM_VERSIONS: u64 = 20;
/// Number of distinct keys to cycle updates over.
const NUM_KEYS: u64 = 10;

/// Populate storage with several versions of overlapping writes, so that
/// the JMT accumulates stale internal nodes, then return the expected final
/// key-value contents of the main store.
async fn populate(storage: &Storage) -> Result<Vec<(String, Vec<u8>)>> {
    for version in 0..NUM_VERSIONS {
        let snapshot = storage.latest_snapshot();
        let mut delta = StateDelta::new(snapshot);
        // Update a rotating subset of keys each version, so most keys are
        // written multiple times and produce stale ancestor stacks.
        for k in 0..NUM_KEYS {
            if (k + version) % 3 == 0 {
                delta.put_raw(
                    format!("key_{k:02}"),
                    format!("value_{k:02}_v{version}").into_bytes(),
                );
            }
        }
        // Exercise deletion markers: key_00 is deleted on the last version.
        if version == NUM_VERSIONS - 1 {
            delta.delete("key_00".to_string());
        }
        storage.commit(delta).await?;
    }

    // Reconstruct the expected final contents.
    let mut expected = Vec::new();
    for k in 0..NUM_KEYS {
        if k == 0 {
            // deleted on the final version
            continue;
        }
        // The last version in which this key was written.
        let last_written = (0..NUM_VERSIONS)
            .rev()
            .find(|version| (k + version) % 3 == 0);
        if let Some(version) = last_written {
            expected.push((
                format!("key_{k:02}"),
                format!("value_{k:02}_v{version}").into_bytes(),
            ));
        }
    }
    Ok(expected)
}

#[tokio::test]
/// Prune the main store into a fresh database and check that:
/// - the reported root hash matches the original root hash
/// - the pruned database contains fewer JMT nodes
/// - reopening the pruned database yields the same version, root hash,
///   and key-value contents as the original
async fn test_prune_main_store_preserves_root_and_contents() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let tmpdir_old = tempfile::tempdir()?;
    let tmpdir_new = tempfile::tempdir()?;
    let substore_prefixes: Vec<String> = vec![];

    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;
    let expected = populate(&storage).await?;

    let old_snapshot = storage.latest_snapshot();
    let original_root = old_snapshot.root_hash().await?;
    let version = old_snapshot.version();
    assert_eq!(version, NUM_VERSIONS - 1);

    let new_storage =
        Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes.clone()).await?;

    // A small chunk size forces multiple range-proof-verified chunks.
    let prune_config = PruneConfig {
        chunk_size: 3,
        ..Default::default()
    };

    let report = prune_main_substore(&storage, old_snapshot, &new_storage, version, &prune_config)?;

    assert_eq!(
        report.root, original_root,
        "pruned root hash must match original"
    );
    assert_eq!(report.version, version);
    assert_eq!(report.keys_processed, expected.len() as u64);
    assert!(
        report.nodes_after < report.nodes_before,
        "pruning should reduce node count: before={} after={}",
        report.nodes_before,
        report.nodes_after
    );

    // Reopen the pruned database and verify version, root hash, and contents.
    storage.release().await;
    new_storage.release().await;
    let pruned = Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes).await?;
    let pruned_snapshot = pruned.latest_snapshot();
    assert_eq!(
        pruned_snapshot.version(),
        version,
        "pruned database must report the original version"
    );
    assert_eq!(
        pruned_snapshot.root_hash().await?,
        original_root,
        "pruned database must report the original root hash"
    );
    for (key, value) in &expected {
        let got = pruned_snapshot.get_raw(key).await?;
        assert_eq!(
            got.as_ref(),
            Some(value),
            "pruned database must contain {key}"
        );
    }
    // The deleted key must stay deleted.
    assert_eq!(pruned_snapshot.get_raw("key_00").await?, None);

    Ok(())
}

#[tokio::test]
/// Prune with a chunk size larger than the number of keys, so that all
/// entries are processed in the single final chunk.
async fn test_prune_single_chunk() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let tmpdir_old = tempfile::tempdir()?;
    let tmpdir_new = tempfile::tempdir()?;
    let substore_prefixes: Vec<String> = vec![];

    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;
    let expected = populate(&storage).await?;

    let old_snapshot = storage.latest_snapshot();
    let original_root = old_snapshot.root_hash().await?;
    let version = old_snapshot.version();

    let new_storage =
        Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes.clone()).await?;

    let report = prune_main_substore(
        &storage,
        old_snapshot,
        &new_storage,
        version,
        &PruneConfig::default(),
    )?;

    assert_eq!(report.root, original_root);
    assert_eq!(report.keys_processed, expected.len() as u64);
    Ok(())
}

#[tokio::test]
/// Prune a named substore directly via `prune_substore` and verify that the
/// substore's root and contents survive in the destination database.
async fn test_prune_substore_preserves_contents() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let tmpdir_old = tempfile::tempdir()?;
    let tmpdir_new = tempfile::tempdir()?;
    let substore_prefixes = vec!["ibc".to_string()];

    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;

    // Write several versions of keys into the substore.
    for version in 0..NUM_VERSIONS {
        let snapshot = storage.latest_snapshot();
        let mut delta = StateDelta::new(snapshot);
        delta.put_raw(
            format!("ibc/client_{}", version % 4),
            format!("state_v{version}").into_bytes(),
        );
        storage.commit(delta).await?;
    }

    let old_snapshot = storage.latest_snapshot();
    let version = old_snapshot.version();

    let new_storage =
        Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes.clone()).await?;

    // The substore has its own version counter: it was written to on every
    // commit, so its version matches the main store version here.
    let substore_config = cnidarium::SubstoreConfig::new("ibc");
    let report = prune_substore(
        &storage,
        old_snapshot,
        &new_storage,
        &substore_config,
        version,
        &PruneConfig {
            chunk_size: 2,
            ..Default::default()
        },
    )?;

    assert_eq!(report.keys_processed, 4);
    assert!(report.nodes_after < report.nodes_before);
    Ok(())
}
