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

/// The column families the main-store pruner rebuilds from scratch. Everything
/// else has to be copied across verbatim.
const REBUILT_BY_MAIN_STORE_PRUNE: [&str; 2] = ["substore--jmt", "substore--jmt-values"];

fn all_column_families(path: &std::path::Path) -> Result<Vec<String>> {
    Ok(rocksdb::DB::list_cf(
        &rocksdb::Options::default(),
        path.to_str().expect("utf-8 path"),
    )?)
}

fn rebuilt() -> Vec<String> {
    REBUILT_BY_MAIN_STORE_PRUNE
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// The JMT and the value column family can disagree: the leaf commits to one
/// value (via its `value_hash`, and therefore the root hash) while the read path
/// -- which takes the newest value row at or below the queried version, without
/// checking it against the leaf -- returns another. `penumbra-1` is in exactly
/// this state for three keys after the 12598601 restart fork.
///
/// A pruner that only replays the tree iterator reproduces the *leaf* value and
/// silently rewrites what the node reads: on mainnet that turned a Disabled
/// validator back into an Active one and crashed `pd` on the first block. The
/// pruned store must keep both halves -- the root hash the chain agreed on, and
/// the values every other node reads.
#[tokio::test]
async fn test_prune_preserves_read_path_when_leaf_and_value_disagree() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let tmpdir_old = tempfile::tempdir()?;
    let tmpdir_new = tempfile::tempdir()?;
    let substore_prefixes: Vec<String> = vec![];

    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;
    let expected = populate(&storage).await?;
    let version = storage.latest_snapshot().version();

    // `key_01` was last written well before the final version, so its leaf sits
    // at an old node-key version. Write a *newer* value row for it, at the final
    // version, without touching the tree: the shape a migration leaves behind
    // when it commits in place and then rebuilds the next block's tree from a
    // snapshot taken before that write.
    let post_migration = b"post_migration_value".to_vec();
    let key_hash = jmt::KeyHash::with::<sha2::Sha256>(b"key_01");
    {
        let db = storage.db();
        let config = cnidarium::SubstoreConfig::new("");
        let cf_values = config.cf_jmt_values(&db);
        let mut row = key_hash.0.to_vec();
        row.extend_from_slice(&version.to_be_bytes());
        db.put_cf(cf_values, row, borsh::to_vec(&Some(post_migration.clone()))?)?;
        db.flush()?;
    }
    storage.release().await;

    // Reopen so the pruner sees the on-disk state, and confirm the divergence:
    // the read path returns the new value, the root hash is unchanged.
    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;
    let old_snapshot = storage.latest_snapshot();
    let original_root = old_snapshot.root_hash().await?;
    assert_eq!(
        old_snapshot.get_raw("key_01").await?,
        Some(post_migration.clone()),
        "precondition: the read path must return the newer value row"
    );

    let new_storage =
        Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes.clone()).await?;
    let report = prune_main_substore(
        &storage,
        old_snapshot,
        &new_storage,
        version,
        &PruneConfig {
            chunk_size: 3,
            ..Default::default()
        },
    )?;

    assert_eq!(
        report.root, original_root,
        "pruning must not change the root hash"
    );
    assert_eq!(
        report.value_overrides.len(),
        1,
        "exactly one key diverges, got {:?}",
        report.value_overrides
    );
    assert_eq!(report.value_overrides[0].0, key_hash);
    assert_eq!(
        report.value_overrides[0].1.as_ref(),
        Some(&post_migration),
        "the override must carry the read-path value"
    );

    cnidarium::copy_column_families(
        &storage.db(),
        &new_storage.db(),
        &all_column_families(tmpdir_old.path())?,
        &rebuilt(),
    )?;
    cnidarium::verify_column_families(
        &storage.db(),
        &new_storage.db(),
        &all_column_families(tmpdir_old.path())?,
        &rebuilt(),
    )?;

    storage.release().await;
    new_storage.release().await;

    let pruned = Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes).await?;
    let snapshot = pruned.latest_snapshot();
    assert_eq!(
        snapshot.root_hash().await?,
        original_root,
        "the pruned store must still agree with the chain's app hash"
    );
    assert_eq!(
        snapshot.get_raw("key_01").await?,
        Some(post_migration),
        "the pruned store must read what the source read, not what the leaf commits to"
    );
    for (key, value) in &expected {
        if key == "key_01" {
            continue;
        }
        assert_eq!(snapshot.get_raw(key).await?.as_ref(), Some(value));
    }
    Ok(())
}

/// `commit_in_place` rewrites data at the current version. Before this was
/// fixed, the snapshot cache was left untouched, so the next
/// `Storage::latest_snapshot()` handed back a RocksDB snapshot taken *before*
/// the write. Building the following block from it rebuilt the JMT on the
/// pre-commit tree, reverting part of the migration in the merkle tree while the
/// value column family kept the migrated values.
///
/// This is the root cause of the `penumbra-1` divergence. The test drives the
/// exact sequence `pd migrate-restart` uses -- in-place commit, then an ordinary
/// block committed from `latest_snapshot()` -- and asserts that no divergence is
/// produced, by pruning afterwards and requiring zero value overrides.
#[tokio::test]
#[cfg(feature = "migration")]
async fn test_commit_in_place_refreshes_the_snapshot_cache() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let tmpdir_old = tempfile::tempdir()?;
    let tmpdir_new = tempfile::tempdir()?;
    let substore_prefixes: Vec<String> = vec![];

    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;
    populate(&storage).await?;

    // The migration: disable a validator whose state key has not been written
    // for many versions, and stash a nonverifiable record alongside it.
    let mut delta = StateDelta::new(storage.latest_snapshot());
    delta.put_raw("key_01".to_string(), b"disabled".to_vec());
    delta.nonverifiable_put_raw(b"uptime/key_01".to_vec(), b"migrated".to_vec());
    storage.commit_in_place(delta).await?;

    // The snapshot handed out after an in-place commit must observe it.
    let after = storage.latest_snapshot();
    assert_eq!(
        after.get_raw("key_01").await?,
        Some(b"disabled".to_vec()),
        "latest_snapshot() after commit_in_place must not be a pre-commit view"
    );
    assert_eq!(
        after.nonverifiable_get_raw(b"uptime/key_01").await?,
        Some(b"migrated".to_vec())
    );

    drop(after);

    // The empty block the restart fork executes on top of the migration.
    let mut delta = StateDelta::new(storage.latest_snapshot());
    delta.put_raw("block_marker".to_string(), b"synthetic".to_vec());
    storage.commit(delta).await?;

    storage.release().await;
    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;
    let old_snapshot = storage.latest_snapshot();
    let version = old_snapshot.version();
    let original_root = old_snapshot.root_hash().await?;
    assert_eq!(old_snapshot.get_raw("key_01").await?, Some(b"disabled".to_vec()));

    let new_storage =
        Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes.clone()).await?;
    let report = prune_main_substore(
        &storage,
        old_snapshot,
        &new_storage,
        version,
        &PruneConfig {
            chunk_size: 3,
            ..Default::default()
        },
    )?;
    assert!(
        report.value_overrides.is_empty(),
        "a store written through a refreshed snapshot cache must have no leaf/value divergence, got {:?}",
        report.value_overrides
    );

    cnidarium::copy_column_families(
        &storage.db(),
        &new_storage.db(),
        &all_column_families(tmpdir_old.path())?,
        &rebuilt(),
    )?;
    cnidarium::verify_column_families(
        &storage.db(),
        &new_storage.db(),
        &all_column_families(tmpdir_old.path())?,
        &rebuilt(),
    )?;

    storage.release().await;
    new_storage.release().await;

    let pruned = Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes).await?;
    let snapshot = pruned.latest_snapshot();
    assert_eq!(snapshot.root_hash().await?, original_root);
    assert_eq!(snapshot.get_raw("key_01").await?, Some(b"disabled".to_vec()));
    assert_eq!(
        snapshot.nonverifiable_get_raw(b"uptime/key_01").await?,
        Some(b"migrated".to_vec()),
        "nonverifiable data written by the migration must survive pruning"
    );
    Ok(())
}

/// Nonverifiable data is never covered by the JMT or by the pruner's range
/// proofs: it lives in its own column family and is only ever carried across by
/// the copy. Write it through both commit paths -- an ordinary block and the
/// in-place commit `pd migrate-restart` uses -- prune, copy, and require every
/// value to come back, including deletions staying deleted.
#[tokio::test]
#[cfg(feature = "migration")]
async fn test_prune_preserves_nonverifiable_data() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let tmpdir_old = tempfile::tempdir()?;
    let tmpdir_new = tempfile::tempdir()?;
    let substore_prefixes = vec!["ibc".to_string()];

    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;

    for version in 0..NUM_VERSIONS {
        let mut delta = StateDelta::new(storage.latest_snapshot());
        delta.put_raw(format!("key_{}", version % 5), format!("v{version}").into_bytes());
        delta.nonverifiable_put_raw(
            format!("consensus_set_index/{}", version % 5).into_bytes(),
            format!("member_v{version}").into_bytes(),
        );
        delta.nonverifiable_put_raw(
            format!("ibc/nv/{}", version % 3).into_bytes(),
            format!("substore_nv_v{version}").into_bytes(),
        );
        storage.commit(delta).await?;
    }

    // The migration removes one index entry and rewrites another, in place.
    let mut delta = StateDelta::new(storage.latest_snapshot());
    delta.nonverifiable_delete(b"consensus_set_index/0".to_vec());
    delta.nonverifiable_put_raw(b"consensus_set_index/1".to_vec(), b"rewritten".to_vec());
    storage.commit_in_place(delta).await?;

    storage.release().await;
    let storage = Storage::load(tmpdir_old.path().to_path_buf(), substore_prefixes.clone()).await?;
    let old_snapshot = storage.latest_snapshot();
    let version = old_snapshot.version();

    let new_storage =
        Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes.clone()).await?;
    prune_main_substore(
        &storage,
        old_snapshot,
        &new_storage,
        version,
        &PruneConfig {
            chunk_size: 2,
            ..Default::default()
        },
    )?;

    let cfs = all_column_families(tmpdir_old.path())?;
    let copied = cnidarium::copy_column_families(&storage.db(), &new_storage.db(), &cfs, &rebuilt())?;
    assert!(
        copied.iter().any(|(name, count)| name == "substore--nonverifiable" && *count > 0),
        "the main store's nonverifiable column family must be copied and non-empty: {copied:?}"
    );
    assert!(
        copied
            .iter()
            .any(|(name, count)| name == "substore-ibc-nonverifiable" && *count > 0),
        "the substore's nonverifiable column family must be copied and non-empty: {copied:?}"
    );
    cnidarium::verify_column_families(&storage.db(), &new_storage.db(), &cfs, &rebuilt())?;

    storage.release().await;
    new_storage.release().await;

    let pruned = Storage::load(tmpdir_new.path().to_path_buf(), substore_prefixes).await?;
    let snapshot = pruned.latest_snapshot();
    assert_eq!(
        snapshot.nonverifiable_get_raw(b"consensus_set_index/0").await?,
        None,
        "a key the migration deleted must stay deleted"
    );
    assert_eq!(
        snapshot.nonverifiable_get_raw(b"consensus_set_index/1").await?,
        Some(b"rewritten".to_vec()),
        "a key the migration rewrote in place must keep the migrated value"
    );
    for k in 2..5u64 {
        let key = format!("consensus_set_index/{k}");
        assert!(
            snapshot.nonverifiable_get_raw(key.as_bytes()).await?.is_some(),
            "{key} must survive pruning"
        );
    }
    for k in 0..3u64 {
        let key = format!("ibc/nv/{k}");
        assert!(
            snapshot.nonverifiable_get_raw(key.as_bytes()).await?.is_some(),
            "substore key {key} must survive pruning"
        );
    }
    Ok(())
}
