//! JMT pruning support for substores.
//!
//! This module provides functionality to prune historical JMT nodes from a substore,
//! collapsing all state to a single version. This is a non-consensus-breaking operation
//! that reclaims disk space without affecting the current state.
//!
//! **Context**
//! The Jellyfish Merkle Tree (JMT) is a **versioned** sparse merkle tree data structure.
//! Every update to the tree bumps a monotonic version counter. The branching factor for
//! the JMT is 16. Each key insert is hashed into a 256 bits path key.
//! k := H(key) e.g, 0x0123456789abcdef, where each nibble is a conceptual level in the tree.
//! It is conceptual because the tree is compact so depth can be lower. Nodes are addressed by
//! their version and nibble path. The former allows storage and abstraction layers to improve
//! locality of caches and SSD layers.
//!
//! By default, historical state is preserved which allows full nodes to answer historical
//! queries, and generate proofs at any version. In practice, there is no code using this
//! conceptual feature. It just takes up storage.
//!
//! **Amplification**
//! Without pruning, stale nodes are preserved across writes. This allows a tree
//! to answer historical queries about arbitrary versions, at the cost of storage
//! and slower queries on cold rocksdb seeks.
//!
//! The amplification from those extranodes is superlinear in the number of inserts.
//! The quick intuition is that each key update results in a new leaf and a stack of
//! ancestors. Without pruning, those ancestors nodes are never reclaimed. Therefore,
//! each update results in one stack of node per update.
//!
//! How many nodes are in that stack? It varies, but it will be about the height of
//! the compact tree which scales with the logarithm of the # of keys in the tree.
//! that gives an amplification of N * log M, where N is the number of updates, and
//! M the number of keys. We don't account for overlapping ancestor nodes since it
//! asymptotically resolves to a logarithm factor anyway.
use std::sync::Arc;

use anyhow::{ensure, Result};
use jmt::{
    restore::{JellyfishMerkleRestore, StateSnapshotReceiver},
    storage::{Node, NodeBatch, NodeKey, TreeReader, TreeWriter},
    JellyfishMerkleIterator, JellyfishMerkleTree, KeyHash, OwnedValue, RootHash,
};
use rocksdb::DB;

use crate::{
    store::substore::{DbNodeKey, SubstoreConfig, SubstoreSnapshot, VersionedKeyHash},
    Snapshot, Storage,
};

/// Mode for pruning operation.
#[derive(Debug, Clone, Copy, Default)]
pub enum PruneMode {
    /// Use range proofs for each chunk to verify correctness.
    /// Provides verification that the pruned tree matches the original.
    #[default]
    Verified,
}

/// Configuration for the pruning operation.
#[derive(Debug, Clone)]
pub struct PruneConfig {
    /// Number of key-value pairs to process per chunk.
    /// Larger values use more memory but fewer proof operations.
    pub chunk_size: usize,
    /// Whether to verify chunks with range proofs.
    pub mode: PruneMode,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            chunk_size: 100_000,
            mode: PruneMode::Verified,
        }
    }
}

impl PruneConfig {
    /// Validates the configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.chunk_size >= 1, "chunk_size must be at least 1");
        ensure!(
            self.chunk_size <= 10_000_000,
            "chunk_size exceeds maximum of 10 million"
        );
        Ok(())
    }
}

/// Report returned after pruning a substore.
#[derive(Debug, Clone)]
pub struct PruneReport {
    /// The root hash after pruning (should match original).
    pub root: RootHash,
    /// The version that was pruned to.
    pub version: jmt::Version,
    /// Total number of keys processed.
    pub keys_processed: u64,
    /// Number of nodes before pruning.
    pub nodes_before: u64,
    /// Number of nodes after pruning.
    pub nodes_after: u64,
    /// Keys whose JMT leaf commits to a *different* value than the one the read
    /// path returns at `version`.
    ///
    /// A consistent store has none. They appear when something wrote a value row
    /// without rebuilding the leaf above it -- historically, a migration that
    /// committed in place and then built the next block's tree from a snapshot
    /// taken before that in-place write (see `Storage::commit_in_place`).
    ///
    /// The pruner reproduces both halves: the rebuilt tree keeps the leaf the
    /// original tree committed to (so the root hash is unchanged), and the value
    /// row is overwritten with what the read path returns (so `get_raw` on the
    /// pruned store answers exactly what it answered on the original). Without
    /// the override the pruned node would read pre-migration values and diverge
    /// from the rest of the network.
    pub value_overrides: Vec<(KeyHash, Option<OwnedValue>)>,
}

/// A TreeWriter/TreeReader implementation for pruning operations.
/// Writes directly to RocksDB column families.
struct PruningTreeStore {
    db: Arc<DB>,
    config: Arc<SubstoreConfig>,
}

impl PruningTreeStore {
    fn new(db: Arc<DB>, config: Arc<SubstoreConfig>) -> Self {
        Self { db, config }
    }
}

impl TreeWriter for PruningTreeStore {
    fn write_node_batch(&self, node_batch: &NodeBatch) -> Result<()> {
        let cf_jmt = self.config.cf_jmt(&self.db);
        let cf_jmt_values = self.config.cf_jmt_values(&self.db);

        let mut batch = rocksdb::WriteBatch::default();

        // Write nodes
        for (node_key, node) in node_batch.nodes() {
            let key_bytes = DbNodeKey::encode_from_node_key(node_key)?;
            let value_bytes = borsh::to_vec(node)?;
            batch.put_cf(cf_jmt, key_bytes, value_bytes);
        }

        // Write values
        for ((version, key_hash), some_value) in node_batch.values() {
            let key_bytes = VersionedKeyHash::encode_from_keyhash(key_hash, version);
            let value_bytes = borsh::to_vec(some_value)?;
            batch.put_cf(cf_jmt_values, key_bytes, value_bytes);
        }

        self.db.write(batch)?;
        Ok(())
    }
}

impl TreeReader for PruningTreeStore {
    fn get_node_option(&self, node_key: &NodeKey) -> Result<Option<Node>> {
        let cf_jmt = self.config.cf_jmt(&self.db);
        let key_bytes = DbNodeKey::encode_from_node_key(node_key)?;

        match self.db.get_pinned_cf(cf_jmt, &key_bytes)? {
            Some(bytes) => {
                let node: Node = borsh::from_slice(&bytes)?;
                Ok(Some(node))
            }
            None => Ok(None),
        }
    }

    fn get_value_option(
        &self,
        max_version: jmt::Version,
        key_hash: KeyHash,
    ) -> Result<Option<OwnedValue>> {
        let cf_jmt_values = self.config.cf_jmt_values(&self.db);

        // Look for the value at exactly max_version
        let key_bytes = VersionedKeyHash::encode_from_keyhash(&key_hash, &max_version);
        match self.db.get_pinned_cf(cf_jmt_values, &key_bytes)? {
            Some(bytes) => {
                let value: Option<Vec<u8>> = borsh::from_slice(&bytes)?;
                Ok(value)
            }
            None => Ok(None),
        }
    }

    fn get_rightmost_leaf(&self) -> Result<Option<(NodeKey, jmt::storage::LeafNode)>> {
        // Not needed for restore operations
        Ok(None)
    }
}

/// Counts the number of JMT nodes in a substore.
fn count_nodes(db: &Arc<DB>, config: &SubstoreConfig) -> Result<u64> {
    let cf_jmt = config.cf_jmt(db);
    let mut count = 0u64;
    let mut iter = db.raw_iterator_cf(cf_jmt);
    iter.seek_to_first();
    while iter.valid() {
        count += 1;
        iter.next();
    }
    Ok(count)
}

/// Prunes the main store's JMT to a single version.
///
/// This is a convenience wrapper around [`prune_substore`] for the main store (empty prefix).
///
/// # Returns
/// A `PruneReport` with statistics about the pruning operation.
pub fn prune_main_substore(
    old_storage: &Storage,
    old_snapshot: Snapshot,
    new_storage: &Storage,
    version: jmt::Version,
    prune_config: &PruneConfig,
) -> Result<PruneReport> {
    let main_store_config = SubstoreConfig::new("");
    prune_substore(
        old_storage,
        old_snapshot,
        new_storage,
        &main_store_config,
        version,
        prune_config,
    )
}

/// Prunes a substore's JMT to a single version.
///
/// This function reads all key-value pairs from the old database at the specified version,
/// and writes them to a fresh JMT in the new database. After pruning, all nodes will be
/// at the same version.
///
/// # Returns
/// A `PruneReport` with statistics about the pruning operation.
#[tracing::instrument(
    skip(old_storage, old_snapshot, new_storage, config, prune_config),
    fields(
        prefix = %hex::encode(&config.prefix),
        version = version,
        chunk_size = prune_config.chunk_size,
        mode = ?prune_config.mode,
    )
)]
pub fn prune_substore(
    old_storage: &Storage,
    old_snapshot: Snapshot,
    new_storage: &Storage,
    config: &SubstoreConfig,
    version: jmt::Version,
    prune_config: &PruneConfig,
) -> Result<PruneReport> {
    prune_config.validate()?;

    let null_root = KeyHash([0u8; 32]);

    let old_snapshot = old_snapshot.0.snapshot.clone();
    let old_db = &old_storage.db();
    let new_db = &new_storage.db();

    let config = Arc::new(SubstoreConfig::new(&config.prefix));

    let nodes_before = count_nodes(old_db, &config)?;

    let substore_snapshot = SubstoreSnapshot {
        config: Arc::clone(&config),
        rocksdb_snapshot: old_snapshot,
        version,
        db: old_db.clone(),
    };
    let substore_snapshot_arc = Arc::new(substore_snapshot);

    let original_root_hash = substore_snapshot_arc.root_hash()?;

    let tree_store = Arc::new(PruningTreeStore::new(new_db.clone(), Arc::clone(&config)));
    let iter = JellyfishMerkleIterator::new(substore_snapshot_arc.clone(), version, null_root)?;

    let chunk_size = prune_config.chunk_size;
    let mut chunk: Vec<(KeyHash, OwnedValue)> = Vec::with_capacity(chunk_size);
    let mut keys_processed = 0u64;
    let mut value_overrides: Vec<(KeyHash, Option<OwnedValue>)> = Vec::new();

    // Creates a tree for range proof generation
    let old_tree = JellyfishMerkleTree::<_, sha2::Sha256>::new(substore_snapshot_arc.as_ref());

    // A restore instance with verification
    let mut restore = JellyfishMerkleRestore::<sha2::Sha256>::new(
        tree_store.clone(),
        version,
        original_root_hash,
    )?;

    for result in iter {
        let (key_hash, value) = result?;

        // `JellyfishMerkleIterator` reads each value at the version of the *leaf
        // node* it is walking (jmt-0.11 `iterator.rs:296,335`), because that is
        // the value the leaf's `value_hash` -- and therefore the root hash --
        // commits to. Reads take a different route: `SubstoreSnapshot::get_jmt`
        // -> `jmt::tree.get(key, version)` -> `get_value(version, ..)`
        // (jmt-0.11 `tree.rs:1057`), i.e. the newest value row at or below the
        // *queried* version. In a consistent store the two agree. When they do
        // not, the tree must keep the value it commits to, and the read path
        // must keep answering what it answered before pruning.
        let read_path_value = substore_snapshot_arc.get_value_option(version, key_hash)?;
        if read_path_value.as_deref() != Some(value.as_slice()) {
            tracing::warn!(
                key_hash = %hex::encode(key_hash.0),
                leaf_value = %hex::encode(&value),
                read_value = ?read_path_value.as_ref().map(hex::encode),
                "jmt leaf and value column family disagree; preserving both"
            );
            value_overrides.push((key_hash, read_path_value));
        }

        chunk.push((key_hash, value));
        keys_processed += 1;

        if chunk.len() == chunk_size {
            let rightmost_key = chunk.last().expect("chunk is not empty").0;
            let proof = old_tree.get_range_proof(rightmost_key, version)?;
            let chunk_data: Vec<_> = chunk.drain(..).collect();
            restore.add_chunk(chunk_data, proof)?;
            tracing::info!(keys_processed, "processed chunk of keys during pruning");
        }
    }

    // Process remaining entries
    if !chunk.is_empty() {
        let rightmost_key = chunk.last().expect("chunk is not empty").0;
        let proof = old_tree.get_range_proof(rightmost_key, version)?;
        restore.add_chunk(chunk, proof)?;
        tracing::info!(
            keys_processed,
            "processed final chunk of keys during pruning"
        );
    }
    restore.finish()?;

    // Re-apply the read-path values on top of the freshly restored value rows.
    // The nodes are already written, so this changes no hash: it only restores
    // the read semantics of the source database.
    if !value_overrides.is_empty() {
        let cf_jmt_values = config.cf_jmt_values(new_db);
        let mut batch = rocksdb::WriteBatch::default();
        for (key_hash, value) in value_overrides.iter() {
            let key_bytes = VersionedKeyHash::encode_from_keyhash(key_hash, &version);
            batch.put_cf(cf_jmt_values, key_bytes, borsh::to_vec(value)?);
        }
        new_db.write(batch)?;
        tracing::warn!(
            count = value_overrides.len(),
            "restored read-path values for keys whose leaf disagrees with the value column family"
        );
    }

    // Silence unused variable warning for mode field
    let _ = prune_config.mode;
    let nodes_after = count_nodes(new_db, &config)?;

    Ok(PruneReport {
        root: original_root_hash,
        version,
        keys_processed,
        nodes_before,
        nodes_after,
        value_overrides,
    })
}


/// Copy every column family named in `cf_names` that is not in `rebuilt`,
/// preserving key and value bytes exactly.
///
/// The pruner rebuilds only the JMT node and value column families of the
/// substores it prunes. Everything else -- the preimage indices, the
/// nonverifiable data, the config column, and every column family of every
/// substore that is not pruned -- has to be carried over unchanged. Taking the
/// list of column families from the database itself (`rocksdb::DB::list_cf`)
/// rather than from a hardcoded list means a column family added later cannot
/// be silently dropped on the floor.
///
/// Returns `(column family, entries copied)` for each column family copied.
pub fn copy_column_families(
    old_db: &DB,
    new_db: &DB,
    cf_names: &[String],
    rebuilt: &[String],
) -> Result<Vec<(String, u64)>> {
    let mut copied = Vec::new();
    for cf_name in cf_names {
        if rebuilt.iter().any(|r| r == cf_name) {
            continue;
        }
        let old_cf = old_db.cf_handle(cf_name).ok_or_else(|| {
            anyhow::anyhow!("column family '{}' not found in old database", cf_name)
        })?;
        let new_cf = new_db.cf_handle(cf_name).ok_or_else(|| {
            anyhow::anyhow!("column family '{}' not found in new database", cf_name)
        })?;

        let mut count = 0u64;
        let mut batch = rocksdb::WriteBatch::default();
        let mut iter = old_db.raw_iterator_cf(old_cf);
        iter.seek_to_first();
        while iter.valid() {
            if let (Some(key), Some(value)) = (iter.key(), iter.value()) {
                batch.put_cf(new_cf, key, value);
                count += 1;
                if count % 10_000 == 0 {
                    new_db.write(std::mem::take(&mut batch))?;
                }
            }
            iter.next();
        }
        iter.status()?;
        if !batch.is_empty() {
            new_db.write(batch)?;
        }
        tracing::info!(cf_name, count, "copied column family");
        copied.push((cf_name.clone(), count));
    }
    Ok(copied)
}

/// Entry count and a rolling SHA-256 over every key and value of a column family.
///
/// Length-prefixing both key and value keeps the digest injective, so two
/// different column families cannot collide by re-splitting the same bytes.
pub fn fingerprint_column_family(db: &DB, cf_name: &str) -> Result<(u64, [u8; 32])> {
    use sha2::Digest as _;

    let cf = db
        .cf_handle(cf_name)
        .ok_or_else(|| anyhow::anyhow!("column family '{}' not found", cf_name))?;
    let mut hasher = sha2::Sha256::new();
    let mut count = 0u64;
    let mut iter = db.raw_iterator_cf(cf);
    iter.seek_to_first();
    while iter.valid() {
        if let (Some(key), Some(value)) = (iter.key(), iter.value()) {
            hasher.update((key.len() as u64).to_be_bytes());
            hasher.update(key);
            hasher.update((value.len() as u64).to_be_bytes());
            hasher.update(value);
            count += 1;
        }
        iter.next();
    }
    iter.status()?;
    Ok((count, hasher.finalize().into()))
}

/// Check that every column family the pruner did not rebuild is identical in
/// the two databases, and fail with the offending column families otherwise.
///
/// This is the guard that makes the directory swap safe: the JMT half of the
/// output is already covered by the per-chunk range proofs, and this covers
/// everything else.
pub fn verify_column_families(
    old_db: &DB,
    new_db: &DB,
    cf_names: &[String],
    rebuilt: &[String],
) -> Result<()> {
    let mut mismatches = Vec::new();
    for cf_name in cf_names {
        if rebuilt.iter().any(|r| r == cf_name) {
            continue;
        }
        let (old_count, old_hash) = fingerprint_column_family(old_db, cf_name)?;
        let (new_count, new_hash) = fingerprint_column_family(new_db, cf_name)?;
        if old_count != new_count || old_hash != new_hash {
            mismatches.push(format!(
                "{cf_name}: {old_count} entries / {} in the source, {new_count} entries / {} in the pruned database",
                hex::encode(old_hash),
                hex::encode(new_hash),
            ));
        } else {
            tracing::info!(
                cf_name,
                count = old_count,
                hash = %hex::encode(old_hash),
                "column family verified"
            );
        }
    }
    ensure!(
        mismatches.is_empty(),
        "pruned database does not match the source in {} column families: {}",
        mismatches.len(),
        mismatches.join("; "),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_config_validation() {
        // Valid config
        let config = PruneConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: zero chunk size
        let config = PruneConfig {
            chunk_size: 0,
            mode: PruneMode::Verified,
        };
        assert!(config.validate().is_err());

        // Invalid: chunk size too large
        let config = PruneConfig {
            chunk_size: 100_000_000,
            mode: PruneMode::Verified,
        };
        assert!(config.validate().is_err());
    }
}
