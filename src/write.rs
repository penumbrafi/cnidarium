use crate::StateRead;
use std::{any::Any, collections::BTreeMap};
use tendermint::abci;

/// Write access to chain state.
pub trait StateWrite: StateRead + Send + Sync {
    /// Puts raw bytes into the verifiable key-value store with the given key.
    fn put_raw(&mut self, key: String, value: Vec<u8>);

    /// Delete a key from the verifiable key-value store.
    fn delete(&mut self, key: String);

    /// Puts raw bytes into the verifiable key-value store under a key that
    /// need not be valid UTF-8.
    ///
    /// The verifiable store hashes key bytes, so this is the same store as
    /// [`put_raw`](Self::put_raw): a key that *is* valid UTF-8 is exactly
    /// equivalent to `put_raw(String::from_utf8(key), value)`, and reads
    /// through either API agree. Keys that are not valid UTF-8 are invisible
    /// to the string-typed `prefix_raw` / `prefix_keys` streams; they are
    /// intended for protocol-fixed paths such as IBC v2 commitment keys,
    /// which are only ever read by exact key or proved by `get_with_proof`.
    fn put_raw_bytes(&mut self, key: Vec<u8>, value: Vec<u8>);

    /// Delete a key from the verifiable key-value store; see
    /// [`put_raw_bytes`](Self::put_raw_bytes).
    fn delete_bytes(&mut self, key: Vec<u8>);

    /// Puts raw bytes into the non-verifiable key-value store with the given key.
    fn nonverifiable_put_raw(&mut self, key: Vec<u8>, value: Vec<u8>);

    /// Delete a key from non-verifiable key-value storage.
    fn nonverifiable_delete(&mut self, key: Vec<u8>);

    /// Puts an object into the ephemeral object store with the given key.
    ///
    /// # Panics
    ///
    /// If the object is already present in the store, but its type is not the same as the type of
    /// `value`.
    fn object_put<T: Clone + Any + Send + Sync>(&mut self, key: &'static str, value: T);

    /// Deletes a key from the ephemeral object store.
    fn object_delete(&mut self, key: &'static str);

    /// Merge a set of object changes into this `StateWrite`.
    ///
    /// Unlike `object_put`, this avoids re-boxing values and messing up the downcasting.
    fn object_merge(&mut self, objects: BTreeMap<&'static str, Option<Box<dyn Any + Send + Sync>>>);

    /// Record that an ABCI event occurred while building up this set of state changes.
    fn record(&mut self, event: abci::Event);
}

impl<'a, S: StateWrite + Send + Sync> StateWrite for &'a mut S {
    fn put_raw(&mut self, key: String, value: jmt::OwnedValue) {
        (**self).put_raw(key, value)
    }

    fn delete(&mut self, key: String) {
        (**self).delete(key)
    }

    fn put_raw_bytes(&mut self, key: Vec<u8>, value: Vec<u8>) {
        (**self).put_raw_bytes(key, value)
    }

    fn delete_bytes(&mut self, key: Vec<u8>) {
        (**self).delete_bytes(key)
    }

    fn nonverifiable_delete(&mut self, key: Vec<u8>) {
        (**self).nonverifiable_delete(key)
    }

    fn nonverifiable_put_raw(&mut self, key: Vec<u8>, value: Vec<u8>) {
        (**self).nonverifiable_put_raw(key, value)
    }

    fn object_put<T: Clone + Any + Send + Sync>(&mut self, key: &'static str, value: T) {
        (**self).object_put(key, value)
    }

    fn object_delete(&mut self, key: &'static str) {
        (**self).object_delete(key)
    }

    fn object_merge(
        &mut self,
        objects: BTreeMap<&'static str, Option<Box<dyn Any + Send + Sync>>>,
    ) {
        (**self).object_merge(objects)
    }

    fn record(&mut self, event: abci::Event) {
        (**self).record(event)
    }
}
