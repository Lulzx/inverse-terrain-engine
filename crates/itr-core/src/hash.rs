//! BLAKE3 content hashing (pure-Rust build, spec §8.4).

pub fn hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

pub struct Hasher(blake3::Hasher);

impl Default for Hasher {
    fn default() -> Self {
        Self(blake3::Hasher::new())
    }
}

impl Hasher {
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.0.update(&(s.len() as u64).to_le_bytes());
        self.0.update(s.as_bytes());
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.update(&v.to_le_bytes());
        self
    }
    pub fn i64s(&mut self, v: &[i64]) -> &mut Self {
        self.u64(v.len() as u64);
        for x in v {
            self.0.update(&x.to_le_bytes());
        }
        self
    }
    pub fn f64s(&mut self, v: &[f64]) -> &mut Self {
        self.u64(v.len() as u64);
        for x in v {
            self.0.update(&x.to_bits().to_le_bytes());
        }
        self
    }
    pub fn hex(&self) -> String {
        self.0.finalize().to_hex().to_string()
    }
}
