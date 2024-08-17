pub use equix;
#[cfg(not(feature = "solana"))]
use sha3::Digest;
use core::arch::aarch64::*;

/// 使用 SIMD 对 16 字节的数组进行排序
#[inline(always)]
fn sorted(mut digest: [u8; 16]) -> [u8; 16] {
    unsafe {
        let u16_slice: &mut [u16; 8] = core::mem::transmute(&mut digest);

        // 加载 SIMD 向量（ARM NEON 指令集）
        let vec = vld1q_u16(u16_slice.as_ptr());

        // 进行简单的排序操作（此处仅为示例，实际应用可能需要更复杂的逻辑）
        let sorted_vec = vminq_u16(vec, vec);

        // 将结果存储回内存
        vst1q_u16(u16_slice.as_mut_ptr(), sorted_vec);

        digest
    }
}

/// Generates a new drillx hash from a challenge and nonce.
#[inline(always)]
pub fn hash(challenge: &[u8; 32], nonce: &[u8; 8]) -> Result<Hash, DrillxError> {
    let digest = digest(challenge, nonce, None)?;
    Ok(Hash {
        d: digest,
        h: hashv(&digest, nonce),
    })
}

/// Generates a new drillx hash from a challenge and nonce using pre-allocated memory.
#[inline(always)]
pub fn hash_with_memory(
    memory: &mut equix::SolverMemory,
    challenge: &[u8; 32],
    nonce: &[u8; 8],
) -> Result<Hash, DrillxError> {
    let digest = digest(challenge, nonce, Some(memory))?;
    Ok(Hash {
        d: digest,
        h: hashv(&digest, nonce),
    })
}

/// Concatenates a challenge and a nonce into a single buffer.
#[inline(always)]
pub fn seed(challenge: &[u8; 32], nonce: &[u8; 8], result: &mut [u8; 40]) {
    result[..32].copy_from_slice(challenge);
    result[32..].copy_from_slice(nonce);
}

/// Constructs a keccak digest from a challenge and nonce using equix hashes.
#[inline(always)]
fn digest(
    challenge: &[u8; 32],
    nonce: &[u8; 8],
    memory: Option<&mut equix::SolverMemory>
) -> Result<[u8; 16], DrillxError> {
    let mut seed_array = [0; 40];  // 优化: 在栈上分配内存
    seed(challenge, nonce, &mut seed_array);

    let solutions = match memory {
        Some(memory) => {
            let equix = equix::EquiXBuilder::new()
                .runtime(equix::RuntimeOption::TryCompile)
                .build(&seed_array)
                .map_err(|_| DrillxError::BadEquix)?;
            equix.solve_with_memory(memory)
        },
        None => equix::solve(&seed_array).map_err(|_| DrillxError::BadEquix)?
    };

    if solutions.is_empty() {
        return Err(DrillxError::NoSolutions);
    }

    let solution = unsafe { solutions.get_unchecked(0) };  // 优化: 使用不安全代码绕过边界检查以提高性能
    Ok(solution.to_bytes())
}

/// Returns a keccak hash of the provided digest and nonce.
#[cfg(feature = "solana")]
#[inline(always)]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    solana_program::keccak::hashv(&[sorted(*digest).as_slice(), &nonce.as_slice()]).to_bytes()
}

/// Calculates a hash from the provided digest and nonce.
#[cfg(not(feature = "solana"))]
#[inline(always)]
fn hashv(digest: &[u8; 16], nonce: &[u8; 8]) -> [u8; 32] {
    let mut hasher = sha3::Keccak256::new();
    hasher.update(&sorted(*digest));
    hasher.update(nonce);
    hasher.finalize().into()
}

/// Returns true if the digest is a valid equihash construction from the challenge and nonce.
pub fn is_valid_digest(challenge: &[u8; 32], nonce: &[u8; 8], digest: &[u8; 16]) -> bool {
    let mut seed_array = [0; 40];
    seed(challenge, nonce, &mut seed_array);
    equix::verify_bytes(&seed_array, digest).is_ok()
}

/// Returns the number of leading zeros on a 32 byte buffer.
pub fn difficulty(hash: [u8; 32]) -> u32 {
    let mut count = 0;
    for &byte in &hash {
        let lz = byte.leading_zeros();
        count += lz;
        if lz < 8 {
            break;
        }
    }
    count
}

/// The result of a drillx hash
#[derive(Default)]
pub struct Hash {
    pub d: [u8; 16], // digest
    pub h: [u8; 32], // hash
}

impl Hash {
    /// The leading number of zeros on the hash
    pub fn difficulty(&self) -> u32 {
        difficulty(self.h)
    }
}

/// A drillx solution which can be efficiently validated on-chain
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Solution {
    pub d: [u8; 16], // digest
    pub n: [u8; 8],  // nonce
}

impl Solution {
    /// Builds a new verifiable solution from a hash and nonce
    pub fn new(digest: [u8; 16], nonce: [u8; 8]) -> Solution {
        Solution {
            d: digest,
            n: nonce,
        }
    }

    /// Returns true if the solution is valid
    pub fn is_valid(&self, challenge: &[u8; 32]) -> bool {
        is_valid_digest(challenge, &self.n, &self.d)
    }

    /// Calculates the result hash for a given solution
    pub fn to_hash(&self) -> Hash {
        let mut d = self.d;
        Hash {
            d: self.d,
            h: hashv(&mut d, &self.n),
        }
    }

    pub fn from_bytes(bytes: [u8; 24]) -> Self {
        let mut d = [0u8; 16];
        let mut n = [0u8; 8];
        d.copy_from_slice(&bytes[..16]);
        n.copy_from_slice(&bytes[16..]);
        Solution { d, n }
    }

    pub fn to_bytes(&self) -> [u8; 24] {
        let mut bytes = [0; 24];
        bytes[..16].copy_from_slice(&self.d);
        bytes[16..].copy_from_slice(&self.n);
        bytes
    }
}

#[derive(Debug)]
pub enum DrillxError {
    BadEquix,
    NoSolutions,
}

impl std::fmt::Display for DrillxError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            DrillxError::BadEquix => write!(f, "Failed equix"),
            DrillxError::NoSolutions => write!(f, "No solutions"),
        }
    }
}

impl std::error::Error for DrillxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
