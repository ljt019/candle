//! Quantized Qwen3 LLM implementation with quantization support.
//!
//! Based on the Qwen3 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! Key characteristics:
//! - Group-Query Attention (GQA)
//! - RMSNorm for layer normalization, including per-head RMSNorm in attention
//! - Feed-forward network with SwiGLU activation
//! - Support for 2/3/4/8-bit quantization (via QTensor)
//! - Rotary Embeddings (RoPE)
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) (architecture based on official implementations)
//!
use crate::{
    quantized_nn::RmsNorm, // Assuming RmsNorm from the guide's source is used
    utils::repeat_kv,
};
use candle::quantized::{gguf_file, QMatMul, QTensor}; // Import QTensor and QMatMul
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, Embedding}; // Keep Embedding, remove Linear/VarBuilder for quantized parts
use std::io::{Read, Seek};
use std::sync::Arc;

// Re-use the Config struct from the unquantized version
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool, // Note: QMatMul::from_qtensor does NOT support bias. This flag might be ignored in the quantized implementation.
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>, // Sliding window is currently not supported in this impl
    pub max_window_layers: usize,      // Not directly used in the model impl but part of config
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool, // Sliding window is currently not supported in this impl
    pub hidden_act: Activation,
}

// Wrapper for QMatMul to include tracing span
#[derive(Debug, Clone)]
struct QMatMulWrapper {
    inner: QMatMul,
    span: tracing::Span,
}

impl QMatMulWrapper {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = QMatMul::from_qtensor(qtensor)?;
        // Name the span based on the weight name if possible, or a default
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul"); // Generic span
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Rotary Embedding remains the same, operates on Tensor inputs
#[derive(Debug, Clone)]
pub(crate) struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    pub(crate) fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// Rename QuantizedQwen3Attention to AttentionWeights for consistency
#[derive(Debug, Clone)]
pub(crate) struct AttentionWeights {
    // projections
    q_proj: QMatMulWrapper,
    k_proj: QMatMulWrapper,
    v_proj: QMatMulWrapper,
    o_proj: QMatMulWrapper,
    // norms - using the quantized RmsNorm
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize, // Need this for output reshape
    // utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>, // Operates on Tensor
    kv_cache: KvCache,                     // Standard KvCache operates on Tensor
    span_attn: tracing::Span,
}

impl AttentionWeights {
    pub(crate) fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_size: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        // Load QTensor weights and convert to QMatMul wrappers
        let q_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.attn_q.weight"),
            device,
        )?)?;
        let k_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.attn_k.weight"),
            device,
        )?)?;
        let v_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.attn_v.weight"),
            device,
        )?)?;
        let o_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.attn_output.weight"),
            device,
        )?)?;

        // Load QTensor norm weights
        let q_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
            rms_norm_eps,
        )?;
        let k_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
            rms_norm_eps,
        )?;

        // Get max_position from metadata or use a reasonable default
        let max_position_embeddings = ct
            .metadata
            .get("qwen3.context_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;
        let kv_cache = KvCache::new(2, max_position_embeddings);

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
            span_attn,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor, // Input x is a Tensor (output of previous layer or embedding)
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        // 1. Proj - QMatMul::forward returns Tensor
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per‑head RMSNorm - RmsNorm::forward handles Tensor input
        // Flatten B*H and L dimensions to apply norm over the HeadDim (last dim)
        let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> if we transpose later, it becomes (BHL, D), or keep as (B*H, L, D)
        let k_flat = k.flatten(0, 2)?; // Qwen applies norm *before* transpose(1,2) according to some sources,
                                       // but the original qwen3.rs code does transpose(1,2), then flatten, then norm, then reshape.
                                       // Let's follow the original qwen3.rs structure here.
        let q_flat = self.q_norm.forward(&q_flat)?; // Norm applied over the last dimension (HeadDim)
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // 4. RoPE - operates on Tensor
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Accumulate KV cache - KvCache operates on Tensor
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // 6. GQA repeat_kv - operates on Tensor
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // 7. Attention score - standard Tensor operations
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            // Need to handle potential dtype differences and ensure broadcast works
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                // Convert mask to scores dtype if necessary for broadcast_add
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // 8. Output proj - QMatMul::forward returns Tensor
        let reshaped_ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?; // Reshape before applying final projection
        self.o_proj.forward(&reshaped_ctx) // Call forward directly
    }

    pub(crate) fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// Rename MLP struct
#[derive(Debug, Clone)]
pub(crate) struct MlpWeights {
    gate_proj: QMatMulWrapper,
    up_proj: QMatMulWrapper,
    down_proj: QMatMulWrapper,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    pub(crate) fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let gate_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.gate_proj.weight"),
            device,
        )?)?;
        let up_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.up_proj.weight"),
            device,
        )?)?;
        let down_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.down_proj.weight"),
            device,
        )?)?;
        // Activation function is part of the config, need to get it from there.
        // For now, we'll assume SwiGLU is handled by the sequence gate*up.
        // A proper config would pass the activation function type.
        // Based on Qwen3 SwiGLU: silu(gate) * up
        let act_fn = Activation::Silu; // SwiGLU uses SiLU
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn, // Storing as Silu, but the forward impl does gate * up
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?; // Apply SiLU to gate
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?; // SwiGLU combine
        self.down_proj.forward(&gated)
    }
}

// Rename QuantizedDecoderLayer to LayerWeights
#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm, // Using quantized RmsNorm
    ln2: RmsNorm, // Using quantized RmsNorm
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        hidden_size: usize,
        rms_norm_eps: f64,
        rotary: Arc<Qwen3RotaryEmbedding>,
        layer_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        // Update prefix to use the blk.X format shown in the GGUF output
        let prefix = format!("blk.{layer_idx}");

        // RmsNorms take QTensor weights - update paths to match GGUF
        let ln1 = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
            rms_norm_eps,
        )?;
        let ln2 = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
            rms_norm_eps,
        )?;

        // Attention and MLP constructors take ct, reader, prefix
        let self_attn = AttentionWeights::new(
            ct,
            reader,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_size,
            rms_norm_eps,
            rotary,
            &prefix,
            device,
        )?;
        let mlp = MlpWeights::new(ct, reader, &prefix, device)?;

        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        // Norms and attention/mlp operate on Tensor inputs and return Tensor outputs
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// Combine QuantizedModel and QuantizedModelForCausalLM into ModelWeights like in gemma3
#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding, // Embedding is not quantized (weights dequantized)
    layers: Vec<LayerWeights>,
    norm: RmsNorm,           // Using quantized RmsNorm
    lm_head: QMatMulWrapper, // Include lm_head in the ModelWeights
    device: Device,
    dtype: DType, // Model's computation dtype
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Create a simplified approach to examine metadata
        println!("==== GGUF Metadata Analysis ====");

        // Group keys by prefix and print
        let mut by_prefix: std::collections::HashMap<&str, Vec<&String>> =
            std::collections::HashMap::new();

        // First pass: organize keys by prefix
        for key in ct.metadata.keys() {
            let prefix = key.split('.').next().unwrap_or("");
            by_prefix.entry(prefix).or_default().push(key);
        }

        // Print summary of prefixes
        println!("Metadata by prefix:");
        let mut prefixes: Vec<_> = by_prefix.keys().collect();
        prefixes.sort();

        for &prefix in &prefixes {
            let keys = by_prefix.get(prefix).unwrap();
            println!("  {}: {} keys", prefix, keys.len());
        }

        // Print all qwen keys - these are most likely to contain our model config
        println!("\nAll qwen-related keys:");
        if let Some(keys) = by_prefix.get("qwen3") {
            for &key in keys {
                if let Some(value) = ct.metadata.get(key) {
                    println!("  {}: {:?}", key, value);
                }
            }
        } else {
            println!("  No qwen3 keys found, searching for any qwen keys...");
            for (prefix, keys) in &by_prefix {
                if prefix.starts_with("qwen") {
                    for &key in keys {
                        if let Some(value) = ct.metadata.get(key) {
                            println!("  {}: {:?}", key, value);
                        }
                    }
                }
            }
        }

        // Print general metadata
        println!("\nGeneral model information:");
        if let Some(keys) = by_prefix.get("general") {
            for &key in keys {
                if let Some(value) = ct.metadata.get(key) {
                    println!("  {}: {:?}", key, value);
                }
            }
        }

        // Print tensor names to help find embeddings
        println!("\nTensors in the model file:");
        let mut tensor_names: Vec<String> = ct.tensor_infos.keys().cloned().collect();
        tensor_names.sort();

        let possible_embedding_names = [
            "token_embd.weight",
            "model.embed_tokens.weight",
            "embedding.weight",
            "embed_tokens.weight",
            "token_embeddings.weight",
        ];

        let mut embedding_tensor_name = "token_embd.weight"; // Default to common GGUF name

        for name in &tensor_names {
            // Print tensor info
            if let Some(info) = ct.tensor_infos.get(name) {
                let shape_str = format!("{:?}", &info.shape);
                println!("  {} - {}", name, shape_str);

                // Try to identify embedding tensor by checking for known names
                if possible_embedding_names.contains(&name.as_str()) {
                    embedding_tensor_name = name;
                    println!("    ^ Likely embedding tensor");
                }
            }
        }
        println!("===========================");

        println!("Using embedding tensor: {}", embedding_tensor_name);

        // Follow gemma3's approach strictly - use md_get with bail on missing
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Extract required parameters using the exact keys we found in the metadata
        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let intermediate_size = md_get("qwen3.feed_forward_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        // Selection of compute dtype - mimic gemma3
        let dtype = match ct.metadata.get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32, // GGML F32
                Ok(1) => DType::F16, // GGML F16
                _ => DType::F16,     // Default to F16 for quantized
            },
            None => DType::F16, // Default to F16 if missing
        };

        // Load embeddings using the name we found
        let embed_tensor = ct.tensor(reader, embedding_tensor_name, device)?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        // Create rotary embedding
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        println!("\nModel configuration from metadata:");
        println!("  layers: {}", num_layers);
        println!("  hidden_size: {}", hidden_size);
        println!("  intermediate_size: {}", intermediate_size);
        println!("  attention_heads: {}", num_attention_heads);
        println!("  kv_heads: {}", num_kv_heads);
        println!("  head_dim: {}", head_dim);
        println!("  max_position_embeddings: {}", max_position_embeddings);
        println!("  rms_norm_eps: {}", rms_norm_eps);
        println!("  rope_freq_base: {}", rope_freq_base);

        // Load all layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &ct,
                reader,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                hidden_size,
                rms_norm_eps,
                rotary.clone(),
                i,
                device,
            )?);
        }

        // Load final norm
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;

        // Check if lm_head.weight exists, otherwise use embedding weights
        let lm_head_tensor = match ct.tensor_infos.contains_key("lm_head.weight") {
            true => ct.tensor(reader, "lm_head.weight", device)?,
            false => {
                println!("lm_head.weight not found, using embeddings for output projection");
                ct.tensor(reader, embedding_tensor_name, device)?
            }
        };
        let lm_head = QMatMulWrapper::from_qtensor(lm_head_tensor)?;

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            span,
            span_output,
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    // Combine forward methods from both previous structs
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }

        // Apply final norm to hidden states
        let h = self.norm.forward(&h)?;

        // Get the last token's hidden state
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;

        // Project to vocabulary
        self.lm_head.forward(&last_hidden)
    }
}
