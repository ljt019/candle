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
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
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

// Quantized MLP using QMatMul
#[derive(Debug, Clone)]
pub(crate) struct QuantizedQwen3MLP {
    gate_proj: QMatMulWrapper,
    up_proj: QMatMulWrapper,
    down_proj: QMatMulWrapper,
    act_fn: Activation,
    span: tracing::Span,
}

impl QuantizedQwen3MLP {
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

impl Module for QuantizedQwen3MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?; // Apply SiLU to gate
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?; // SwiGLU combine
        self.down_proj.forward(&gated)
    }
}

// Quantized Attention using QMatMul and quantized RmsNorm
#[derive(Debug, Clone)]
pub(crate) struct QuantizedQwen3Attention {
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

impl QuantizedQwen3Attention {
    pub(crate) fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        cfg: &Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            // Based on the original code's behavior
            candle::bail!("sliding window is not suppored in this quantized implementation");
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        // Load QTensor weights and create QMatMul wrappers
        let q_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.q_proj.weight"),
            device,
        )?)?;
        let k_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.k_proj.weight"),
            device,
        )?)?;
        let v_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.v_proj.weight"),
            device,
        )?)?;
        let o_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.o_proj.weight"),
            device,
        )?)?;

        // Load QTensor norm weights and create RmsNorm instances
        let q_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.q_norm.weight"), device)?,
            cfg.rms_norm_eps,
        )?;
        let k_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.k_norm.weight"), device)?,
            cfg.rms_norm_eps,
        )?;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * cfg.num_attention_heads;

        let kv_cache = KvCache::new(2, cfg.max_position_embeddings);

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

// Quantized Decoder Layer
#[derive(Debug, Clone)]
struct QuantizedDecoderLayer {
    self_attn: QuantizedQwen3Attention,
    mlp: QuantizedQwen3MLP,
    ln1: RmsNorm, // Using quantized RmsNorm
    ln2: RmsNorm, // Using quantized RmsNorm
}

impl QuantizedDecoderLayer {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        cfg: &Config,
        rotary: Arc<Qwen3RotaryEmbedding>,
        layer_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");

        // RmsNorms take QTensor weights
        let ln1 = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.input_layernorm.weight"), device)?,
            cfg.rms_norm_eps,
        )?;
        let ln2 = RmsNorm::from_qtensor(
            ct.tensor(
                reader,
                &format!("{prefix}.post_attention_layernorm.weight"),
                device,
            )?,
            cfg.rms_norm_eps,
        )?;

        // Attention and MLP constructors take ct, reader, prefix
        let self_attn = QuantizedQwen3Attention::new(
            ct,
            reader,
            cfg,
            rotary,
            &format!("{prefix}.self_attn"),
            device,
        )?;
        let mlp = QuantizedQwen3MLP::new(ct, reader, &format!("{prefix}.mlp"), device)?;

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

// Quantized Model
#[derive(Debug, Clone)]
pub struct QuantizedModel {
    embed_tokens: Embedding, // Embedding is not quantized (weights dequantized)
    layers: Vec<QuantizedDecoderLayer>,
    norm: RmsNorm, // Using quantized RmsNorm
    device: Device,
    dtype: DType, // Model's computation dtype
    span: tracing::Span,
}

impl QuantizedModel {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        cfg: &Config,
        device: &Device,
    ) -> Result<Self> {
        // Pick computation dtype: F32 if metadata code 0, else F16
        let dtype = ct
            .metadata
            .get("general.dtype")
            .and_then(|v| v.to_u32().ok())
            .map(|code| match code {
                0 => DType::F32, // GGML F32
                1 => DType::F16, // GGML F16
                _ => DType::F16, // other (quantized), use F16
            })
            .unwrap_or(DType::F32);

        // Load embedding weights - dequantize to use in standard Embedding
        let embed_tensor = ct.tensor(reader, "model.embed_tokens.weight", device)?;
        let embed_tensor = embed_tensor.dequantize(device)?;
        let embed_tokens = Embedding::new(embed_tensor, cfg.hidden_size);

        // Create rotary embedding
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(dtype, cfg, device)?);

        // Load decoder layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedDecoderLayer::new(
                &ct,
                reader,
                cfg,
                rotary.clone(),
                i,
                device,
            )?);
        }

        // Load final norm
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "model.norm.weight", device)?,
            cfg.rms_norm_eps,
        )?;

        let span = tracing::span!(tracing::Level::TRACE, "model");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: device.clone(),
            dtype,
            span,
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
        self.norm.forward(&h)
    }
}

// Quantized Model for Causal Language Modeling
#[derive(Debug, Clone)]
pub struct QuantizedModelForCausalLM {
    base: QuantizedModel,
    lm_head: QMatMulWrapper,
    span_output: tracing::Span,
}

impl QuantizedModelForCausalLM {
    // Use Read + Seek bounds directly
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        cfg: &Config,
        device: &Device,
    ) -> Result<Self> {
        // Load lm_head tensor *before* constructing base model
        let lm_head_tensor = if cfg.tie_word_embeddings {
            ct.tensor(reader, "model.embed_tokens.weight", device)?
        } else {
            ct.tensor(reader, "lm_head.weight", device)?
        };
        let lm_head = QMatMulWrapper::from_qtensor(lm_head_tensor)?;

        // Now construct base model, moving ct
        let base = QuantizedModel::from_gguf(ct, reader, cfg, device)?;

        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            base,
            lm_head,
            span_output,
        })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let hidden_states = self.base.forward(input, offset)?;

        let _enter = self.span_output.enter();
        // Get the last token's hidden state
        let last_hidden = hidden_states.narrow(1, l - 1, 1)?;
        // Project to vocabulary
        self.lm_head.forward(&last_hidden)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}
