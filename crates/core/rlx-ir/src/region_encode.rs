// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared [`Op::ElementwiseRegion`] metadata encoding for GPU region kernels.

use crate::op::*;
use crate::shape::Shape;

pub const REGION_META_INPUT_WORDS: usize = 16;
pub const REGION_META_CHAIN_WORDS: usize = 128;
pub const REGION_META_TAIL_WORDS: usize = 6;
pub const REGION_META_WORDS: usize =
    REGION_META_INPUT_WORDS + REGION_META_CHAIN_WORDS + REGION_META_TAIL_WORDS;

/// Max batch slices for the single-launch batch region kernel (CUDA/ROCm/Metal/wgpu).
pub const FK_BATCH_SINGLE_KERNEL_MAX: usize = 64;

/// `RLX_FK_BATCH_SINGLE_KERNEL=1` at compile time.
pub fn fk_batch_single_kernel_enabled() -> bool {
    crate::env::flag("RLX_FK_BATCH_SINGLE_KERNEL")
}

/// Whether `BatchElementwiseRegion` should use one batch-region launch.
pub fn fk_batch_use_single_launch(num_batch: usize, prologue: RegionPrologue) -> bool {
    fk_batch_single_kernel_enabled()
        && prologue == RegionPrologue::None
        && num_batch <= FK_BATCH_SINGLE_KERNEL_MAX
}

pub const REGION_PROLOGUE_NONE: u32 = 0;
pub const REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW: u32 = 1;

/// NCHW output dimensions for prologue kernels (`n,c,h,w`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionNchwDims {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

impl RegionNchwDims {
    pub fn from_shape(shape: &Shape) -> Option<Self> {
        if shape.rank() != 4 {
            return None;
        }
        Some(Self {
            n: shape.dim(0).unwrap_static() as u32,
            c: shape.dim(1).unwrap_static() as u32,
            h: shape.dim(2).unwrap_static() as u32,
            w: shape.dim(3).unwrap_static() as u32,
        })
    }

    /// Linear element count for an NCHW tensor.
    pub fn num_elements(self) -> u32 {
        self.n * self.c * self.h * self.w
    }
}

/// 3D launch grid for resize-prologue region kernels (width x height x N*C).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrologueLaunchGrid {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl PrologueLaunchGrid {
    pub fn from_output_shape(shape: &Shape) -> Option<Self> {
        let d = RegionNchwDims::from_shape(shape)?;
        Some(Self {
            width: d.w,
            height: d.h,
            depth: d.n * d.c,
        })
    }
}

/// Encode operand for region chain steps (shared across CUDA / Metal / wgpu).
pub fn encode_chain_operand(op: &ChainOperand) -> u32 {
    match *op {
        ChainOperand::Input(i) => i & 0x7FFF_FFFFu32,
        ChainOperand::Step(i) => 0x8000_0000u32 | (i & 0x7FFF_FFFFu32),
    }
}

pub fn activation_sub(a: Activation) -> u32 {
    match a {
        Activation::Gelu => 0,
        Activation::GeluApprox => 1,
        Activation::Silu => 2,
        Activation::Relu => 3,
        Activation::Sigmoid => 4,
        Activation::Tanh => 5,
        Activation::Exp => 6,
        Activation::Log => 7,
        Activation::Sqrt => 8,
        Activation::Rsqrt => 9,
        Activation::Neg => 10,
        Activation::Abs => 11,
        Activation::Round => 12,
        Activation::Sin => 13,
        Activation::Cos => 14,
        Activation::Tan => 15,
        Activation::Atan => 16,
        Activation::Recip => 17,
        Activation::Floor => 18,
        Activation::Ceil => 19,
        Activation::Sign => 20,
        Activation::Softplus => 21,
        Activation::Elu => 22,
        Activation::Erf => 23,
        Activation::HardSwish => 24,
        Activation::HardSigmoid => 25,
        Activation::Mish => 26,
        Activation::Softsign => 27,
        Activation::LogSigmoid => 28,
    }
}

pub fn binary_sub(b: BinaryOp) -> u32 {
    match b {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
        BinaryOp::Pow => 6,
        BinaryOp::Mod => 7,
        BinaryOp::BitAnd => 8,
        BinaryOp::BitOr => 9,
        BinaryOp::BitXor => 10,
        BinaryOp::Shl => 11,
        BinaryOp::Shr => 12,
        BinaryOp::Atan2 => 13,
    }
}

pub fn compare_sub(c: CmpOp) -> u32 {
    match c {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

/// Pack chain steps into 128 u32 words.
pub fn encode_chain_steps(chain: &[ChainStep]) -> [u32; REGION_META_CHAIN_WORDS] {
    let mut chain_enc = [0u32; REGION_META_CHAIN_WORDS];
    for (k, step) in chain.iter().enumerate() {
        let base = k * 4;
        let (kind, sub, lhs, rhs) = match step {
            ChainStep::Activation(a, src) => {
                (0u32, activation_sub(*a), encode_chain_operand(src), 0u32)
            }
            ChainStep::Cast(_, src) => (1u32, 0, encode_chain_operand(src), 0u32),
            ChainStep::Binary(op, l, r) => (
                2u32,
                binary_sub(*op),
                encode_chain_operand(l),
                encode_chain_operand(r),
            ),
            ChainStep::Compare(op, l, r) => (
                3u32,
                compare_sub(*op),
                encode_chain_operand(l),
                encode_chain_operand(r),
            ),
            ChainStep::Where(c, t, f) => (
                4u32,
                encode_chain_operand(c),
                encode_chain_operand(t),
                encode_chain_operand(f),
            ),
        };
        chain_enc[base] = kind;
        chain_enc[base + 1] = sub;
        chain_enc[base + 2] = lhs;
        chain_enc[base + 3] = rhs;
    }
    chain_enc
}

/// Prologue tag + NCHW shape + external input index for the region output tensor.
pub fn encode_prologue_tail(
    prologue: RegionPrologue,
    out_shape: &Shape,
    prologue_input: u32,
) -> [u32; REGION_META_TAIL_WORDS] {
    let mut tail = [0u32; REGION_META_TAIL_WORDS];
    match prologue {
        RegionPrologue::None => {}
        RegionPrologue::ResizeNearest2x => {
            if let Some(d) = RegionNchwDims::from_shape(out_shape) {
                tail[0] = REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW;
                tail[1] = d.n;
                tail[2] = d.c;
                tail[3] = d.h;
                tail[4] = d.w;
            }
        }
    }
    tail[5] = prologue_input.min(15);
    tail
}

/// Per-slice output shape for [`Op::BatchElementwiseRegion`] (batch axis 0 ? 1).
pub fn batch_region_slice_shape(batch_out: &Shape) -> Shape {
    if batch_out.rank() >= 1 {
        batch_out.clone().with_dim(0, crate::shape::Dim::Static(1))
    } else {
        batch_out.clone()
    }
}

/// Element count of one batch slice in a contiguous batch output tensor.
pub fn batch_region_slice_elems(batch_out: &Shape, num_batch: usize) -> Option<u32> {
    let total = batch_out.num_elements()?;
    let n = num_batch.max(1);
    Some((total / n) as u32)
}

/// f32-linear offset of batch slice `index` within a packed output buffer.
pub fn batch_region_slice_dst_off_f32(base_dst_off: u32, slice_elems: u32, index: usize) -> u32 {
    base_dst_off.saturating_add(index as u32 * slice_elems)
}

/// The `input_modulus[i]` value a region encoding can express for `input` read
/// against a region output of `output`, or the reason it cannot express it.
///
/// Region kernels store **one divisor per operand** and read a broadcast operand as
/// `arena[input_offs[i] + (gid % input_modulus[i])]`. That is the broadcast index the
/// unfused path computes, `(i / rep) % len`, only when `rep == 1` — i.e. when the
/// operand is a trailing slice of the region's output. A mid-axis singleton such as
/// `[1, C, 1, 1]` against `[1, C, 1, L]` needs `(gid / L) % C`, while `gid % C` walks
/// the channel with time: on the snake_beta operand of a `[1, 768, 1, 72]` output that
/// is the wrong element on 53 077 of 55 296 elements, max |Δ| 0.331 on a ±2 signal —
/// a plausible-looking tensor with no error anywhere.
///
/// The encoding carries no per-axis stride ([`REGION_META_INPUT_WORDS`] is one `u32`
/// per input: source bit + 30-bit index, no shape), so it cannot be widened to express
/// a mid-axis singleton without changing every backend's region params. Until it is,
/// this is the admission rule: a region may carry only operands this function answers
/// for, and the unfused `Op::Binary` / `Op::Compare` / `Op::Where` path — which reads
/// through `(i / rep) % len` — computes what the fused one must. The refusal's text
/// says whether the encoding mis-indexes (it would read a different element) or
/// merely has no answer for the shape.
///
/// Returns `0` for an operand that holds the output's element count — both are
/// contiguous row-major, so element `gid` of the operand is element `gid` of the
/// output whatever the axis layout says — and the operand's element count
/// otherwise: `1` for a scalar (which `Op::ElementwiseRegion`'s `scalar_input_mask`
/// also marks) or the trailing extent product for a trailing broadcast.
pub fn region_input_modulus(output: &Shape, input: &Shape) -> Result<u32, String> {
    let Some(out_elems) = output.num_elements() else {
        return Err("the region output's element count is not static".to_string());
    };
    let Some(in_elems) = input.num_elements() else {
        return Err("its element count is not static".to_string());
    };
    if in_elems == 0 {
        return Err("it has no elements".to_string());
    }
    if in_elems == out_elems {
        // Contiguous row-major: element `gid` of the operand is element `gid` of the
        // output, so the divisor is the identity whatever the axis layout says.
        return Ok(0);
    }
    if input.rank() > output.rank() {
        return Err(format!(
            "its rank {} exceeds the region output's {}",
            input.rank(),
            output.rank()
        ));
    }
    // Right-align the operand against the output's trailing axes, the way
    // broadcasting does: its axes cover the output's last `rank` ones, and an axis
    // that is not the same extent there is a broadcast from 1.
    let lead = output.rank() - input.rank();
    let dim = |i: usize| input.dim(i).unwrap_static();
    let out_dim = |i: usize| output.dim(lead + i).unwrap_static();
    for i in 0..input.rank() {
        let (s, t) = (dim(i), out_dim(i));
        if s != 1 && s != t {
            return Err(format!(
                "its axis {i} is {s} against the output's {t}, which is not a broadcast, so \
                 one divisor would read a different element than the unfused path"
            ));
        }
    }
    // The kept axes have to be one contiguous block that reaches the last axis:
    // every axis from the first non-unit one on must be covered one-for-one, or the
    // divisor would repeat the block across an axis the operand does not have (the
    // `[1, C, 1, 1]` case above).
    let Some(first) = (0..input.rank()).find(|&i| dim(i) == out_dim(i) && dim(i) != 1) else {
        return Ok(1); // every operand axis is 1 — a scalar
    };
    if (first..input.rank()).any(|i| dim(i) != out_dim(i)) {
        return Err(
            "it keeps a mid-axis extent, and one divisor can only tile a trailing slice, \
             so it would read a different element than the unfused path"
                .to_string(),
        );
    }
    u32::try_from(in_elems).map_err(|_| format!("it has {in_elems} elements, beyond a u32 divisor"))
}

/// Bitmask of the operand slots `chain` reads: bit `i` is set when a step takes
/// `ChainOperand::Input(i)`.
fn region_input_slots(chain: &[ChainStep]) -> u32 {
    fn mark(slots: &mut u32, op: ChainOperand) {
        if let ChainOperand::Input(i) = op
            && i < 32
        {
            *slots |= 1 << i;
        }
    }
    let mut slots = 0u32;
    for step in chain {
        match step {
            ChainStep::Activation(_, x) | ChainStep::Cast(_, x) => mark(&mut slots, *x),
            ChainStep::Binary(_, l, r) | ChainStep::Compare(_, l, r) => {
                mark(&mut slots, *l);
                mark(&mut slots, *r);
            }
            ChainStep::Where(c, t, f) => {
                mark(&mut slots, *c);
                mark(&mut slots, *t);
                mark(&mut slots, *f);
            }
        }
    }
    slots
}

/// Why the stored `input_modulus` of operand `operand` cannot be honoured, or `None`
/// when it can.
///
/// The message names the operand, the output it is read over and the rule it breaks,
/// so a refusal from a hand-built or pre-fused region says what to change as well as
/// that the region was refused ([`region_input_modulus`] is the rule).
fn region_operand_refusal(
    output: &Shape,
    input: &Shape,
    stored: u32,
    operand: usize,
) -> Option<String> {
    match region_input_modulus(output, input) {
        Err(why) => Some(format!(
            "operand {operand} ({input}) over the {output} region output: {why}"
        )),
        Ok(expected) if !divisor_is_equivalent(expected, stored, output) => Some(format!(
            "operand {operand} ({input}) over the {output} region output admits \
             input_modulus={expected}, but the region stores {stored}: the divisor is the whole \
             broadcast rule, not a free parameter"
        )),
        Ok(_) => None,
    }
}

/// Whether `stored` reads the same index as the canonical divisor `expected` over a
/// region output of that shape: `0` means "read `gid` directly", and so does any divisor
/// at least the output's size. Every other equivalence would have to repeat a shorter
/// block over the same elements, which only `expected` itself does.
fn divisor_is_equivalent(expected: u32, stored: u32, output: &Shape) -> bool {
    if expected == stored {
        return true;
    }
    expected == 0 && output.num_elements().is_none_or(|n| stored as usize >= n)
}

/// The refusals for a region node whose stored `input_modulus` cannot express the
/// operands its chain reads; empty when the region computes what the unfused path
/// computes.
///
/// This is the single admission rule for [`Op::ElementwiseRegion`] and
/// [`Op::BatchElementwiseRegion`]: the IR verifier checks it on every graph, and a
/// backend's lowering applies it again because the direct compile entries run neither
/// the verifier nor `LegalizeBroadcast`. Both therefore judge the same operands against
/// the same output basis.
///
/// `ElementwiseRegion` binds chain slot `i` to `node.inputs[i]`. A
/// `BatchElementwiseRegion` reuses its template's chain and binds each of its inputs as
/// the chain's slot-0 operand for that input's own run of the output's batch axis (the
/// output's dim 0 is the sum of the inputs' dim 0), so each input is judged against its
/// own slice of the output — not against [`batch_region_slice_shape`], which forces dim
/// 0 to 1 and would call a legitimate slice of two or more rows a mid-axis broadcast.
pub fn region_modulus_refusals(graph: &crate::Graph, node: &crate::Node) -> Vec<String> {
    let (chain, declared, scalar_input_mask, input_modulus, prologue, prologue_input, batch) =
        match &node.op {
            Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask,
                input_modulus,
                prologue,
                prologue_input,
            } => (
                chain,
                *num_inputs as usize,
                *scalar_input_mask,
                input_modulus,
                *prologue,
                *prologue_input as usize,
                false,
            ),
            Op::BatchElementwiseRegion {
                chain,
                num_batch_inputs,
                scalar_input_mask,
                input_modulus,
                prologue,
                prologue_input,
            } => (
                chain,
                *num_batch_inputs as usize,
                *scalar_input_mask,
                input_modulus,
                *prologue,
                *prologue_input as usize,
                true,
            ),
            _ => return Vec::new(),
        };
    // Report rather than index past the divisor array. A batch region's count names
    // slices of one operand, not operand slots, so it is not bounded by that array.
    if !batch && declared > REGION_META_INPUT_WORDS {
        return vec![format!(
            "declares {declared} operand(s) and the encoding has {REGION_META_INPUT_WORDS} \
             `input_modulus` slots"
        )];
    }
    let read = region_input_slots(chain);
    let mut refusals = Vec::new();
    for (i, id) in node.inputs.iter().enumerate().take(declared) {
        let slot = if batch { 0 } else { i };
        // A slot the chain never reads, a scalar (read as element 0 through
        // `scalar_input_mask`, never through the divisor), the prologue operand (read
        // through the prologue row) and an unresolvable operand are not judged.
        if read & (1 << slot) == 0
            || scalar_input_mask & (1 << slot) != 0
            || (prologue != RegionPrologue::None && slot == prologue_input)
            || (id.0 as usize) >= graph.len()
        {
            continue;
        }
        let input = graph.shape(*id);
        // `slot` is where the divisor lives (a batch region keeps its one operand's
        // divisor in slot 0); the message names the input that failed, `i`.
        let refusal = match (batch, node.shape.rank(), input.rank()) {
            (true, out_rank, in_rank) if out_rank >= 1 && in_rank >= 1 => {
                let basis = node.shape.clone().with_dim(0, input.dim(0));
                region_operand_refusal(&basis, input, input_modulus[slot], i)
            }
            _ => region_operand_refusal(&node.shape, input, input_modulus[slot], i),
        };
        if let Some(reason) = refusal {
            refusals.push(reason);
        }
    }
    refusals
}

/// Full device metadata buffer for [`Op::ElementwiseRegion`].
pub fn encode_elementwise_region_meta(
    input_offs: &[u32; REGION_META_INPUT_WORDS],
    chain: &[ChainStep],
    prologue: RegionPrologue,
    out_shape: &Shape,
    prologue_input: u32,
) -> [u32; REGION_META_WORDS] {
    let mut meta = [0u32; REGION_META_WORDS];
    meta[..REGION_META_INPUT_WORDS].copy_from_slice(input_offs);
    meta[REGION_META_INPUT_WORDS..REGION_META_INPUT_WORDS + REGION_META_CHAIN_WORDS]
        .copy_from_slice(&encode_chain_steps(chain));
    let tail = encode_prologue_tail(prologue, out_shape, prologue_input);
    let tail_start = REGION_META_INPUT_WORDS + REGION_META_CHAIN_WORDS;
    meta[tail_start..tail_start + REGION_META_TAIL_WORDS].copy_from_slice(&tail);
    meta
}

/// Activation codes the wgsl region kernel has an arm for, in [`activation_sub`]
/// numbering.
///
/// The kernel is `rlx-wgpu`'s `kernels/elementwise_region.wgsl` — one source, three
/// entry points. Its `apply_activation` ends in `return x`, so a code it has no arm
/// for computes the *identity* of its input instead of the activation, with no error
/// and a plausible value. [`crate::op::Activation::region_fusable`] keeps eleven codes
/// out of regions for that reason, but it is shared by every backend and so cannot
/// speak for this one kernel: it leaves the rounding code and the reciprocal in, and
/// the kernel has neither.
///
/// This is the set every region path on the labels that run that kernel consults,
/// through the ambient judgement the compile pipeline installs around the fusion passes
/// (`rlx_fusion::region_kernel`, filled from the device). It is declared here, beside the
/// numbering, rather than in a backend's own tables because the operator-coverage
/// generator parses one fixed file and one fixed constant name per backend. It is held
/// to the kernel's own arms by an in-crate unit test in `rlx-wgpu`
/// (`kernels::tests::declared_region_activation_codes_match_the_kernel_arms`), which
/// parses this constant and that kernel source and requires them to agree.
pub const WGSL_REGION_KERNEL_ACTIVATION_SUBS: &[u32] =
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16];

/// The first activation step in `chain` whose code `subs` has no arm for — the step
/// that would silently be computed as its own input.
pub fn first_unimplemented_activation(subs: &[u32], chain: &[ChainStep]) -> Option<Activation> {
    chain.iter().find_map(|step| match step {
        ChainStep::Activation(a, _) if !subs.contains(&activation_sub(*a)) => Some(*a),
        _ => None,
    })
}

/// Whether `subs` has an arm for every activation step in `chain`.
pub fn activation_subs_cover_chain(subs: &[u32], chain: &[ChainStep]) -> bool {
    first_unimplemented_activation(subs, chain).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn meta_word_count_matches_layout() {
        assert_eq!(REGION_META_WORDS, 150);
    }

    #[test]
    fn batch_slice_elems_and_dst_off() {
        let shape = Shape::new(&[2, 3, 8, 8], DType::F32);
        assert_eq!(batch_region_slice_elems(&shape, 2), Some(192));
        assert_eq!(batch_region_slice_dst_off_f32(100, 192, 1), 100 + 192);
    }

    #[test]
    fn resize_prologue_tail_packed() {
        let shape = Shape::new(&[1, 3, 16, 16], DType::F32);
        let tail = encode_prologue_tail(RegionPrologue::ResizeNearest2x, &shape, 0);
        assert_eq!(tail[0], REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW);
        assert_eq!((tail[1], tail[2], tail[3], tail[4]), (1, 3, 16, 16));
        assert_eq!(tail[5], 0);
        let tail1 = encode_prologue_tail(RegionPrologue::ResizeNearest2x, &shape, 1);
        assert_eq!(tail1[5], 1);
    }

    #[test]
    fn fk_batch_single_kernel_cap() {
        assert_eq!(FK_BATCH_SINGLE_KERNEL_MAX, 64);
    }

    #[test]
    fn fk_batch_use_single_launch_gating() {
        assert!(!fk_batch_use_single_launch(2, RegionPrologue::None));
        assert!(!fk_batch_use_single_launch(
            FK_BATCH_SINGLE_KERNEL_MAX + 1,
            RegionPrologue::None,
        ));
        assert!(!fk_batch_use_single_launch(
            2,
            RegionPrologue::ResizeNearest2x
        ));
    }

    #[test]
    fn the_modulus_rule_admits_equal_counts_and_a_trailing_broadcast() {
        let f32s = |dims: &[usize]| Shape::new(dims, DType::F32);
        // The no-broadcast fast path and a genuine trailing broadcast: `gid` and
        // `gid % L` are what the unfused path reads.
        assert_eq!(region_input_modulus(&f32s(&[4, 8]), &f32s(&[4, 8])), Ok(0));
        assert_eq!(
            region_input_modulus(&f32s(&[1, 3, 8]), &f32s(&[1, 1, 8])),
            Ok(8)
        );
        assert_eq!(region_input_modulus(&f32s(&[1, 3, 8]), &f32s(&[])), Ok(1));
        // A mid-axis singleton needs `(gid / L) % C`, which one divisor cannot say.
        let mid_axis = region_input_modulus(&f32s(&[1, 4, 1, 32]), &f32s(&[1, 4, 1, 1]))
            .expect_err("a mid-axis singleton has no divisor");
        assert!(
            mid_axis.contains("mid-axis") && mid_axis.contains("different element"),
            "the refusal must name the cause and that it would mis-index: {mid_axis}"
        );
        assert_eq!(
            region_input_modulus(&f32s(&[3, 1]), &f32s(&[3])),
            Ok(0),
            "equal counts read `gid`: the unit axis changes the rank, not the element \
             order, so one divisor is the identity here"
        );
        assert_eq!(
            region_input_modulus(&f32s(&[4]), &f32s(&[2, 2])),
            Ok(0),
            "equal counts read `gid` whatever the axis layout: the operand's elements are \
             contiguous, so they line up with the output's element for element"
        );
        // Leading unit axes change the rank, not the element order, so these read
        // `gid`. The higher-rank direction is the rule's, not a graph's: a region's
        // shape is inferred from its chain, so no valid node carries one.
        assert_eq!(region_input_modulus(&f32s(&[4]), &f32s(&[1, 4])), Ok(0));
        assert_eq!(
            region_input_modulus(&f32s(&[3, 4]), &f32s(&[1, 3, 4])),
            Ok(0)
        );
        assert_eq!(region_input_modulus(&f32s(&[]), &f32s(&[1])), Ok(0));
        assert_eq!(
            region_input_modulus(
                &f32s(&[4, 8]),
                &Shape::from_dims(&[crate::shape::Dim::Dynamic(0)], DType::F32)
            ),
            Err("its element count is not static".to_string()),
            "an unresolved operand shape has no divisor"
        );
    }

    /// The canonical divisor `0` means "read `gid` directly", so a hand-built region
    /// that stores the output's own size there reads the same elements — the strict
    /// equality this rule used to demand would refuse a redundant but correct
    /// encoding. Nothing else is equivalent: a shorter divisor repeats a block.
    #[test]
    fn a_divisor_at_least_the_output_size_is_the_identity() {
        let shape = Shape::new(&[2, 3], DType::F32);
        let refusal = |stored: u32| {
            region_operand_refusal(&shape, &shape, stored, 0).map(|r| r.contains("stores"))
        };
        assert_eq!(refusal(0), None);
        assert_eq!(refusal(8), None, "6 elements: `gid % 8 == gid`");
        assert_eq!(refusal(6), None);
        assert_eq!(refusal(3), Some(true), "`gid % 3` repeats over the row");
    }

    /// A batch region binds each input as the chain's slot-0 operand for that input's
    /// own run of the output's batch axis, so its basis is that slice. Judged against
    /// [`batch_region_slice_shape`] (dim 0 forced to 1) a two-row slice looks like a
    /// mid-axis broadcast and the region is refused.
    #[test]
    fn a_batch_slice_of_two_rows_is_admitted() {
        let shape = |dims: &[usize]| Shape::new(dims, DType::F32);
        let mut g = crate::Graph::new("batch");
        let x = g.input("x", shape(&[4, 6]));
        let slices: Vec<crate::NodeId> = [0, 2]
            .into_iter()
            .map(|start| {
                g.add_node(
                    Op::Narrow {
                        axis: 0,
                        start,
                        len: 2,
                    },
                    vec![x],
                    shape(&[2, 6]),
                )
            })
            .collect();
        let region = g.add_node(
            Op::BatchElementwiseRegion {
                chain: vec![ChainStep::Activation(
                    Activation::Relu,
                    ChainOperand::Input(0),
                )],
                num_batch_inputs: 2,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            slices,
            shape(&[4, 6]),
        );
        g.set_outputs(vec![region]);
        assert_eq!(
            region_modulus_refusals(&g, g.node(region)),
            Vec::<String>::new()
        );
        assert!(
            crate::verify::verify(&g).is_empty(),
            "{:?}",
            crate::verify::verify(&g)
        );
    }

    #[test]
    fn a_batch_refusal_names_the_input_that_failed() {
        let shape = |dims: &[usize]| Shape::new(dims, DType::F32);
        let mut g = crate::Graph::new("batch_bad");
        let x = g.input("x", shape(&[4, 6]));
        let wide = g.add_node(
            Op::Narrow {
                axis: 0,
                start: 0,
                len: 2,
            },
            vec![x],
            shape(&[2, 6]),
        );
        let tall = g.add_node(
            Op::Narrow {
                axis: 0,
                start: 2,
                len: 2,
            },
            vec![x],
            shape(&[2, 6]),
        );
        let narrow = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 0,
                len: 3,
            },
            vec![tall],
            shape(&[2, 3]),
        );
        let region = g.add_node(
            Op::BatchElementwiseRegion {
                chain: vec![ChainStep::Activation(
                    Activation::Relu,
                    ChainOperand::Input(0),
                )],
                num_batch_inputs: 2,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![wide, narrow],
            shape(&[4, 6]),
        );
        g.set_outputs(vec![region]);
        // Both inputs read slot 0, so a message keyed on the slot would blame the
        // first slice for the second one's shape.
        let refusals = region_modulus_refusals(&g, g.node(region));
        assert_eq!(refusals.len(), 1, "{refusals:?}");
        assert!(
            refusals[0].contains("operand 1") && refusals[0].contains("axis 1"),
            "the refusal must name the input that failed and its axis: {}",
            refusals[0]
        );
    }

    #[test]
    fn a_region_with_more_slots_than_the_encoding_reports_rather_than_panics() {
        let shape = Shape::new(&[16], DType::F32);
        let mut g = crate::Graph::new("wide");
        let inputs: Vec<crate::NodeId> = (0..17)
            .map(|i| g.input(format!("x{i}"), shape.clone()))
            .collect();
        let region = g.add_node(
            Op::ElementwiseRegion {
                chain: vec![ChainStep::Activation(
                    Activation::Relu,
                    ChainOperand::Input(0),
                )],
                num_inputs: 17,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            inputs,
            shape.clone(),
        );
        g.set_outputs(vec![region]);
        let refusals = region_modulus_refusals(&g, g.node(region));
        assert_eq!(refusals.len(), 1);
        assert!(refusals[0].contains("17"), "{:?}", refusals[0]);
    }

    /// A mid-axis operand is refused whatever the region stores, so a stored
    /// divisor cannot paper over a shape the encoding has no index for: both the
    /// operand's own element count and `0` land on the same refusal, named.
    #[test]
    fn a_stored_divisor_cannot_rescue_a_mid_axis_operand() {
        for stored in [0u32, 64] {
            let mut modulus = [0u32; 16];
            modulus[1] = stored;
            let mut g = crate::Graph::new("mid");
            let x = g.input("x", Shape::new(&[1, 64, 1, 32], DType::F32));
            let bias = g.input("bias", Shape::new(&[1, 64, 1, 1], DType::F32));
            let region = g.add_node(
                Op::ElementwiseRegion {
                    chain: vec![ChainStep::Binary(
                        BinaryOp::Add,
                        ChainOperand::Input(0),
                        ChainOperand::Input(1),
                    )],
                    num_inputs: 2,
                    scalar_input_mask: 0,
                    input_modulus: modulus,
                    prologue: RegionPrologue::None,
                    prologue_input: 0,
                },
                vec![x, bias],
                Shape::new(&[1, 64, 1, 32], DType::F32),
            );
            g.set_outputs(vec![region]);
            let refusals = region_modulus_refusals(&g, g.node(region));
            assert_eq!(refusals.len(), 1, "stored={stored}: {refusals:?}");
            assert!(
                refusals[0].contains("operand 1") && refusals[0].contains("mid-axis"),
                "stored={stored}: {}",
                refusals[0]
            );
        }
    }
}
