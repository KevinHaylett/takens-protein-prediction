# Architecture: The Takens-Based Transformer for Protein Structure Prediction

*Kevin R. Haylett, PhD — Manchester, UK*

---

## The Problem Reframed

Most approaches to protein structure prediction treat the problem as a **sequence-to-structure mapping**: given a string of amino acid identifiers, learn a function that outputs 3D coordinates. The model is trained to minimise the distance between predicted and known structures, and success is measured by how well the learned mapping generalises to unseen sequences.

This is a reasonable engineering framing. But it is not the only framing, and it may not be the most natural one.

Protein folding is a **physical process that unfolds in time**. A newly synthesised polypeptide chain explores conformational space — it is a dynamical system, not a lookup table. The folded structure is not a static destination that the sequence maps to; it is an **attractor** — a stable configuration that the physical dynamics converge to under the constraints imposed by the amino acid sequence, solvent, temperature, and molecular forces. The sequence is the control parameter that shapes the energy landscape; the folded structure is the geometry that the trajectory settles into.

This distinction matters because it suggests a different mathematical tool.

---

## Takens' Delay Embedding Theorem

In 1981, Floris Takens proved a fundamental result about nonlinear dynamical systems. Consider a system evolving on some hidden manifold — a protein folding through conformational space, for example. We cannot observe the full internal state, but we can observe a single time-varying measurement: the amino acid sequence as it is processed position by position.

Takens proved that if we construct delay coordinates from this single observable:

```
x(t) = [e(t), e(t-τ₁), e(t-τ₂), ..., e(t-τₘ)]
```

then for sufficiently large embedding dimension m, the resulting trajectory is **diffeomorphic to the original system's attractor**. It preserves the qualitative geometry — the topology, the flow structure, the shape of the manifold — from a single observable time series.

This is the foundation of the architecture. The amino acid sequence is the observable. The 3D fold is the attractor. The model's job is to reconstruct the attractor geometry from delay coordinates of the sequence embedding — exactly as Takens' theorem prescribes.

---

## Why Exponential Delays

Protein structure exhibits organisation at multiple spatial scales simultaneously:

- **Local backbone geometry** (3–5 residues): peptide bond angles, immediate steric constraints
- **Secondary structure** (5–30 residues): alpha helices, beta strands, loop regions
- **Tertiary topology** (30–300+ residues): domain packing, long-range contacts, overall fold

A fixed-window delay embedding would treat all these scales equally, which wastes representational capacity. Exponentially-spaced delays allocate capacity efficiently:

```
delays = [1, 2, 4, 8, 16, 32, 64, 128]
```

Dense sampling at short delays captures local backbone dynamics. Sparse sampling at long delays captures domain-level topology. The full multi-scale structure of the protein is represented in a single fixed-size vector, regardless of sequence length.

This is the same exponential delay schedule used in the language modelling experiments, where it captures phoneme-level, sentence-level, and discourse-level structure simultaneously. The mathematical justification is the same in both domains: the system has structure at multiple timescales, and logarithmic spacing respects that.

---

## The Architecture in Detail

### Residue Encoding

Each amino acid is mapped to a learned embedding vector of dimension `embed_dim`. The 20 standard amino acids are represented directly; non-standard residues are handled via a small extension vocabulary.

Unlike language models, there are no positional encodings. In the Takens framework, temporal position is encoded implicitly in the delay structure — the relative positions of delayed embeddings carry the sequential information. An explicit positional signal would be redundant and potentially distorting.

### Exponential Takens Embedding

For each residue position `t`, a delay-coordinate vector is constructed:

```
z(t) = [e(t), e(t-1), e(t-2), e(t-4), e(t-8), e(t-16), e(t-32), e(t-128)]
```

where `e(t-τ)` is the embedding of the residue at delay `τ` from the current position. Positions before the sequence start are zero-padded.

This produces a vector of dimension `(num_delays + 1) × embed_dim`. For 8 delays and embed_dim=128, this is a 1152-dimensional vector per position.

The implementation uses a circular buffer of size `2^(k+1)` where `k` is the maximum delay power. This means memory usage is **O(1) regardless of sequence length** — the buffer does not grow as the sequence is processed.

### Adaptive Manifold Projection

The raw delay vector is high-dimensional and sparse (many entries are zero-padded for early positions). A learned linear projection compresses it onto a lower-dimensional manifold:

```
h(t) = LayerNorm(W_p · z(t) + b_p)
```

where `W_p ∈ R^{d_out × (M+1)d}` is the projection matrix.

This matrix is the geometric heart of the model. It learns which combinations of which timescales are most informative for predicting structure. After training, the rows of `W_p` can be inspected to understand what temporal patterns the model considers important — an interpretability property unavailable in attention-based models, where context is aggregated implicitly across all tokens.

The bias term `b_p` provides reference positioning of the manifold in projection space. It does not introduce temporal structure.

### Temporal Mixing Layers

A stack of feedforward residual layers further transforms the manifold state:

```
x ← x + FFN(LayerNorm(x))
```

These layers allow the model to learn non-linear interactions across the manifold state at each position. They do not introduce any cross-position comparisons — there is no attention, no token-to-token interaction. Each position's manifold state evolves independently, informed only by its own delay history.

### Coordinate Prediction

Three independent linear heads predict the x, y, and z coordinates of the Cα atom at each position from the final manifold state. The model is trained with mean squared error loss in Ångström space.

---

## The Duplication Training Strategy

A key methodological contribution of this work is the deliberate duplication of training samples as a means of deepening geometric structure rather than increasing statistical diversity.

Under standard statistical learning theory, duplicating training data provides no new information and should not improve generalisation. If a model is learning statistical associations, seeing the same example twice teaches it nothing new.

However, a Takens-based model is learning **manifold geometry** — the shape of the conformational attractor, not the statistical frequency of sequence-structure associations. Repeated exposure to the same protein deepens the attractor basin in the learned manifold, analogous to deepening a potential well. The trajectory through conformational space becomes more sharply defined; the manifold curvature around the attractor becomes steeper.

This was first demonstrated clearly in the language modelling experiments (Haylett, 2025), where repeated exposure to identical QA pairs caused the train/validation gap to collapse toward zero — an outcome inconsistent with overfitting (which widens the gap) but consistent with geometric learning (which narrows it as both training and validation trajectories flow through the same deepened attractor tubes).

The same effect is used in the protein preprocessing pipeline. Training proteins are duplicated to increase trajectory density in the learned manifold. The result is more stable, more precisely defined attractor geometry, and improved prediction accuracy on structurally similar proteins.

---

## Computational Properties

| Property | Value |
|---|---|
| Complexity per position | O(log N) |
| Memory footprint | O(1) — fixed circular buffer |
| KV-cache | None |
| Attention | None |
| Positional encodings | None (optional) |
| Hardware requirement | CPU sufficient |

For the proof-of-concept model:

| Property | Value |
|---|---|
| Parameters | ~15M |
| Embedding dimension | 128 |
| Hidden dimension | 512 |
| Layers | 6 |
| Delay schedule | [1, 2, 4, 8, 16, 32, 64, 128] |
| Maximum sequence length | 256 residues |
| Training hardware | Intel i7, 32GB RAM |

---

## Interpretability

The Takens architecture offers a form of geometric interpretability unavailable in attention-based models.

The **projection matrix** `W_p` encodes which temporal scales and combinations of delays are most informative. Analysing its structure reveals what the model has learned about the multi-scale organisation of protein structure.

The **manifold state** at each position is a point in the learned conformational phase space. Trajectories through this space can be visualised and analysed using standard tools from dynamical systems theory — phase portraits, Lyapunov exponents, basin stability measures.

The **attractor structure** can be probed by perturbing the input sequence and observing how the trajectory responds — whether it returns to the same basin (stable attractor) or diverges to a different one (separatrix crossing). This provides a principled framework for understanding why some mutations are structurally neutral and others are destabilising.

---

## Relationship to the Broader TBT Programme

This protein structure prediction work is one application of a general architecture. The same Takens-based transformer — with the same core modules (`takens_embedding.py`, `tbt_architecture.py`) — has been demonstrated to work for:

- **Language modelling** (Brown Corpus, QA, generative text)
- **Protein structure prediction** (this repository)
- **Time series prediction** (Lorenz system, preliminary)

The domain-independence of the architecture is not incidental. Takens' theorem applies to any nonlinear dynamical system with a measurable observable. Language, protein folding, climate systems, cardiac dynamics, and financial time series are all nonlinear dynamical systems. The architecture is domain-agnostic at the mathematical level.

This generality is the central claim of the broader research programme. The transformer architecture is not a statistical pattern matcher — it is a phase space reconstructor. Making that reconstruction explicit, rather than implicit, opens a new design space for sequence models that is only beginning to be explored.

---

## References

Takens, F. (1981). Detecting strange attractors in turbulence. *Dynamical Systems and Turbulence*, Springer Lecture Notes in Mathematics, 898:366–381.

Haylett, K.R. (2025). Pairwise Phase Space Embedding in Transformer Architectures. https://finitemechanics.com/papers/pairwise-embeddings.pdf

Haylett, K.R. (2025). Introducing the Takens-Based Transformer. https://www.finitemechanics.com/takens_transformer.pdf

Haylett, K.R. (2025). Finite Tractus: The Hidden Geometry of Language and Thought. ISBN-13: 979-8281127776.

---

*Kevin R. Haylett, PhD — Manchester, UK — Simul Pariter*
