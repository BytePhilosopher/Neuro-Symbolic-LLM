"""Neuro-symbolic LLM research codebase.

Stage A1 (this stage) implements the *ordinary residual baseline*: a frozen
pretrained transformer plus a small trainable low-rank residual adapter applied
to a hidden state, trained with ordinary backpropagation.

Deliberately absent at A1 (reserved for later stages A2/A3/A4): predictive
coding, latent inference, symbolic heads, and every associated loss term.
"""

__version__ = "0.1.0"

STAGE_A1 = "a1_lora_residual_baseline"

__all__ = ["STAGE_A1", "__version__"]
