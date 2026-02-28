"""
Benchmarking suite for the Transformer model using the GPT-2 vocabulary (50,257 tokens).

Tests convergence on synthetic pretraining tasks of increasing difficulty:

1. Memorization       – Can the model memorize a small fixed dataset?
2. Copying            – Can the model learn to copy an input sequence after a separator token?
3. Reversing          – Can the model learn to reverse a sequence? (harder positional reasoning)
4. Bigram Statistics  – Can the model learn token-pair transition probabilities from a synthetic corpus?

All benchmarks use vocab_size=50257 (GPT-2) to test generalization at realistic scale.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import Transformer, TransformerConfig

# ────────────────────────────────────────────────────────────────────
# GPT-2 Vocabulary
# ────────────────────────────────────────────────────────────────────

VOCAB_PATH = Path(__file__).parent / "vocab.json"

def load_gpt2_vocab() -> dict[str, int]:
    with open(VOCAB_PATH) as f:
        return json.load(f)

GPT2_VOCAB_SIZE = 50257  # official GPT-2 vocab size

# Use well-known GPT-2 special token ids
# <|endoftext|> = 50256, we use it as our separator
SEP_TOKEN = 50256


# ────────────────────────────────────────────────────────────────────
# Synthetic Datasets
# ────────────────────────────────────────────────────────────────────


class MemorizationDataset(Dataset):
    """Fixed set of random sequences the model must memorize.
    Tokens are drawn uniformly from the full GPT-2 vocab."""

    def __init__(self, num_sequences: int, seq_len: int, vocab_size: int, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_sequences, seq_len), generator=rng)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


class CopyDataset(Dataset):
    """Input: [random tokens] SEP → Target: [same random tokens].

    Full sequence: [tok1 tok2 ... tokN SEP tok1 tok2 ... tokN]
    Tokens are drawn from a subset of the GPT-2 vocab to keep the task
    learnable while still exercising the full 50K embedding table.
    """

    def __init__(
        self, num_sequences: int, half_len: int, vocab_size: int,
        num_active_tokens: int = 512, seed: int = 42,
        active_tokens: torch.Tensor | None = None,
    ):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        if active_tokens is not None:
            self.active_tokens = active_tokens
        else:
            # Use a fixed seed for active token selection (independent of data seed)
            at_rng = torch.Generator().manual_seed(777)
            self.active_tokens = torch.randperm(vocab_size, generator=at_rng)[:num_active_tokens]
        n = len(self.active_tokens)
        indices = torch.randint(0, n, (num_sequences, half_len), generator=rng)
        source = self.active_tokens[indices]
        sep = torch.full((num_sequences, 1), SEP_TOKEN, dtype=torch.long)
        self.data = torch.cat([source, sep, source], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


class ReverseDataset(Dataset):
    """Input: [random tokens] SEP → Target: [reversed random tokens].

    Full sequence: [tok1 tok2 ... tokN SEP tokN ... tok2 tok1]
    Tokens are drawn from a subset of the GPT-2 vocab.
    """

    def __init__(
        self, num_sequences: int, half_len: int, vocab_size: int,
        num_active_tokens: int = 512, seed: int = 42,
        active_tokens: torch.Tensor | None = None,
    ):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        if active_tokens is not None:
            self.active_tokens = active_tokens
        else:
            at_rng = torch.Generator().manual_seed(888)
            self.active_tokens = torch.randperm(vocab_size, generator=at_rng)[:num_active_tokens]
        n = len(self.active_tokens)
        indices = torch.randint(0, n, (num_sequences, half_len), generator=rng)
        source = self.active_tokens[indices]
        sep = torch.full((num_sequences, 1), SEP_TOKEN, dtype=torch.long)
        reversed_source = source.flip(dims=[1])
        self.data = torch.cat([source, sep, reversed_source], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


class BigramDataset(Dataset):
    """Sequences drawn from a random bigram (Markov-1) transition matrix
    over a subset of GPT-2 tokens.

    Uses num_active_tokens from the full vocab to keep the transition
    matrix tractable while exercising the full 50K embedding/output layers.
    """

    def __init__(
        self,
        num_sequences: int,
        seq_len: int,
        vocab_size: int,
        num_active_tokens: int = 256,
        seed: int = 42,
        transition: torch.Tensor | None = None,
        active_tokens: torch.Tensor | None = None,
        transition_seed: int = 0,
    ):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)

        if active_tokens is not None:
            self.active_tokens = active_tokens
        else:
            at_rng = torch.Generator().manual_seed(transition_seed + 1000)
            self.active_tokens = torch.randperm(vocab_size, generator=at_rng)[:num_active_tokens]

        n = len(self.active_tokens)

        if transition is not None:
            self.transition = transition
        else:
            t_rng = torch.Generator().manual_seed(transition_seed)
            raw = torch.rand(n, n, generator=t_rng)
            raw = raw ** 8  # make it peaky
            self.transition = raw / raw.sum(dim=1, keepdim=True)

        # Sample sequences using indices into active_tokens
        data_indices = torch.zeros(num_sequences, seq_len, dtype=torch.long)
        data_indices[:, 0] = torch.randint(0, n, (num_sequences,), generator=rng)
        for t in range(1, seq_len):
            prev = data_indices[:, t - 1]
            probs = self.transition[prev]
            data_indices[:, t] = torch.multinomial(probs, 1, generator=rng).squeeze(1)

        # Map indices to actual GPT-2 token ids
        self.data = self.active_tokens[data_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


# ────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    final_loss: float
    final_perplexity: float
    final_accuracy: float
    target_loss: float
    target_accuracy: float
    epochs_run: int
    wall_time_sec: float
    loss_history: list
    accuracy_history: list


def train_and_evaluate(
    name: str,
    model: Transformer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    max_epochs: int,
    batch_size: int,
    lr: float,
    target_loss: float,
    target_accuracy: float,
    device: torch.device,
    eval_only_suffix: int = 0,
    patience: int = 20,
    warmup_epochs: int = 5,
) -> BenchmarkResult:
    """Train the model and evaluate convergence.

    Args:
        eval_only_suffix: if > 0, only evaluate loss/accuracy on the last N
            tokens of each sequence (useful for copy/reverse where the first
            half is just context).
        patience: early stop if eval loss hasn't improved in this many epochs.
        warmup_epochs: number of epochs for linear LR warmup.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    # Linear warmup then cosine decay
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    loss_history = []
    accuracy_history = []
    best_loss = float("inf")
    stale_epochs = 0
    avg_eval_loss = float("inf")
    eval_acc = 0.0
    eval_ppl = float("inf")
    epoch = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * targets.numel()
            epoch_tokens += targets.numel()
        scheduler.step()

        # ── Eval ──
        model.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_tokens = 0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)

                if eval_only_suffix > 0:
                    logits = logits[:, -eval_only_suffix:, :]
                    targets = targets[:, -eval_only_suffix:]

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
                )
                preds = logits.argmax(dim=-1)
                eval_loss += loss.item()
                eval_correct += (preds == targets).sum().item()
                eval_tokens += targets.numel()

        avg_eval_loss = eval_loss / eval_tokens
        eval_acc = eval_correct / eval_tokens
        eval_ppl = math.exp(min(avg_eval_loss, 20.0))  # cap to avoid overflow

        loss_history.append(avg_eval_loss)
        accuracy_history.append(eval_acc)

        # Logging every 10 epochs or first/last
        if epoch <= 3 or epoch % 10 == 0 or epoch == max_epochs:
            grad_norm = sum(
                p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
            ) ** 0.5
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  [{name}] Epoch {epoch:4d}/{max_epochs} | "
                f"train_loss={epoch_loss / epoch_tokens:.4f}  "
                f"eval_loss={avg_eval_loss:.4f}  ppl={eval_ppl:.2f}  "
                f"acc={eval_acc:.4f}  grad_norm={grad_norm:.4f}  lr={lr_now:.2e}"
            )

        # Early stopping
        if avg_eval_loss < best_loss - 1e-4:
            best_loss = avg_eval_loss
            stale_epochs = 0
        else:
            stale_epochs += 1

        # Check if we already meet the targets
        if avg_eval_loss <= target_loss and eval_acc >= target_accuracy:
            print(f"  [{name}] Converged at epoch {epoch}!")
            break

        if stale_epochs >= patience and epoch > warmup_epochs + 10:
            print(f"  [{name}] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    wall_time = time.time() - start_time
    passed = avg_eval_loss <= target_loss and eval_acc >= target_accuracy

    return BenchmarkResult(
        name=name,
        passed=passed,
        final_loss=avg_eval_loss,
        final_perplexity=eval_ppl,
        final_accuracy=eval_acc,
        target_loss=target_loss,
        target_accuracy=target_accuracy,
        epochs_run=epoch,
        wall_time_sec=wall_time,
        loss_history=loss_history,
        accuracy_history=accuracy_history,
    )


# ────────────────────────────────────────────────────────────────────
# Benchmark Definitions
# ────────────────────────────────────────────────────────────────────


def run_memorization(device: torch.device) -> BenchmarkResult:
    """Test 1: Memorize 64 fixed sequences of length 64 using full GPT-2 vocab."""
    config = TransformerConfig(
        vocab_size=GPT2_VOCAB_SIZE, max_seq_len=64, d_model=384, num_heads=6,
        num_layers=6, d_ff=1536, dropout=0.0,  # no dropout for memorization
    )
    model = Transformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  [Memorization] Parameters: {num_params:,}")

    train_ds = MemorizationDataset(num_sequences=64, seq_len=64, vocab_size=GPT2_VOCAB_SIZE, seed=42)
    eval_ds = train_ds  # same data — we want perfect memorization

    return train_and_evaluate(
        name="Memorization", model=model, train_dataset=train_ds, eval_dataset=eval_ds,
        max_epochs=500, batch_size=32, lr=3e-4,
        target_loss=0.10, target_accuracy=0.99, device=device,
        patience=50,
    )


def run_copy(device: torch.device) -> BenchmarkResult:
    """Test 2: Copy 8-token sequences after a separator.
    Tokens drawn from 32 active GPT-2 tokens, full 50K output layer."""
    config = TransformerConfig(
        vocab_size=GPT2_VOCAB_SIZE, max_seq_len=64, d_model=256, num_heads=8,
        num_layers=4, d_ff=1024, dropout=0.05,
    )
    model = Transformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  [Copy] Parameters: {num_params:,}")

    train_ds = CopyDataset(
        num_sequences=10000, half_len=8, vocab_size=GPT2_VOCAB_SIZE,
        num_active_tokens=32, seed=42,
    )
    eval_ds = CopyDataset(
        num_sequences=2000, half_len=8, vocab_size=GPT2_VOCAB_SIZE,
        num_active_tokens=32, seed=99,
        active_tokens=train_ds.active_tokens,
    )

    return train_and_evaluate(
        name="Copy", model=model, train_dataset=train_ds, eval_dataset=eval_ds,
        max_epochs=300, batch_size=128, lr=5e-4,
        target_loss=0.05, target_accuracy=0.99, device=device,
        eval_only_suffix=8,
        patience=80,
    )


def run_reverse(device: torch.device) -> BenchmarkResult:
    """Test 3: Reverse 8-token sequences after a separator.
    Tokens drawn from 32 active GPT-2 tokens, full 50K output layer."""
    config = TransformerConfig(
        vocab_size=GPT2_VOCAB_SIZE, max_seq_len=64, d_model=256, num_heads=8,
        num_layers=4, d_ff=1024, dropout=0.1,
    )
    model = Transformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  [Reverse] Parameters: {num_params:,}")

    train_ds = ReverseDataset(
        num_sequences=10000, half_len=8, vocab_size=GPT2_VOCAB_SIZE,
        num_active_tokens=32, seed=42,
    )
    eval_ds = ReverseDataset(
        num_sequences=2000, half_len=8, vocab_size=GPT2_VOCAB_SIZE,
        num_active_tokens=32, seed=99,
        active_tokens=train_ds.active_tokens,
    )

    return train_and_evaluate(
        name="Reverse", model=model, train_dataset=train_ds, eval_dataset=eval_ds,
        max_epochs=300, batch_size=128, lr=5e-4,
        target_loss=0.10, target_accuracy=0.98, device=device,
        eval_only_suffix=8,
        patience=50,
    )


def run_bigram(device: torch.device) -> BenchmarkResult:
    """Test 4: Learn bigram transition statistics over 32 active GPT-2 tokens."""
    config = TransformerConfig(
        vocab_size=GPT2_VOCAB_SIZE, max_seq_len=128, d_model=256, num_heads=8,
        num_layers=4, d_ff=1024, dropout=0.15,
    )
    model = Transformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  [Bigram] Parameters: {num_params:,}")

    train_ds = BigramDataset(
        num_sequences=8192, seq_len=64, vocab_size=GPT2_VOCAB_SIZE,
        num_active_tokens=32, seed=42, transition_seed=0,
    )
    eval_ds = BigramDataset(
        num_sequences=1024, seq_len=64, vocab_size=GPT2_VOCAB_SIZE,
        num_active_tokens=32, seed=99,
        transition=train_ds.transition, active_tokens=train_ds.active_tokens,
    )

    # Compute theoretical lower bound
    transition = train_ds.transition
    entropy = -(transition * torch.log(transition + 1e-10)).sum(dim=1).mean().item()
    best_acc = transition.max(dim=1).values.mean().item()
    print(f"  [Bigram] Active tokens: {len(train_ds.active_tokens)} / {GPT2_VOCAB_SIZE}")
    print(f"  [Bigram] Theoretical entropy lower bound: {entropy:.4f}")
    print(f"  [Bigram] Theoretical best top-1 accuracy: {best_acc:.4f}")

    return train_and_evaluate(
        name="Bigram", model=model, train_dataset=train_ds, eval_dataset=eval_ds,
        max_epochs=200, batch_size=64, lr=5e-4,
        target_loss=entropy + 0.05, target_accuracy=best_acc * 0.95, device=device,
        patience=60,
    )


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

BENCHMARKS = [
    ("Memorization", run_memorization),
    ("Copy", run_copy),
    ("Reverse", run_reverse),
    ("Bigram", run_bigram),
]


def main():
    # Verify GPT-2 vocab is present
    if not VOCAB_PATH.exists():
        print(f"ERROR: GPT-2 vocab not found at {VOCAB_PATH}")
        print("Download it: curl -sL -o vocab.json https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json")
        return False

    vocab = load_gpt2_vocab()
    print(f"GPT-2 vocab loaded: {len(vocab)} tokens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results: list[BenchmarkResult] = []

    for bench_name, bench_fn in BENCHMARKS:
        print(f"{'=' * 70}")
        print(f"BENCHMARK: {bench_name}")
        print(f"{'=' * 70}")
        result = bench_fn(device)
        results.append(result)
        print()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Benchmark':<15} {'Status':<8} {'Loss':>8} {'Target':>8} {'Acc':>8} {'Target':>8} {'Epochs':>8} {'Time':>8}")
    print(f"{'-' * 15} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        all_passed = all_passed and r.passed
        print(
            f"{r.name:<15} {status:<8} {r.final_loss:>8.4f} {r.target_loss:>8.4f} "
            f"{r.final_accuracy:>8.4f} {r.target_accuracy:>8.4f} {r.epochs_run:>8d} {r.wall_time_sec:>7.1f}s"
        )

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"Total wall time: {sum(r.wall_time_sec for r in results):.1f}s")

    return all_passed


if __name__ == "__main__":
    main()
