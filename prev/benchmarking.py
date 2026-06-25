import time
import torch

class PerformanceProfiler:
    """Captures native, non-theoretical hardware metrics across token generation streams."""

    @staticmethod
    def profile_generation_speed(model, tokenizer, text_payload: str, tracker) -> dict:
        """
        Measures true wall-clock token generation throughput and active VRAM footprint.
        """
        model.eval()
        tracker.update({
            "tokens_q4": torch.tensor(0, device=model.device, dtype=torch.long),
            "tokens_q8": torch.tensor(0, device=model.device, dtype=torch.long),
            "tokens_fp16": torch.tensor(0, device=model.device, dtype=torch.long),
            "total_tokens": torch.tensor(0, device=model.device, dtype=torch.long),
            "bytes_saved": torch.tensor(0.0, device=model.device, dtype=torch.float64)
        })

        inputs = tokenizer(text_payload * 4, return_tensors="pt", truncation=True, max_length=128).to(model.device)

        with torch.no_grad():
            for _ in range(3):
                _ = model(**inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(12):
                _ = model(**inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            vram = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        else:
            vram = 0.0
        end = time.perf_counter()

        t4 = tracker["tokens_q4"].item()
        t8 = tracker["tokens_q8"].item()
        f16 = tracker["tokens_fp16"].item()
        tot = max(1, tracker["total_tokens"].item())
        bytes_saved = tracker["bytes_saved"].item()

        elapsed = end - start
        total_sequence_tokens = inputs["input_ids"].size(0) * inputs["input_ids"].size(1) * 12
        speeds = total_sequence_tokens / max(1e-5, elapsed)
        bpt = ((t4 * 4) + (t8 * 8) + (f16 * 16)) / tot

        return {
            "speed": speeds,
            "bpt": bpt,
            "vram": vram,
            "saved_mb": bytes_saved / (1024 ** 2),
            "splits": (t4/tot, t8/tot, f16/tot)
        }
