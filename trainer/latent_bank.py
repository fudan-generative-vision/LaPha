# trainer/latent_bank.py
import torch
import torch.nn.functional as F

class LatentBank:
    """
    Append-only latent store with optional CPU mirror.
    - add(h): h is CPU tensor from your value_fn (B,H). We normalize & cast, then:
        * keep a CUDA shard (if CUDA available)
        * optionally keep a CPU mirror (store_cpu_copy=True)
      Returns global row indices.
    - index_select(idxs): returns a CUDA tensor (N,H). If CUDA shards missing but CPU mirror exists,
      it will move the selected rows to CUDA on-the-fly.
    - offload_to_cpu(delete_cuda=True): ensure CPU mirror exists, then free CUDA shards (optional).
    - reload_to_gpu(): rebuild a single CUDA tensor from CPU mirror (when you need fast batched math again).
    - clear(): free both CUDA and CPU storages.
    """
    def __init__(self, device, dtype=torch.bfloat16, store_cpu_copy=True, normalize=True):
        self.device = torch.device(device)
        self.dtype = dtype
        self.normalize = bool(normalize)
        self.store_cpu_copy = bool(store_cpu_copy)

        self._cuda_shards = []   # list[Tensor[m_i, H]] on CUDA
        self._cpu_shards  = []   # list[Tensor[m_i, H]] on CPU
        self._shape_H = None     # hidden dim
        self._length = 0

        # cached concatenations
        self._cuda_cat = None    # Tensor[N,H] on CUDA (optional cache)
        self._cpu_cat  = None    # Tensor[N,H] on CPU (optional cache)

    @property
    def N(self) -> int:
        return int(self._length)

    def _invalidate_cats(self):
        self._cuda_cat = None
        self._cpu_cat = None

    @torch.no_grad()
    def add(self, h_cpu: torch.Tensor):
        """
        h_cpu: CPU tensor of shape (B,H) or (1,H) returned by your value_fn.
        Returns:
          - int if B==1
          - list[int] if B>1
        """
        assert h_cpu.device.type == "cpu", "LatentBank.add expects CPU tensor from value_fn()."
        if h_cpu.ndim != 2:
            h_cpu = h_cpu.view(h_cpu.size(0), -1)
        if self._shape_H is None:
            self._shape_H = int(h_cpu.size(1))
        else:
            assert h_cpu.size(1) == self._shape_H, "Hidden size mismatch across additions."

        # normalize (cosine distance space)
        if self.normalize:
            h_cpu = F.normalize(h_cpu.float(), dim=-1)

        # cast dtype
        h_cpu = h_cpu.to(dtype=self.dtype, copy=False)

        B = int(h_cpu.size(0))
        idx0 = self._length
        idxs = list(range(idx0, idx0 + B))
        self._length += B

        # shard append
        # 1) CUDA shard
        if self.device.type == "cuda":
            h_cuda = h_cpu.to(self.device, non_blocking=True)
            self._cuda_shards.append(h_cuda)

        # 2) CPU shard (mirror or sole storage)
        if self.store_cpu_copy or (self.device.type != "cuda"):
            self._cpu_shards.append(h_cpu)  # keep the incoming CPU tensor (already dtype-cast)

        self._invalidate_cats()
        return idxs[0] if B == 1 else idxs

    def _get_cuda_cat(self):
        if self._cuda_cat is not None:
            return self._cuda_cat
        if len(self._cuda_shards) == 0:
            return None
        self._cuda_cat = torch.cat(self._cuda_shards, dim=0) if len(self._cuda_shards) > 1 else self._cuda_shards[0]
        return self._cuda_cat

    def _get_cpu_cat(self):
        if self._cpu_cat is not None:
            return self._cpu_cat
        if len(self._cpu_shards) == 0:
            return None
        self._cpu_cat = torch.cat(self._cpu_shards, dim=0) if len(self._cpu_shards) > 1 else self._cpu_shards[0]
        return self._cpu_cat

    @torch.no_grad()
    def index_select(self, indices):
        """
        Return CUDA tensor (n, H) for given indices. If CUDA storage is not present but we have CPU,
        we will gather on CPU and move the slice to CUDA.
        """
        if isinstance(indices, (list, tuple)):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device if self.device.type=="cuda" else "cpu")
        elif isinstance(indices, torch.Tensor):
            if indices.dtype != torch.long:
                indices = indices.to(torch.long)
            # ensure device consistent
            if self.device.type == "cuda" and indices.device.type != "cuda":
                indices = indices.to(self.device, non_blocking=True)
            if self.device.type != "cuda" and indices.device.type == "cuda":
                indices = indices.to("cpu", non_blocking=True)
        else:
            indices = torch.tensor([int(indices)], dtype=torch.long, device=self.device if self.device.type=="cuda" else "cpu")

        cuda_cat = self._get_cuda_cat()
        if cuda_cat is not None:
            return cuda_cat.index_select(0, indices if indices.device == cuda_cat.device else indices.to(cuda_cat.device))

        # fallback: gather from CPU and move to CUDA as a slice
        cpu_cat = self._get_cpu_cat()
        if cpu_cat is None:
            raise RuntimeError("LatentBank is empty or has no storage.")
        sel_cpu = cpu_cat.index_select(0, indices.to("cpu"))
        if self.device.type == "cuda":
            return sel_cpu.to(self.device, non_blocking=True)
        return sel_cpu

    @torch.no_grad()
    def offload_to_cpu(self, delete_cuda: bool = True, pin_memory: bool = False):
        """
        Ensure CPU mirror exists, then optionally delete CUDA shards to free VRAM.
        """
        # Build CPU concat if missing
        if len(self._cpu_shards) == 0 and len(self._cuda_shards) > 0:
            for t in self._cuda_shards:
                self._cpu_shards.append(t.to("cpu", non_blocking=False))
            self._invalidate_cats()

        if pin_memory and len(self._cpu_shards) > 0:
            self._cpu_shards = [t.pin_memory() for t in self._cpu_shards]
            self._invalidate_cats()

        if delete_cuda and len(self._cuda_shards) > 0:
            # Aggressively free CUDA shards
            for t in self._cuda_shards:
                try:
                    t.storage().resize_(0)  # free underlying storage
                except Exception:
                    pass
            self._cuda_shards.clear()
            self._cuda_cat = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    @torch.no_grad()
    def reload_to_gpu(self):
        """
        Move the whole bank back to CUDA as a single concatenated tensor for fast batched ops.
        """
        if self.device.type != "cuda":
            return
        cpu_cat = self._get_cpu_cat()
        if cpu_cat is None:
            return
        self._cuda_shards = [cpu_cat.to(self.device, non_blocking=False)]
        self._cuda_cat = self._cuda_shards[0]
        # keep CPU mirror
        # (if you don't need it, you can clear CPU shards here)

    @torch.no_grad()
    def clear(self):
        """
        Free both CPU and CUDA storages.
        """
        for buf in self._cuda_shards:
            try:
                buf.storage().resize_(0)
            except Exception:
                pass
        self._cuda_shards.clear()
        self._cuda_cat = None

        for buf in self._cpu_shards:
            try:
                buf.storage().resize_(0)
            except Exception:
                pass
        self._cpu_shards.clear()
        self._cpu_cat = None

        self._shape_H = None
        self._length = 0

    def stats(self):
        """
        Return a tiny dict for logging.
        """
        H = self._shape_H or -1
        return {
            "N": self.N,
            "H": H,
            "cuda_shards": len(self._cuda_shards),
            "cpu_shards": len(self._cpu_shards),
            "has_cuda_cat": self._cuda_cat is not None,
            "has_cpu_cat": self._cpu_cat is not None,
        }
