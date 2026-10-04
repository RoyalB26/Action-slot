import copy
import random
import torch
import torch.nn as nn


class PCGrad:
    def __init__(self, optimizer):
        self._optim = optimizer

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def pcgrad_backward(self, objectives):
        """
        objectives: Danh sách các loss tensors cần tối ưu đồng thời.
        Ví dụ: [loss_supervised, loss_grpo]
        """
        assert isinstance(objectives, list) and len(objectives) > 0

        # Lọc các loss có gradient (bỏ qua loss bằng 0 hoặc không requires_grad)
        active_objectives = [
            obj for obj in objectives 
            if isinstance(obj, torch.Tensor) and obj.requires_grad and obj.grad_fn is not None
        ]

        if len(active_objectives) == 0:
            return
        if len(active_objectives) == 1:
            active_objectives[0].backward()
            return

        # 1. Tính toán gradient riêng biệt cho từng objective
        grads, shapes, has_grads = self._pack_grad(active_objectives)

        # 2. Chiếu trực giao khi phát hiện xung đột gradient (g_i . g_j < 0)
        pc_grads = self._project_conflicting(grads, has_grads)

        # 3. Gán gradient đã được xử lý xung đột ngược lại vào tham số mô hình
        self._set_grad(pc_grads)

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(dim=0).bool()
        pc_grads = copy.deepcopy(grads)

        for i in range(len(pc_grads)):
            g_i = pc_grads[i]
            # Shuffle thứ tự để tránh thiên vị task nào trước
            indices = list(range(len(pc_grads)))
            random.shuffle(indices)

            for j in indices:
                g_j = grads[j]
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    # g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j
                    g_i -= (g_i_g_j / (g_j.norm() ** 2 + 1e-8)) * g_j

        # Tổng hợp gradient không còn xung đột
        merged_grad = torch.zeros_like(grads[0])
        for g in pc_grads:
            merged_grad += g

        return merged_grad

    def _set_grad(self, grads):
        """Đưa gradient 1D gộp trở lại từng parameter trong optimizer."""
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                numel = p.data.numel()
                p.grad.data.copy_(grads[idx : idx + numel].view_as(p.data))
                idx += numel

    def _pack_grad(self, objectives):
        """Lấy flat gradient vector cho từng task."""
        grads = []
        has_grads = []

        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)

            grad_list = []
            has_grad_list = []

            for group in self._optim.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        grad_list.append(torch.zeros_like(p.data).view(-1))
                        has_grad_list.append(torch.zeros(p.data.numel(), dtype=torch.bool, device=p.device))
                    else:
                        grad_list.append(p.grad.data.clone().view(-1))
                        has_grad_list.append(torch.ones(p.data.numel(), dtype=torch.bool, device=p.device))

            grads.append(torch.cat(grad_list))
            has_grads.append(torch.cat(has_grad_list))

        self._optim.zero_grad()
        return grads, None, has_grads