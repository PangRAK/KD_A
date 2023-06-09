import torch

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils.module_with_records import ModuleWithRecords
import torch.nn.functional as F


class CrossBatchMemory(ModuleWithRecords):
    def __init__(self, loss, embedding_size, memory_size=1024, miner=None, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.miner = miner
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        
        

        self.reset_queue()
        
        
        
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "memory_size", "queue_idx"], is_stat=False
        )

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_idx=None):
        if enqueue_idx is not None:
            assert len(enqueue_idx) <= len(self.embedding_memory)
            assert len(enqueue_idx) < len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        self.reset_stats()
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(
            self.embedding_memory, device=device, dtype=embeddings.dtype
        )
        self.label_memory = c_f.to_device(
            self.label_memory, device=device, dtype=labels.dtype
        )

        if enqueue_idx is not None:
            mask = torch.zeros(
                len(embeddings), device=device, dtype=torch.bool)
            mask[enqueue_idx] = True
            # print("embedding ", embeddings.shape)  # [512, 10]
            emb_for_queue = embeddings[mask]
            # print("emb for queue: ", emb_for_queue.shape)   # [256, 10]
            labels_for_queue = labels[mask]

            embeddings = embeddings[~mask]
            labels = labels[~mask]
            do_remove_self_comparisons = False
            # do_remove_self_comparisons = True

        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            do_remove_self_comparisons = True

        batch_size = len(embeddings)

        queue_batch_size = len(emb_for_queue)


        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

        if not self.has_been_filled:
            E_mem = self.embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory


        # # [B, K]
        # sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)


        indices_tuple = self.create_indices_tuple(
            batch_size,
            embeddings,
            labels,
            E_mem,
            L_mem,
            indices_tuple, # 이전에는 None
            do_remove_self_comparisons
        )

        # print("ㅡㅡ",len(indices_tuple[0]),end="\n\n\n")

        # print(indices_tuple)  # a, p ,n ?

        # loss= self.loss(embeddings, indices_tuple = indices_tuple, E_mem, L_mem)
        loss = self.loss(embeddings, labels, ref_emb=E_mem, ref_labels=L_mem)

        # a, p, n =indices_tuple
        # return E_mem, L_mem, loss, self.has_been_filled, self.queue_idx
        return loss , self.label_memory


    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )

        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True
            
            

    def create_indices_tuple(
        self,
        batch_size,
        embeddings,
        labels,
        E_mem,
        L_mem,
        input_indices_tuple,
        do_remove_self_comparisons,
    ):
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)

        if do_remove_self_comparisons:
            indices_tuple = lmu.remove_self_comparisons(
                indices_tuple, self.curr_batch_idx, self.memory_size
            )

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(
                    input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = c_f.concatenate_indices_tuples(
                indices_tuple, input_indices_tuple
            )

        return indices_tuple

    def reset_queue(self):
        self.embedding_memory = torch.zeros(
            self.memory_size, self.embedding_size)
        # print("self embedding size: ", self.embedding_size)
        self.label_memory = torch.zeros(self.memory_size).long()
        self.has_been_filled = False
        self.queue_idx = 0
