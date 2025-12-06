# from sdxl-unbox.SAE.sae import SparseAutoencoder
# from dictionary_learning.dictionary import Dictionary
# import torch

# class SDXL_SAE(Dictionary,SparseAutoencoder):
#     def __init__(self,
#         n_dirs_local: int,
#         d_model: int,
#         k: int,
#         auxk: int | None,
#         dead_steps_threshold: int,):

#         super().__init__(self,
#         n_dirs_local: int,
#         d_model: int,
#         k: int,
#         auxk: int | None,
#         dead_steps_threshold: int,)

#     def encode(self, x):
#         x = x - self.pre_bias
#         latents_pre_act = self.encoder(x) + self.latent_bias

#         vals, inds = torch.topk(
#             latents_pre_act,
#             k=self.k,
#             dim=-1
#         )

#         latents = torch.zeros_like(latents_pre_act)
#         latents.scatter_(-1, inds, torch.relu(vals))

#         return latents

#     def forward(self, x):
#         x = x - self.pre_bias
#         latents_pre_act = self.encoder(x) + self.latent_bias
#         vals, inds = torch.topk(
#             latents_pre_act,
#             k=self.k,
#             dim=-1
#         )

#         ## set num nonzero stat ##
#         tmp = torch.zeros_like(self.stats_last_nonzero)
#         tmp.scatter_add_(
#             0,
#             inds.reshape(-1),
#             (vals > 1e-3).to(tmp.dtype).reshape(-1),
#         )
#         self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
#         self.stats_last_nonzero += 1
#         ## end stats ##

#         ## auxk
#         if self.auxk is not None:  # for auxk
#             # IMPORTANT: has to go after stats update!
#             # WARN: auxk_mask_fn can mutate latents_pre_act!
#             auxk_vals, auxk_inds = torch.topk(
#                 self.auxk_mask_fn(latents_pre_act),
#                 k=self.auxk,
#                 dim=-1
#             )
#         else:
#             auxk_inds = None
#             auxk_vals = None

#         ## end auxk

#         vals = torch.relu(vals)
#         if auxk_vals is not None:
#             auxk_vals = torch.relu(auxk_vals)


#         rows, cols = latents_pre_act.size()
#         row_indices = torch.arange(rows).unsqueeze(1).expand(-1, self.k).reshape(-1)
#         vals = vals.reshape(-1)
#         inds = inds.reshape(-1)

#         indices = torch.stack([row_indices.to(inds.device), inds])

#         sparse_tensor = torch.sparse_coo_tensor(indices, vals, torch.Size([rows, cols]))

#         recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias


#         return recons, {
#             "inds": inds,
#             "vals": vals,
#             "auxk_inds": auxk_inds,
#             "auxk_vals": auxk_vals,
#         }


#     def decode_sparse(self, inds, vals):
#         rows, cols = inds.shape[0], self.n_dirs

#         row_indices = torch.arange(rows).unsqueeze(1).expand(-1, inds.shape[1]).reshape(-1)
#         vals = vals.reshape(-1)
#         inds = inds.reshape(-1)

#         indices = torch.stack([row_indices.to(inds.device), inds])

#         sparse_tensor = torch.sparse_coo_tensor(indices, vals, torch.Size([rows, cols]))

#         recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias
#         return recons


