import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ORCABackbone

class ORCAModel(nn.Module):
    def __init__(self,
                 n_features=36,
                 n_clusters=30,
                 d_model=128,
                 n_bins=64,
                 dropout=0.1,
                 tau_instance=0.5,
                 tau_cluster=1.0):
        super().__init__()

        self.n_clusters = n_clusters
        self.tau_instance = tau_instance
        self.tau_cluster = tau_cluster

        # 1. Backbone
        self.backbone = ORCABackbone(n_features, n_bins, d_model)

        # 2. Heads
        # Instance Head: Non-linear projection (Likely MLP: Linear-Relu-Linear)
        self.instance_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model) # Normalized before loss usually
        )

        # Cluster Head: Soft assignment
        self.cluster_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_clusters),
            nn.Softmax(dim=1)
        )

        # OU Parameter Generator Head
        # Input: Cluster Centroid (d_model)
        # Output: theta, mu, sigma
        self.ou_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
            # Output ranges: theta>0, sigma>0. We handle this in forward (Softplus)
        )

        # Loss Params for Automatic Weighting (Kendall et al. 2018)
        # s_ins, s_clu, s_pinn
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        # x: (Batch, N_features)
        h = self.backbone(x)
        return h

    def get_cluster_prob(self, h):
        return self.cluster_head(h)

    def get_ou_params(self, h_bar):
        # h_bar: (K, d_model) - Centroids
        raw = self.ou_head(h_bar)

        # Enforce constraints
        # theta > 0, sigma > 0. mu is return (unbounded, near 0)
        theta = F.softplus(raw[:, 0])
        mu = raw[:, 1]
        sigma = F.softplus(raw[:, 2]) + 1e-6 # Avoid div by zero

        return theta, mu, sigma

    def compute_loss(self, x_a, x_b, returns, dt=1/21):
        """
        Compute Unified Loss (Instance + Cluster + PINN)
        returns: (Batch,) - Return series at time t (r_t) for PINN validation?
        Wait, PINN Loss requires time series residuals:
        R(t) = (r_t - r_{t-1}) - theta(mu - r_{t-1})dt
        The paper minimizes log likelihood of this residual.

        We need r_t and r_{t-1}.
        The standard training loop usually passes batches of (Features, Returns).
        The Features X_t generate the cluster.
        Using the cluster params, we check if r_t obeys OU wrt r_{t-1}.
        So 'returns' input should be (r_t, r_{t-1}) pair?
        Or the batch contains sequences.
        Simpler: batch includes 'ret' and 'prev_ret' columns.
        """

        # 1. Forward Pass (Two Views)
        h_a = self.backbone(x_a)
        h_b = self.backbone(x_b)

        z_a = F.normalize(self.instance_head(h_a), dim=1)
        z_b = F.normalize(self.instance_head(h_b), dim=1)

        y_a = self.cluster_head(h_a)
        y_b = self.cluster_head(h_b)

        # --- L_ins (Instance Contrastive) ---
        # SimCLR Loss
        l_ins = self.contrastive_loss(z_a, z_b, self.tau_instance)

        # --- L_clu (Cluster Contrastive) ---
        # "Treats cluster assignment probability vectors ... as positive pair"
        # Rows are samples, Cols are clusters.
        # Construct pair similarity on Columns (Features=Batch) or Rows?
        # CC Paper: Contrastive on columns (Clusters).
        # "Maximize similarity between cluster assignments y_a and y_b"
        # We treat Columns of Y (Cluster distributions) as samples?
        # Eq (6) uses s(y^a_i, y^b_i) which is row-wise similarity?
        # Paper Eq(6) sums over i=1..N (samples).
        # So it maximizes similarity of *vector y_i* (cluster prob) for same sample.
        # Wait, if y is One-Hot, similarity is 1. If soft, it aligns.
        # But also minimizes similarity to other samples j.
        l_clu_contrast = self.contrastive_loss(y_a, y_b, self.tau_cluster)

        # Entropy Regularization H(Y)
        # Minimize -H(Y) -> Maximize H(Y) -> Uniform usage of clusters
        # P_k = mean(y_k) across batch
        avg_probs = (y_a + y_b).mean(dim=0) / 2
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
        l_clu = l_clu_contrast - entropy

        # --- L_PINN (Physics Informed) ---
        # 1. Compute Centroids h_bar (Weighted Average)
        # We use y_a to weight h_a (and y_b to weight h_b?)
        # Combine or just use a? Let's average.
        y_avg = (y_a + y_b) / 2
        h_avg = (h_a + h_b) / 2

        # P_ik = y_avg[i, k]
        # h_bar_k = Sum_i (P_ik * h_i) / Sum_i P_ik
        # Shapes: y: (N, K), h: (N, D)
        # Sum_i y_ik * h_id -> (K, D)

        cluster_weights = y_avg.sum(dim=0).unsqueeze(1) + 1e-6 # (K, 1)
        # (K, N) @ (N, D) -> (K, D)
        h_bar = torch.matmul(y_avg.T, h_avg) / cluster_weights

        # 2. Predict OU Params
        theta, mu, sigma = self.get_ou_params(h_bar) # (K,) each

        # 3. Compute Residuals & Likelihood
        # Need r_t, r_{t-1}. passed in 'returns' as (Batch, 2)
        # r_curr = returns[:, 0], r_prev = returns[:, 1]
        r_curr = returns[:, 0]
        r_prev = returns[:, 1]

        # Res_ik = (r_i(t) - r_i(t-1)) - theta_k(mu_k - r_i(t-1))dt
        # We handle broadcasting (N, K)
        # r: (N, 1)
        # theta: (1, K)

        dr = (r_curr - r_prev).unsqueeze(1) # (N, 1)
        r_prev_exp = r_prev.unsqueeze(1) # (N, 1)

        theta_exp = theta.unsqueeze(0) # (1, K)
        mu_exp = mu.unsqueeze(0) # (1, K)

        # Drift: theta * (mu - r_{t-1})
        drift = theta_exp * (mu_exp - r_prev_exp)

        residual = dr - drift * dt # (N, K)

        # 4. Weighted Loss
        # LogLikelihood ~ log(sigma) + Res^2 / (2*sigma^2*dt)
        # We ignore const pi part
        sigma_exp = sigma.unsqueeze(0) # (1, K)

        nll = torch.log(sigma_exp) + (residual ** 2) / (2 * (sigma_exp ** 2) * dt)

        # Weight by P_ik (y_avg)
        # Sum_i Sum_k P_ik * NLL_ik
        l_pinn = (y_avg * nll).sum() / x_a.shape[0] # Average over batch

        return l_ins, l_clu, l_pinn

    def contrastive_loss(self, z_i, z_j, temperature):
        # InfoNCE / NT-Xent
        batch_size = z_i.shape[0]

        # Cosine similarity
        # z are normalized? Yes (instance). Softmax (cluster) is not norm-2 but prob.
        # Paper uses cosine similarity s(u,v) for both via s(a,b).
        # For cluster head (softmax), do we L2 normalize before dot product?
        # Usually yes in Contrastive Clustering.
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Similarity matrices
        # (N, N)
        sim_ij = torch.matmul(z_i, z_j.T) / temperature
        sim_ii = torch.matmul(z_i, z_i.T) / temperature
        sim_ji = torch.matmul(z_j, z_i.T) / temperature
        sim_jj = torch.matmul(z_j, z_j.T) / temperature

        # Positives: diag(sim_ij)
        # Negatives: all others
        # We mask diagonal for self-similarity in ii/jj (not needed for ij part of loss?)
        # Standard SimCLR:
        # l(i,j) = -log( exp(sim_ij) / ( sum_k!=i exp(sim_ik) + sum_k exp(sim_jk) ) )
        # Denom includes all distinct images (2N-1).

        # Simplified:
        # logits = cat(sim_ij, sim_ii_masked)
        # target = arange(N)

        # Let's use direct implementation
        logits = torch.cat([sim_ij, sim_ii], dim=1) # (N, 2N)
        # Create mask to remove self-comparison i,i in sim_ii part?
        # mask diagonal of sim_ii = -inf
        mask = torch.eye(batch_size, device=z_i.device).bool()
        logits[:, batch_size:][mask] = -float('inf')

        labels = torch.arange(batch_size, device=z_i.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def get_weighted_loss(self, l_ins, l_clu, l_pinn):
        """
        Compute weighted loss using learned homoscedastic uncertainty.
        L = Sum (1/2 * exp(-s_i) * L_i + 1/2 * s_i)
        """
        # precision = exp(-log_var)
        precision = torch.exp(-self.log_vars)

        # Losses: [Ins, Clu, PINN]
        # Weighted sum
        # L_total = 0.5 * (precision[0]*l_ins + log_vars[0]) + ...

        loss_ins = 0.5 * (precision[0] * l_ins + self.log_vars[0])
        loss_clu = 0.5 * (precision[1] * l_clu + self.log_vars[1])
        loss_pinn = 0.5 * (precision[2] * l_pinn + self.log_vars[2])

        return loss_ins + loss_clu + loss_pinn
