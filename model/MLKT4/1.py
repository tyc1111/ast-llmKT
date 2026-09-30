"""
mlkt4_kc_mastery.py — VAE Per-KC Mastery (单步KL)

核心架构:
  VAE + 双隐变量 (Z_k 知识, Z_e 执行)
  编码器每步推断后验, 重参数化采样驱动状态转移
  KL仅在最后一步计算 (不是每步都算, 避免KL压倒BCE)

  Loss = BCE + beta * KL_last
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEPerKCMasteryModel(nn.Module):
    def __init__(self, num_questions, num_kcs, D, K=8, device='cpu', beta=0.1):
        super().__init__()
        self.num_kcs = num_kcs
        self.num_questions = num_questions + 1
        self.D = D
        self.K = K
        self.device = device
        self.beta = beta

        # ---- Embeddings ----
        self.kc_emb = nn.Embedding(num_kcs + 1, D, padding_idx=0)
        self.problem_emb = nn.Embedding(self.num_questions, D)
        self.score_emb = nn.Embedding(101, D)

        # ---- Reason编码 ----
        self.reason_mlp = nn.Sequential(
            nn.Linear(5, D), nn.ReLU(), nn.Linear(D, D),
        )

        # ---- Encoder (per-KC) ----
        # 所有输入统一投影到D维再相加, 避免标量与embedding量纲不匹配
        self.zk_proj = nn.Linear(K, D)      # Z_k 向量(K维)→D维
        self.z_proj = nn.Linear(1, D)       # Z_e 标量→D维
        self.ti_proj = nn.Linear(1, D)      # time interval 标量→D维
        self.mode_proj = nn.Linear(1, D)    # mode 标量→D维
        self.dropout = nn.Dropout(0.3)
        self.enc_shared = nn.Sequential(
            nn.Linear(D, D * 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(D * 2, D), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(D, D), nn.ReLU(),
        )
        self.k_mu = nn.Linear(D, K)          # Z_k向量后验均值
        self.k_lv = nn.Linear(D, K)          # Z_k向量后验对数方差
        self.e_mu = nn.Linear(D, 1)
        self.e_lv = nn.Linear(D, 1)

        # ---- KC图注意力 (GAT) ----
        self.register_buffer('kc_adj', torch.eye(num_kcs))
        self.gat_query = nn.Linear(D, D)
        self.gat_key = nn.Linear(D, D)
        self.gat_proj = nn.Linear(D, D)
        self.gat_norm = nn.LayerNorm(D)

        # ---- Prior参数 (可学习) ----
        self.log_tau_k = nn.Parameter(torch.log(torch.tensor(0.2)))
        self.log_tau_e = nn.Parameter(torch.log(torch.tensor(0.15)))
        self.log_gamma = nn.Parameter(torch.logit(torch.tensor(0.7)))

        # ---- 执行权重 ----
        self.w_exec = nn.Parameter(torch.tensor(0.0))

        # ---- KC权重 + IRT ----
        self.kc_weight = nn.Parameter(torch.zeros(self.num_questions, num_kcs))
        self.problem_difficulty = nn.Parameter(torch.zeros(self.num_questions))
        self.problem_discrimination = nn.Parameter(torch.ones(self.num_questions))
        self.ability_proj = nn.Linear(K, 1)   # 向量化能力→标量

        # ---- 温度缩放: 单调变换, 拉ACC不动AUC ----
        # log参数化保证temp>0; 初始temp=exp(0)=1.0 (不改基线行为)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        # ---- 初始状态 (logit空间) ----
        self.init_y = nn.Parameter(torch.zeros(num_kcs, K))   # 向量化初始知识状态
        self.init_e = nn.Parameter(torch.zeros(num_kcs))

        # ---- 模式检测 ----
        self.momentum = 0.9
        self.mode_threshold = nn.Parameter(torch.tensor(1.5))
        self.reason_weight = nn.Parameter(torch.tensor(0.3))

        # ---- 遗忘参数 (方案3: 练习次数调制的幂律遗忘) ----
        self.log_tau_0 = nn.Parameter(torch.log(torch.tensor(5000.0)))    # 基线半衰期(分钟)
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(0.3)))        # 练习增强系数

        # ---- 吸收率参数 (分数×模式依赖) ----
        self.log_score_abs = nn.Parameter(torch.log(torch.tensor(2.0)))     # 分数→吸收率的映射斜率
        self.debug_abs_scale = nn.Parameter(torch.tensor(0.3))              # 调试模式吸收衰减

    # ================================================================
    # 工具
    # ================================================================

    def _kl_gaussian(self, mu_q, lv_q, mu_p, lv_p):
        var_q = torch.exp(lv_q)
        var_p = torch.exp(lv_p).clamp(min=1e-8)
        kl = 0.5 * (lv_p - lv_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0)
        return torch.nan_to_num(kl, nan=0.0, posinf=5.0, neginf=0.0)

    def _reparameterize(self, mu, lv):
        lv = lv.clamp(-10, 5)
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def _weighted_kc_ability(self, state, kc_mask_t, q_ids):
        # state: [B, N, K] 向量化KC状态
        raw_w = self.kc_weight[q_ids]
        has_any_kc = kc_mask_t.sum(dim=1, keepdim=True) > 0
        safe_mask = kc_mask_t.clone() + (~has_any_kc).float()
        masked_w = raw_w.masked_fill(safe_mask == 0, float('-inf'))
        w = F.softmax(masked_w, dim=-1)
        w = torch.where(has_any_kc, w, torch.zeros_like(w))
        weighted = (state * w.unsqueeze(-1)).sum(dim=1)  # [B, K]
        return self.ability_proj(weighted).squeeze(-1)   # [B]

    def _encode(self, Z_k, Z_e, s_emb, r_emb, q_emb, ti, mode=None):
        """Per-KC编码器, 返回 mu/lv; mode: [B] (0=调试, 1=学习)"""
        N = self.num_kcs
        s_exp = s_emb.unsqueeze(1).expand(-1, N, -1)
        r_exp = r_emb.unsqueeze(1).expand(-1, N, -1)
        q_exp = q_emb.unsqueeze(1).expand(-1, N, -1)
        # 标量输入先投影到D维, 再expand到[B,N,D]
        # 时间间隔对数缩放: log(1+ti) 缓解极端右偏(median 3.9 vs mean 531分钟)
        # 仅用于编码器输入; 遗忘衰减仍用原始分钟(与tau_eff=5000分钟校准一致)
        ti_exp = self.ti_proj(torch.log1p(ti)).unsqueeze(1).expand(-1, N, -1)
        mode_1d = mode.view(-1, 1).float() if mode is not None else torch.zeros(Z_k.shape[0], 1, device=Z_k.device)
        mode_exp = self.mode_proj(mode_1d).unsqueeze(1).expand(-1, N, -1)
        zk_exp = self.zk_proj(Z_k)                 # [B, N, K]→[B, N, D]
        ze_exp = self.z_proj(Z_e.unsqueeze(-1))   # [B, N, D]
        # 所有D维向量相加, 避免标量与embedding量纲失衡
        inp = zk_exp + ze_exp + s_exp + r_exp + q_exp + ti_exp + mode_exp
        h = self.enc_shared(inp)  # [B, N, D]

        # ---- 1层GAT: 可学习的KC注意力 ----
        # 自注意力分数: query @ key
        Q = self.gat_query(h)  # [B, N, D]
        K = self.gat_key(h)    # [B, N, D]
        attn = torch.bmm(Q, K.transpose(1, 2)) / (self.D ** 0.5)  # [B, N, N]
        # 只保留有边连接的注意力, 无边位置设为 -inf
        attn = attn.masked_fill(self.kc_adj.unsqueeze(0) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)  # [B, N, N], 每行和为1
        attn = self.dropout(attn)

        # 注意力聚合
        h_attn = torch.bmm(attn, h)  # [B, N, D]
        h1 = F.relu(self.gat_proj(h_attn))
        h1 = self.dropout(h1)
        h = self.gat_norm(h + h1)

        return (self.k_mu(h), self.k_lv(h),
                self.e_mu(h), self.e_lv(h))

    # ================================================================
    # 前向
    # ================================================================

    def forward(self, q_id, c_id, score, err_reason, time_intervals, global_debug=None):
        q_id = q_id.long().to(self.device)
        c_id = c_id.long().to(self.device)
        raw_score = score.float().to(self.device)
        score_idx = (score * 100).long().to(self.device)
        err_reason = err_reason.to(self.device)
        time_intervals = time_intervals.to(self.device)

        B, S = q_id.shape
        N = self.num_kcs

        # ---- KC mask ----
        kc_mask = torch.zeros(B, S, N, device=self.device)
        c_id_clamped = c_id.clamp(0, N - 1)
        kc_mask.scatter_(2, c_id_clamped, 1.0)
        has_kc = (c_id != 0).any(dim=2, keepdim=True).float()
        kc_mask = kc_mask * has_kc

        # ---- 初始状态 ----
        Z_k = torch.sigmoid(self.init_y).unsqueeze(0).expand(B, -1, -1).clone()  # [B, N, K]
        Z_e = self.init_e.unsqueeze(0).expand(B, -1).clone()                      # [B, N]
        attempt_count = torch.zeros(B, N, device=self.device)  # per-KC练习次数

        # ---- 模式检测 ----
        running_mean = torch.zeros(B, device=self.device)
        running_std = torch.ones(B, device=self.device) * 0.1
        mode_th = torch.sigmoid(self.mode_threshold) * 2.0 + 0.5
        reason_w = torch.sigmoid(self.reason_weight) * 0.5

        predict_score = torch.zeros(B, S, device=self.device)
        kl_k_list, kl_e_list = [], []

        for i in range(S):
            # ---- 预测当前步 ----
            effective = Z_k + torch.tanh(self.w_exec) * torch.tanh(Z_e * 0.1).unsqueeze(-1)
            kc_a = self._weighted_kc_ability(effective, kc_mask[:, i, :], q_id[:, i])
            diff = self.problem_difficulty[q_id[:, i]]
            disc = torch.abs(self.problem_discrimination[q_id[:, i]]) + 1e-6
            # 温度缩放: temp>1 → 概率锐化(提ACC); 单调变换不影响AUC排序
            temperature = torch.exp(self.log_temperature)
            predict_score[:, i] = torch.sigmoid(temperature * disc * (kc_a - diff)).clamp(1e-7, 1 - 1e-7)

            if i == S - 1:
                break

            # ---- 模式检测 ----
            if i > 0:
                sd = raw_score[:, i] - raw_score[:, i - 1]
                rd = (err_reason[:, i, :5] - err_reason[:, i - 1, :5]).norm(dim=1)
                sig = sd + reason_w * rd
                running_mean = self.momentum * running_mean + (1 - self.momentum) * sig
                running_std = self.momentum * running_std + (1 - self.momentum) * (sig - running_mean).abs()
                is_learning = ((sig - running_mean).abs() > mode_th * running_std).float()
            else:
                is_learning = torch.ones(B, device=self.device)
            is_debug = 1.0 - is_learning

            # ---- 提取 obs_i ----
            q_emb = F.softplus(self.problem_emb(q_id[:, i]))
            s_emb = F.softplus(self.score_emb(score_idx[:, i]))
            r_emb = F.relu(self.reason_mlp(err_reason[:, i, :5]))
            ti = time_intervals[:, i:i + 1]

            # ---- 编码器 (输入包含模式) ----
            mu_y, lv_y, mu_e, lv_e = self._encode(Z_k, Z_e, s_emb, r_emb, q_emb, ti, mode=is_learning)

            # ---- 先验 (模式条件) ----
            tau_k = torch.exp(is_learning.view(B, 1, 1) * self.log_tau_k +
                              is_debug.view(B, 1, 1) * self.log_tau_k * 0.3)
            tau_e = torch.exp(is_learning.view(B, 1, 1) * self.log_tau_e +
                              is_debug.view(B, 1, 1) * self.log_tau_e * 1.5)
            gamma = torch.sigmoid(self.log_gamma)

            prior_mu_y = Z_k  # [B, N, K]
            prior_lv_y = 2 * torch.log(tau_k + 1e-8)
            prior_mu_e = gamma * Z_e.unsqueeze(-1)
            prior_lv_e = 2 * torch.log(tau_e + 1e-8)

            # ============================================================
            # KL: 每步都算 + 线性warmup (前轻后重)
            # ============================================================
            cur_mask = kc_mask[:, i, :].unsqueeze(-1)
            kl_k = self._kl_gaussian(mu_y, lv_y, prior_mu_y, prior_lv_y)
            kl_e = self._kl_gaussian(mu_e, lv_e, prior_mu_e, prior_lv_e)
            # warmup: 前20%的步KL权重从0线性升到1
            warmup_ratio = min(1.0, (i + 1) / max(1, int((S - 1) * 0.2)))
            # KL按K维归一化: 向量化后kl_k是[B,N,K], 若不除以K则KL压力是标量版K=8倍
            # → 正则过强导致ACC下降; 除以K保持与标量版同等的KL压力
            kl_k_step = (kl_k * cur_mask * warmup_ratio).sum(dim=(1, 2)) / self.K
            kl_e_step = (kl_e * cur_mask * warmup_ratio).sum(dim=(1, 2))

            # ---- 重参数化采样 -> 下一状态 ----
            Z_k_new = torch.sigmoid(torch.nan_to_num(self._reparameterize(mu_y, lv_y), nan=0.0))  # [B, N, K]
            Z_e = torch.nan_to_num(self._reparameterize(mu_e, lv_e).squeeze(-1), nan=0.0)

            # ---- 吸收门控 (分数×模式) ----
            # 分数越高→吸收越多, 调试模式→吸收打折
            score_abs = torch.sigmoid(torch.exp(self.log_score_abs) * (raw_score[:, i] - 0.5))  # [B]
            mode_scale = is_learning + is_debug * torch.sigmoid(self.debug_abs_scale)           # [B]
            absorption = score_abs * mode_scale  # [B]
            # 新旧状态按吸收率混合: 高吸收→全盘接受新状态
            cur_mask_2d = cur_mask.squeeze(-1)  # [B,N]
            gate = (absorption.unsqueeze(1) * cur_mask_2d).unsqueeze(-1)  # [B,N,1]
            Z_k = Z_k * (1 - gate) + Z_k_new * gate

            # ---- 数值稳定: 清除NaN/Inf ----
            Z_k = torch.nan_to_num(Z_k, nan=0.5, posinf=1.0, neginf=0.0)
            Z_e = torch.nan_to_num(Z_e, nan=0.0)

            # ---- 时间效应 (遗忘: 练习次数调制的幂律) ----
            tau_0 = torch.exp(self.log_tau_0)                        # 基线半衰期
            kappa = torch.sigmoid(self.log_kappa)                     # 练习增强系数
            tau_eff = tau_0 * (1 + attempt_count) ** kappa            # 练习越多→半衰期越长
            retention = torch.exp(-ti / (tau_eff + 1e-6)).unsqueeze(-1)  # [B, N, 1]
            init_k = torch.sigmoid(self.init_y).unsqueeze(0).expand(B, -1, -1)  # [B, N, K]
            Z_k = Z_k * retention + init_k * (1 - retention)

            e_decay = 0.2
            Z_e = Z_e * (1 - e_decay)

            # ---- 更新练习计数器 ----
            cur_mask = kc_mask[:, i, :]
            attempt_count = attempt_count + cur_mask

            # ---- 记录KL ----
            kl_k_list.append(kl_k_step)
            kl_e_list.append(kl_e_step)

        # ---- 整理 ----
        kl_k_all = torch.stack(kl_k_list, dim=1) if kl_k_list else torch.zeros(B, 0, device=self.device)
        kl_e_all = torch.stack(kl_e_list, dim=1) if kl_e_list else torch.zeros(B, 0, device=self.device)
        kl_step = kl_k_all + kl_e_all  # [B, S-1], 只有最后一步非零

        extra = {'kl_k': kl_k_all, 'kl_e': kl_e_all}
        return predict_score, kl_step, extra

    def set_epoch(self, epoch):
        """设置当前epoch, 用于KL退火"""
        self.current_epoch = epoch

    def compute_loss(self, predict_score, target_score, kl_step, mask=None):
        """
        BCE + beta * annealing_ratio * mean(KL)
        mask: [B, S-1], 1=有效位置, 0=padding, 不传则全1
        """
        target = target_score.float().to(self.device)
        pred_safe = torch.nan_to_num(predict_score, nan=0.5).clamp(1e-7, 1 - 1e-7)

        if mask is not None:
            mask = mask.float().to(self.device)
            bce = F.binary_cross_entropy(pred_safe, target, reduction='none')
            bce = (bce * mask).sum() / (mask.sum() + 1e-8)
        else:
            bce = F.binary_cross_entropy(pred_safe, target, reduction='mean')

        # ---- epoch退火: 前20个epoch KL从0线性升到1 ----
        annealing_ratio = min(1.0, getattr(self, 'current_epoch', 0) / 20.0)
        kl_val = kl_step.mean()
        return bce + self.beta * annealing_ratio * kl_val, bce, kl_val
