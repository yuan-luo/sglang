# 线性注意力的 Token/Chunk 级前缀缓存(方向 B)— 设计文档

> 目标:为 KDA(Kimi Delta Attention)/ GDN(Gated DeltaNet)这类**线性注意力 / 状态空间(SSM)**层,做出真正能跨请求复用的细粒度前缀缓存(prefix cache),把它的复用能力追平 full-attention 的 token 级 RadixCache。
>
> 本文从原理、现状、痛点、探索过程、到落地方案逐层展开,结合 SGLang 当前 main 的代码与公式(LaTeX)。符号沿用 [`gdn_kda_mamba_relationship.md`](./gdn_kda_mamba_relationship.md) 与 [`pr27658_compact_linear_spec_cache.md`](./pr27658_compact_linear_spec_cache.md)。

---

## 0. TL;DR

- **线性注意力的前缀缓存早就有了**(`MambaRadixCache` / `UnifiedRadixCache` 的 `MambaComponent`),但复用粒度是 **chunk 级(`mamba_cache_chunk_size`,默认 = `max(FLA_CHUNK_SIZE=64, page_size)`)**,且每个复用点都要存一份**完整的 SSM 状态**。
- **真正的瓶颈不是粒度,是显存单价**:一份 SSM 状态是 $\mathcal{O}(L\cdot H\cdot d_k\cdot d_v)$,几十 MB 量级;mamba 状态池只能放下极少数复用点,远跟不上 full-attn KV 的 token 级容量。混合模型里 **mamba 侧拖了前缀复用的后腿**。
- **核心洞察**:每步状态增量是 **rank-1** 的,$S_t-\mathrm{diag}(\alpha_t)S_{t-1}=k_t u_t^\top$;一段区间的状态可由**稀疏满 checkpoint + 每 chunk 的紧凑增量** $(k, v_{\text{new}}, \alpha)$ 通过 **replay** 精确重建。
- **极其有利的现实**:`chunk_gated_delta_rule_fwd_h` 这个递推核 **GDN/KDA 共用**(只差 `USE_G` 标量 vs `USE_GK` 逐通道),而且它在 prefill 时**本就算出 per-chunk 满状态 `h` 与 `v_new`,用完即弃**。所以 checkpoint 和 delta 都是"现成的",**replay 核 ≈ 复用同一个核令 `w=0, u=v_new`**,几乎零新内核。
- **已验证**(M1 oracle,H20-3e):在 KDA 的逐通道 decay 路径上,`replay(w=0, u=v_new, init=fp32 checkpoint)` 与全量 forward **bit-exact**(`max|Δ|=0`);若 checkpoint 用 bf16 存则有可控漂移(决定"fp32 无损 vs bf16 省半"的旋钮)。
- **决策**:GDN 与 KDA 对 B 而言同核同路径、工作量对称;KDA 是主场且 243 H20-3e 上有 Kimi-Linear-48B-A3B 能端到端验证,GDN 无模型只能离线 → **KDA 先行**。

---

## 1. 背景:线性注意力为什么"天生不友好"前缀缓存

### 1.1 统一的门控 delta 递推

GDN / KDA / Mamba2 都可写成同一条**门控 delta-rule** 递推(每层每头,状态 $S_t\in\mathbb{R}^{d_k\times d_v}$):

$$
S_t \;=\; \mathrm{diag}(\alpha_t)\,S_{t-1} \;+\; k_t\,u_t^\top,
\qquad
u_t \;=\; \beta_t\big(v_t - \tilde S_t^\top k_t\big),
\qquad
\tilde S_t=\mathrm{diag}(\alpha_t)\,S_{t-1},
$$

$$
o_t \;=\; S_t^\top q_t .
$$

- **GDN**:$\alpha_t=\alpha_t\mathbf{1}$ 是**逐头标量**衰减(≈ Mamba2 / SSD)。
- **KDA**:$\alpha_t\in\mathbb{R}^{d_k}$ 是**逐通道**衰减 $\mathrm{diag}(\alpha_t)$(≈ Mamba1 的选择性),低秩 $f_a/f_b$ 投影 + 输出门控 RMSNorm。

### 1.2 与 full-attention 的本质差异

| | full-attention | 线性注意力 / SSM |
|---|---|---|
| "状态" | 每 token 一份 $(k_t,v_t)$,**token 可寻址** | 一个递推矩阵 $S_t$,**汇总了 0..t 全部 token** |
| 前缀复用 | RadixCache 树,token 级共享前缀,天然 | 不能从 $S_T$ 反推任意中间 $S_t$;**无 token 可寻址性** |
| 缓存单价 | 每 token $\mathcal{O}(d)$ | 每 checkpoint $\mathcal{O}(d_k d_v)$,**贵几个数量级** |

这就是"线性注意力做不了细粒度前缀缓存"这一刻板印象的根源——但 SGLang 实际上**已经部分解决了它**,只是粗。

---

## 2. 现状:SGLang 已有 chunk 级 mamba 前缀缓存(代码勘探)

主线已有成熟设施:`mamba_radix_cache.py`(`MambaRadixCache`)、`unified_radix_cache.py`(`MambaComponent`)、host offload 的 `hi_mamba_radix_cache.py`、`allocator/mamba.py`。混合模型里 full-attn KV 走 token 粒度,mamba 状态走"节点粒度"。

### 2.1 状态只挂在节点边界,分叉即丢

`mamba_radix_cache.py:TreeNode` 每个节点带 `mamba_value`(指向 `MambaPool` 的一个 slot)。两处关键代码决定了**粒度**:

```python
# mamba_radix_cache.py  _split_node()  (≈ L1074)
new_node.mamba_value = None  # mamba cache can not be split
```

```python
# mamba_radix_cache.py  _match_prefix_helper()  (≈ L972)
while len(key) > 0 and child_key in node.children.keys():
    child = node.children[child_key]
    if node.mamba_value is not None:          # ← 复用边界只在“有满状态”的节点推进
        best_value_len = len(value)
        best_last_node = node
    ...
```

即:**在 token 中间分叉(`_split_node`)时,父节点的 `mamba_value` 被置 `None`(成为 mamba tombstone),中间态丢弃**;前缀匹配只能复用到"上一次请求 / 上一个 chunk 存过满状态的节点边界"。

### 2.2 checkpoint 落点 = chunk 对齐 + 分叉点自适应

落点粒度由 `mamba_cache_chunk_size` 决定:

```python
# server_args.py  (≈ L7738)
@property
def mamba_cache_chunk_size(self) -> int:
    chunk_size = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)  # FLA_CHUNK_SIZE=64
    self._mamba_cache_chunk_size = max(chunk_size, self.page_size)
```

`_match_post_processor` 会算出一个 chunk 对齐的 `mamba_branching_seqlen`,调度器据此**在分叉点强制补存一个 checkpoint**(`schedule_batch.py:_force_track_h` + `MambaPool` 的 ping-pong track buffer / `enable_mamba_extra_buffer`):

```python
# mamba_radix_cache.py  _match_post_processor()  (≈ L1036)
chunk_aligned_seqlen = (sum(len(v) for v in value) // self.mamba_cache_chunk_size) * self.mamba_cache_chunk_size
mamba_branching_seqlen = chunk_aligned_seqlen if chunk_aligned_seqlen > 0 else None
```

**小结**:现状 = "chunk 对齐(~64)+ 分叉点自适应"的**满状态** checkpoint 复用。第一次在位置 $P$ 分叉、最近 checkpoint 在 $Q$,需重算 $(P-Q)$(最坏≈一个 prefill forward);但会在 $\lfloor P/64\rfloor\cdot64$ 补存 checkpoint,之后同点分叉只重算 $<64$ token。

---

## 3. 痛点:瓶颈是 checkpoint 的显存单价,不是粒度

### 3.1 一份状态有多大

`MambaPool`(`memory_pool.py:279`)的 temporal(SSM)状态:

```python
# memory_pool.py  (≈ L369)
temporal_state = torch.zeros(
    size=(num_mamba_layers, size + 1) + temporal_state_shape,   # [L, size+1, HV, d_k, d_v]
    dtype=ssm_dtype, device=device)
```

单个 slot(一份完整状态)的字节数:

$$
\text{bytes/slot} \;=\; L_{\text{lin}}\cdot H_v\cdot d_k\cdot d_v\cdot \text{sizeof(dtype)} .
$$

以 Kimi-Linear-48B-A3B(KDA)量级估:$L_{\text{lin}}\sim 35$、$H_v=32$、$d_k=d_v=128$、bf16 →

$$
35\times32\times128\times128\times2 \approx \mathbf{367\ MB/slot}.
$$

(GDN/Qwen3-Next 同量级。)对比 full-attn KV 是**每 token** $\mathcal{O}(d)$、能放下数百万 token。

### 3.2 容量不对称 → mamba 侧先饿死

混合模型里同一棵 radix 树:full-attn 侧能 token 级留住几百万 token 的多个长上下文;mamba 侧因为**每个复用点要一份满状态**,只能留极少数 checkpoint。后果:

> full-attn KV 还在缓存里,但对应的 mamba 状态已被挤掉 → 命中后**仍要从最近残存 checkpoint 重算 SSM 递推**(甚至从头)。**mamba 侧的低容量决定了整条混合前缀复用的命中率上限。**

绑定约束随**缓存前缀的多样性**(不同文档/会话数)增长,而非并发数(共享前缀的 N 个并发请求复用同一个 checkpoint)。典型受害 workload:多文档 RAG、多会话 agentic、长上下文多分叉。

### 3.3 B 的价值重定位

所以 B 的价值**不是**省那 $<64$ token 的重算,而是:

> **把"每复用点一份满状态"换成"稀疏满 checkpoint(每 $C$ token)+ 每 chunk 紧凑 delta + replay 重建"**,让 mamba 复用点的显存单价从 $\mathcal{O}(d_kd_v)$ 降到 $\mathcal{O}(d_k+d_v)$/token,从而**容量/命中率上一个台阶**,顺带得到精确 chunk 粒度的任意分叉。

---

## 4. 核心洞察:rank-1 增量 + 共享递推核

### 4.1 增量是 rank-1 的

由 §1.1 递推,

$$
\boxed{\,S_t - \mathrm{diag}(\alpha_t)\,S_{t-1} \;=\; k_t\,u_t^\top\,}
$$

一步的全部信息由三个向量 $(\alpha_t, k_t, u_t)$(尺寸 $d_k, d_k, d_v$)决定,而非整个 $d_k\times d_v$ 矩阵。把它沿一个 chunk(长度 $\mathrm{BT}=64$)聚合,就是 chunk 级递推:

$$
S_{c} \;=\; \mathrm{diag}\!\Big(\textstyle\prod_{t\in c}\alpha_t\Big)\, S_{c-1} \;+\; \sum_{t\in c} \Big(\textstyle\prod_{t<s\le \text{end}(c)}\alpha_s\Big)\,k_t\,u_t^\top .
$$

这正是 chunk 扫描(prefill)算的东西。

### 4.2 GDN / KDA 共用的 `chunk_gated_delta_rule_fwd_h`

`chunk_delta_h.py` 的核 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 是**跨 chunk 状态递推**的主体,**GDN 与 KDA 共用**,唯一差别是衰减分支:

```python
# chunk_delta_h.py 主循环(精简)  (L143–272)
for i_t in range(NT):
    tl.store(p_h, b_h, ...)                 # ← 把“进入 chunk i_t 的状态”写到 h[i_t]
    b_v = tl.load(p_v_u) - b_w @ b_h.T      # v_new = u - W·H  (chunk 内 WY 修正)
    if SAVE_NEW_VALUE: tl.store(p_v_new, b_v)
    if USE_G:                               # GDN：逐头标量
        b_h  *= exp(b_g_last)
        b_v  *= safe_exp(b_g_last - b_g)
    if USE_GK:                              # KDA：逐通道 [K]
        b_h  *= exp(b_gk_last)[None, :]
    b_h += tl.trans(tl.dot(b_k, b_v))       # h += k ⊗ v_new
```

三个事实(对 B 极其有利):

1. **per-chunk 满状态 `h[B, NT, H, V, K]` 在 prefill 本就被物化**(`h = k.new_empty(B, NT, H, V, K)`,L324;循环顶 `tl.store`)。
2. **`v_new` 已由 `SAVE_NEW_VALUE` 存出**(`chunk_gated_delta_rule_fwd_h` 返回 `(h, v_new)`)。
3. 递推就是 `b_h *= decay; b_h += k⊗v_new`。

而它们**用完即弃**:

```python
# kda.py  chunk_kda_fwd()  (L1107–1131)
h, v_new = chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=..., ...)
o = chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=Aqk, h=h, ...)   # 算输出
del Aqk, v_new, h                                            # ← checkpoint + delta 全丢
```

> 这是 B 最大的认知修正:我原以为最大风险在"改 FLA 核抽 delta"。**其实 delta 与 checkpoint 全是现成的,replay 复用现有核** → B 的重心从高风险内核移到可控的系统活。

---

## 5. 探索过程中的关键发现

### 5.1 replay 核 ≈ 复用现有核,令 $w=0,\,u=v_{\text{new}}$

观察主循环:`b_v = load(u) - b_w @ b_h.T`。若取 $w=0$,则 $b_v = u - 0 = u$。因此

$$
\text{replay}(k_g,\,w{=}0,\,u{=}v_{\text{new}},\,\alpha,\,S_0)\;\Longrightarrow\; b_v=v_{\text{new}},\quad b_h \mathrel{*}= \text{decay},\quad b_h \mathrel{+}= k_g\otimes v_{\text{new}},
$$

**精确重放**原 forward 在 $[c_0, \text{end}]$ 上对状态做的每一步。`v_new` 是 chunk 内 WY 修正后的"等效值",只要 replay 从**同一个 chunk 边界 checkpoint** 起步,它就与原 forward 完全一致——无需重算 $W,U$,也无需 `b_w @ b_h`。

> 工程含义:**replay 不必新写内核**,直接调 `chunk_gated_delta_rule_fwd_h(k=kg_stored, w=0, u=v_new_stored, gk=g_stored, initial_state=checkpoint)`;GDN/KDA 仅在 `g`(标量)/`gk`(逐通道)入参上分流。

### 5.2 精度:fp32 checkpoint 无损,bf16 有界漂移(已实测)

一个对真实系统至关重要的细节:核内运行态 `b_h` 全程 **fp32**,但 per-chunk `h` 数组按**输入 dtype(bf16)**落盘。

- 全量 forward:跨 chunk 一路 fp32,`h[c]` 存 bf16(仅供算输出,随后丢)。
- 若 B 的 checkpoint 也存 bf16,replay 从 $\text{bf16}(S_{c_0})$ 起 → 与全量(fp32 续算)差**一次 bf16 舍入**,并沿 replay 传播。
- 若 checkpoint 存 **fp32**(用 fp32 的 `initial_state` 池,`INPLACE_UPDATE` 写回即 fp32)→ **bit-exact**。

**M1 oracle 实测(H20-3e,KDA `USE_GK` 逐通道路径):**

| 配置 | $\max|\Delta|$ 状态 | 结论 |
|---|---|---|
| **fp32 checkpoint** · Part A(随机 KDA 形状,$T\in\{256,512,1024\}$,所有分叉点 $c_0$) | $0.000\mathrm{e}{+}0$ | **ALL BIT-EXACT** |
| **fp32 checkpoint** · Part B(真 `chunk_kda_fwd_intra` 驱动的真实逐通道门,$T{=}512$) | $0.000\mathrm{e}{+}0$ | **BIT-EXACT** |
| bf16 checkpoint(Part A,随机无界输入) | 随 replay 长度增长 | 漂移(方向 F 旋钮) |

> 结论:**checkpoint 存 fp32 → replay 与全量 bit-exact**。这把 §3 的"省内存"与"无损"统一了:checkpoint 用 fp32 是无损方案;若要再省一半 checkpoint 显存,可存 bf16 并以 $C$(checkpoint 间距)压住漂移——这是显存↔精度的可调旋钮。(注:真实 KDA 的状态因 $\alpha<1$ 收缩 + qk l2norm 而有界,bf16 漂移远小于 oracle 随机输入下的放大值。)

### 5.3 别忘了 conv 状态

线性注意力层的"状态"不止 SSM 矩阵 $S$,还有 **causal conv1d 的短窗口**(`short_conv_kernel_size=4` → 窗口 $W-1=3$ 个 token)。它不是 rank-1 delta,但极小。注意:进入 `chunk_delta_h` 的 $k_g$ 是 **post-conv** 的(`kda_backend.forward_extend` 先做 `causal_conv1d_fn` 再进核),所以:

- 存的 $k_g, v_{\text{new}}, \alpha$ delta 是 **conv 无关的**,replay 不需要重做 conv;
- 但"在分叉点继续 decode"还需要该点的 conv 窗口 → 像现有 mamba conv cache 一样,**每 checkpoint 存满 conv 窗口**($\mathcal{O}((W{-}1)\cdot d)$,极小)即可,不需 per-token。

---

## 6. 解决方案:系统设计

### 6.1 数据布局

在 `MambaPool` 旁新增两类资源:

```
checkpoint pool (稀疏, fp32 优先):   每 C token 一个满状态  [L, n_ckpt, Hv, d_k, d_v]
                                     + 对应 conv 窗口        [L, n_ckpt, conv_dim, W-1]
delta pool      (每 token/chunk):    kg     [L, n_tok, Hg, d_k]   ┐
                                     v_new  [L, n_tok, Hv, d_v]   ├ 紧凑增量 (~KV 量级)
                                     gk/g   [L, n_tok, Hv, d_k或1]┘
```

单位 token 的 delta 体量 $\mathcal{O}(d_k+d_v)$(KDA 的 $gk$ 是 $[d_k]$;GDN 的 $g$ 是标量)——与一个 token 的 KV 同量级,而线性层本来不存 KV。所以"给前缀加 delta"≈ 把这部分前缀缓存单价翻倍,**便宜**。checkpoint 显存被 $C$ 摊薄($C=m\cdot\mathrm{BT}$)。

$$
\text{B 的 mamba 缓存}\;\approx\; \underbrace{\frac{n_{\text{tok}}}{C}\cdot d_kd_v}_{\text{稀疏 checkpoint}} \;+\; \underbrace{n_{\text{tok}}\cdot(d_k+d_v)}_{\text{delta}} \;\;\ll\;\; \underbrace{n_{\text{reuse}}\cdot d_kd_v}_{\text{现状:每复用点一份满态}} .
$$

### 6.2 抽出 delta 与 checkpoint(改 1 行 + 持久化)

`chunk_kda_fwd` / GDN 对应路径里,`h` 与 `v_new` 现在被 `del`。改为:

1. `chunk_gated_delta_rule_fwd_h` 已支持 `save_new_value` → 拿到 `v_new`;
2. 把 `h` 的**稀疏子集**(每 $C/\mathrm{BT}$ 个 chunk 取一个)+ 该位置 conv 窗口写入 checkpoint pool;
3. 把 `kg, v_new, gk` 按 token 写入 delta pool(可只在"该前缀将被缓存"时落,避免 decode 常驻开销)。

> 关键:`kg/v_new/gk` 与 token 一一对应,生命周期跟随 full-attn KV(同 page、同淘汰节奏),天然挂到现有 `req_to_token` / 分页分配器上。

### 6.3 树集成:`MambaRadixCache` 节点从"满态"放宽到"(checkpoint, delta 区间)"

- 节点语义:`mamba_value` → `(ckpt_ref, delta_range=[q0, q1])`,表示"状态 = 从 `ckpt_ref` replay `delta[q0:q1]`"。
- `match_prefix` 命中非 checkpoint 边界(chunk 对齐)时:找**最近 $\le$ 匹配点的 checkpoint**,`replay` 到精确边界,得到该请求的初始 $S$(COW 进请求的状态 slot)。
- `_split_node` **不再丢中间态**:父节点记 `(同一 ckpt_ref, 截断的 delta 区间)`,子节点记其后续区间。中间态可重建 → **真正的 chunk 级任意分叉**。
- 兼容现有 `mamba_branching_seqlen` / extra_buffer:由"自适应补满态"退化为"delta 始终在、任意 chunk 边界可重建",可去掉 `_force_track_h` 的对齐特判。

### 6.4 replay 的并行度(方向 C,B 的内核使能器)

draft=4 的串行 replay 无所谓;但 B 里从 checkpoint replay $C/\mathrm{BT}=$ 数个~数十个 chunk,且在 **match 关键路径**上(命中即需同步重建)。直接复用 `chunk_gated_delta_rule_fwd_h` 的分块扫描即可并行 replay 整个区间(它本就是 chunk 并行核),无需退化成长串行循环。

### 6.5 淘汰 / 锁的级联

- delta 区间 $[q_0,q_1]$ 的有效性依赖其 `ckpt_ref` 存活 → 淘汰 checkpoint 必须让依赖它的复用点失效,或把某个下游点**提升为新 checkpoint**(replay 一次落盘)。
- 接入 `UnifiedRadixCache` 的 `MambaComponent` cascade:checkpoint 与 delta 作为同一 component 的两层资源,优先淘汰 delta、保 checkpoint;或反之,按命中模型调。
- 这是系统活里最易出 bug 的地方(现有 tombstone / cascade 逻辑要扩),需重点测试。

### 6.6 GDN vs KDA 参数化

唯一真实差异:衰减布局。replay / 抽取 / 存储一律参数化 `USE_G`(GDN 标量,delta 的 `g` 存 1 标量/step)vs `USE_GK`(KDA 逐通道,存 $[d_k]$)。核已两条分支都在 → **同一套代码两模型通吃**。

---

## 7. 为什么 KDA 先行

| 维度 | GDN | KDA |
|---|---|---|
| B 所需递推核 | `chunk_gated_delta_rule_fwd_h`(`USE_G`) | 同核(`USE_GK`)——**同一文件同函数** |
| delta/checkpoint 可得性 | 现成(同上) | 现成(同上) |
| 已有 spec-decode 基建 | 有 `intermediate_ssm`+`target_verify` | 无(只与**方向 A** 相关,**与 B 无关**) |
| 主场 | — | ✅ 用户主场 |
| 243 H20-3e 端到端验证 | ✅ `Qwen3.5-35B-A3B`(`qwen3_5_moe`,GDN-hybrid:`layer_types` 多为 `linear_attention`+每 4 层 `full_attention`;`linear_num_key_heads=16`/`value_heads=32` GQA) | ✅ Kimi-Linear-48B-A3B(KimiLinear,KDA-hybrid,TP4) |

> 修正:**GDN = Qwen3.5 系列**(Qwen3-Next 谱系)。已下 `Qwen/Qwen3.5-35B-A3B` 到 243 → GDN 也能端到端验证(不再只能离线)。

对 **B** 而言两者对称(同核同路径);KDA 是主场且已 bit-exact 验证(M1),两模型现都有 243 端到端环境 → **KDA 先做(已起步),GDN 用同一套参数化(`USE_G` 标量 vs `USE_GK` 逐通道)顺带通用化**。注意 GDN 这里 `Hg=16 ≠ H=32`(GQA),`chunk_delta_h` 的 `Hg` 参数已支持。

---

## 8. 分期、验证与风险

### M0 — 量化瓶颈 ✅(已测,Kimi-Linear-48B-A3B TP4 @ H20-3e)
纯客户端 probe(无需改服务端):warm K 个 distinct ~P-token 前缀 → probe 每个,读 `meta_info.cached_tokens`(= **mamba 受限**复用)。KV 容量 `max_total_num_tokens=4,523,301`,远超 K·P → KV 不淘汰;于是 `cached/prefix_len` 随 K 的塌缩就是 mamba 池饱和的直接证据。harness:`tools/kda_b/m0_sweep.py`。

**结果(P=1000 token 前缀):**

| K(distinct 前缀) | reuse_frac | 严重淘汰(reuse<0.5) | 解读 |
|---|---|---|---|
| ≤512 | **0.940** | 0% | chunk 对齐天花板(~50/1000 token 损失);mamba 全留 |
| 1024 | 0.919 | 2.1% | 拐点起 |
| 1536 | 0.623 | 33.6% | 塌缩中 |
| 2048 | **0.055** | 94.2% | 近乎全失:**缓存了 ~2.05M token(<4.5M KV),mamba 复用仅 5.5%** |
| 3072 | 0.005 | 99.5% | 全失 |

**核心数字:mamba 池只存 ~1200 个 checkpoint(每个缓存请求 1 个满状态,与长度无关),而 KV 容 ~4500 个 1K 前缀 → mamba 在 ~27% KV 容量就饱和。** 超过 ~1200 条 distinct 前缀后,SSM 复用塌向 0,而 token KV 仍在缓存 → 每次重访都白算 SSM 全程递推。**这就是 B 要消除的浪费,实测坐实。**

**长度依赖(P=4000,reuse 天花板 0.988):** K≤1024 不塌(2.9% severe,刚到拐点),与 P=1000 同 K 行为一致 → 塌缩由**前缀条数 K**(≈mamba slot 数)决定,**与前缀长度无关**。推论:
- **短/中前缀(≲4K):KV 容 ~4500、mamba 仅 ~1200 → mamba 早饱和 ~3.7×,B 大赢。**
- **长前缀(≳4K):KV 与 mamba 同时饱和甚至 KV 先饱和 → B 收益小。**

→ **B 的目标场景:多文档 RAG / 多短会话 / agentic 多 distinct 短上下文(高基数前缀)**,而非少数超长上下文。`mamba_track_interval=256`、`mamba_scheduler_strategy=extra_buffer`、`mamba_backend=triton`。
> gap 说明:M0 跑在 243 的 theta2 repo;已 diff 确认 `mamba_radix_cache.py` + `MambaPool` 与 github4 当前 main **逻辑一致**(差异仅 cosmetic 引号注解 / 一个 `mamba_extra_buffer_no_aligned` flag 门控 / 无关的 FP4-KV 命名),M0 结论可迁移。

### M1 — replay 核 + delta 抽取的离线 bit-exact 验证 ✅(已通过)
- harness:`tools/kda_b/replay_oracle.py`。
- 已验证(两路都过):
  - **Part A**(随机 KDA 形状,模型无关地证明 `chunk_delta_h` replay 恒等式):`replay(w=0, u=v_new, init=fp32 ckpt)` 与全量 forward **bit-exact**($\max|\Delta|=0$,$T\le1024$,任意分叉点 $c_0$);
  - **Part B**(真 `chunk_kda_fwd_intra` 驱动的真实逐通道门):同样 **bit-exact**。
- 旁证:bf16 checkpoint 会按 replay 长度漂移 → checkpoint 存 fp32 = 无损。

### M2 — 系统集成(主体工程量),分子步:
- **M2.1 ✅ `LinearDeltaPool` + varlen 抽取 + replay round-trip(bit-exact)** — `tools/kda_b/m2_pool_replay.py`。flat per-token `(kg,v_new,gk)` + 稀疏 fp32 checkpoint;`cu_seqlens` 多序列;从最近 checkpoint 重建任意 chunk 边界。Part A(随机,3 种 varlen×C=1/2/4)+ Part B(真 KDA)全 `max|Δ|=0`。**算法地基已稳。**
- **M2.2 活体抽取**:在 KDA backend `forward_extend`/`chunk_kda_fwd` 处把 `(kg,v_new,gk)` + 稀疏 fp32 checkpoint 落进 `LinearDeltaPool`(production:让 `chunk_delta_h` 在稀疏 chunk 边界多输出一个 fp32 state,免段重算)。验证:真 Kimi-Linear-48B-A3B prefill 抽出的 delta 重建 == 活体末态(fp32 ssm 池下 bit-exact;bf16 池下 in-tolerance)。需改后端 + 起服务。
- **M2.3 radix 接入**(进行中):
  - ✅ **地基模块 `LinearDeltaPool` 已落 github4**(`python/sglang/srt/mem_cache/linear_delta_pool.py`):存储 `kg/v_new/decay`(每 token)+ 稀疏 fp32 checkpoint(每 `ckpt_interval`),`store_deltas/store_checkpoint/reconstruct` API,GDN(`per_channel_decay=False`,USE_G)/KDA(True,USE_GK)通吃;replay 复用 `chunk_gated_delta_rule_fwd_h(w=0,u=v_new)`。生命周期(分配/淘汰/COW)留给 radix,镜像 `MambaPool`+`MambaSlotAllocator` 设计。**公开 API bit-exact 验证**(varlen×C=1/2/4×真KDA,`max|Δ|=0`,harness `tools/kda_b/m2_module_test.py`)。
  - 待做:`MambaRadixCache` 节点 `mamba_value → (ckpt_ref, delta_range)`;`match_prefix` 命中非 checkpoint 边界 → `reconstruct` 重建 → COW 进请求态;`_split_node` 用 delta 区间保态(不再 `mamba_value=None`);淘汰级联(delta 依赖 ckpt);`HybridReqToTokenPool._init_mamba_pool` 旁挂 `LinearDeltaPool` + 两个 allocator(delta-token 分页 + ckpt-slot)。
- **M2.4 端到端**:Kimi-Linear-48B-A3B(KDA)+ Qwen3.5-35B-A3B(GDN)正确性(GSM8K)+ 命中率/吞吐(复跑 M0 的高基数前缀 workload,看 reuse_frac 从塌缩恢复)。$C$ 调参(显存↔精度↔replay 长度)。
> 在 github4 上实现;243 上验证(KDA Kimi-Linear-48B-A3B / GDN Qwen3.5)。M2.1 已用 theta2 跑通(核一致)。

### 风险与旋钮
- **F. 长 replay 数值漂移**:fp32 checkpoint 无损;bf16 省半但需 $C$ 压漂移。$C$ = 显存↔精度↔replay 长度旋钮。
- **淘汰级联一致性**:delta 依赖 checkpoint,易漏删/悬挂引用。
- **match 路径延迟**:replay 在关键路径,靠 §6.4 并行 + 合理 $C$ 控制。
- **收益 workload 依赖**:分叉/共享前缀/多上下文收益大,单流吞吐收益小 → 用 M0 先框定目标场景。

---

## 9. 关键文件与符号

### 代码索引(当前 main)
| 文件 | 作用 |
|---|---|
| `mem_cache/mamba_radix_cache.py` | mamba 前缀树;`_split_node`(丢态)、`_match_prefix_helper`(复用边界)、`mamba_branching_seqlen` |
| `mem_cache/unified_radix_cache.py` + `unified_cache_components/mamba_component.py` | 统一树 + `MambaComponent`(COW、cascade 淘汰、host offload) |
| `mem_cache/memory_pool.py:MambaPool` | `State(conv, temporal)` / `SpeculativeState(intermediate_ssm)`;状态 shape 与单价 |
| `attention/fla/chunk_delta_h.py` | **共享递推核** `chunk_gated_delta_rule_fwd_h`(`USE_G`/`USE_GK`,`h`、`v_new`) |
| `attention/fla/kda.py:chunk_kda_fwd` | KDA chunk prefill;L1107 调上核、L1131 `del h, v_new` |
| `attention/linear/kda_backend.py` | KDA backend(conv→核);`forward_extend` 无 `intermediate_ssm`(无 spec-decode) |
| `attention/linear/gdn_backend.py` | GDN backend;有 `target_verify`+`intermediate_ssm`(方向 A,与 B 无关) |
| `server_args.py:mamba_cache_chunk_size` | 复用粒度 = `max(FLA_CHUNK_SIZE=64, page_size)` |
| `tools/kda_b/replay_oracle.py` | M1 bit-exact 验证 harness |

### 符号
$S_t$ 状态矩阵 $d_k\times d_v$;$\alpha_t$ 衰减(GDN 标量 / KDA $\in\mathbb{R}^{d_k}$);$k_t,v_t,q_t$ key/value/query;$\beta_t$ delta 步长;$u_t=\beta_t(v_t-\tilde S_t^\top k_t)$;$v_{\text{new}}$ chunk 内 WY 修正后的等效值(核内 `b_v`);$\mathrm{BT}=64$ chunk 长;$C$ checkpoint 间距;$H_v,d_k,d_v,L_{\text{lin}}$ 头数 / k 维 / v 维 / 线性层数。

---

## 10. 下一步

1. ✅ M1 已通过(Part A + 真 KDA Part B 均 bit-exact)。
2. M0 量化(Kimi-Linear-48B-A3B,造分叉/多上下文负载)定收益,框定目标场景。
3. M2 起步:先做 **delta/checkpoint 抽取落盘 + 单序列 replay 重建在 serve 路径里复现 bit-exact**,再接 `MambaRadixCache` 节点语义($(\text{ckpt},\text{delta range})$)、match 时 replay/COW、与淘汰级联。
