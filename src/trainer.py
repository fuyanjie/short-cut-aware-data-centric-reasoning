"""Training loops for all methods.

Supports both small-scale (local) and large-scale (server) configurations
via config.py profile system.
"""
import copy
import torch
import torch.nn.functional as F
from collections import Counter
from src.config import Config as C, PROFILE
from src.data import get_dataloader
from src.methods import (
    masked_ce_loss, compute_validation_gradient, compute_sample_gradients,
    compute_sample_gradients_batched, compute_shortcut_score,
    compute_shortcut_scores_batched, compute_sample_weight,
    apply_gradient_surgery, get_grad_vector, set_grad_vector
)


def train_standard(model, dataset, epochs=None, device=C.device, verbose=True, cfg=None):
    """Standard supervised fine-tuning (Baseline a).

    cfg: optional dict with batch_size, lr, weight_decay, epochs overrides.
    """
    _c = cfg or {}
    epochs = epochs if epochs is not None else _c.get('epochs', C.epochs)
    bs = _c.get('batch_size', C.batch_size)
    lr = _c.get('lr', C.lr)
    wd = _c.get('weight_decay', C.weight_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = get_dataloader(dataset['train'], batch_size=bs, shuffle=True)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = masked_ce_loss(logits, target_ids, loss_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    return model


def train_data_filtering(model, dataset, epochs=None, device=C.device, verbose=True, cfg=None):
    """Data Filtering baseline (Baseline c).

    Warm up for a few epochs, identify high-confidence samples as potential shortcuts,
    filter them out, then retrain.
    """
    _c = cfg or {}
    epochs = epochs if epochs is not None else _c.get('epochs', C.epochs)
    bs = _c.get('batch_size', C.batch_size)
    lr = _c.get('lr', C.lr)
    wd = _c.get('weight_decay', C.weight_decay)
    df_warmup = _c.get('df_warmup_epochs', C.df_warmup_epochs)
    df_thresh = _c.get('df_confidence_threshold', C.df_confidence_threshold)

    # Phase 1: Warmup training
    warmup_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(warmup_model.parameters(), lr=lr, weight_decay=wd)
    train_loader = get_dataloader(dataset['train'], shuffle=True, batch_size=bs)

    warmup_model.train()
    for epoch in range(df_warmup):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            optimizer.zero_grad()
            logits = warmup_model(input_ids)
            loss = masked_ce_loss(logits, target_ids, loss_mask)
            loss.backward()
            optimizer.step()

    # Phase 2: Identify suspicious samples (high confidence = likely shortcut)
    warmup_model.eval()
    all_confs = []
    eval_loader = get_dataloader(dataset['train'], shuffle=False, batch_size=bs)
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            answer_mask = batch['answer_mask'].to(device)

            logits = warmup_model(input_ids)
            probs = F.softmax(logits, dim=-1)
            B, T, V = probs.shape
            target_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

            for i in range(B):
                am_i = answer_mask[i]
                tp_i = target_probs[i]
                conf = (tp_i * am_i).sum() / am_i.sum().clamp(min=1)
                all_confs.append(conf.item())

    # Keep low-confidence samples
    keep_indices = [i for i, c in enumerate(all_confs) if c < df_thresh]

    # Ensure at least 30% of samples are kept
    if len(keep_indices) < int(0.3 * len(dataset['train'])):
        keep_n = int(0.3 * len(dataset['train']))
        sorted_idx = sorted(range(len(all_confs)), key=lambda i: all_confs[i])
        keep_indices = sorted_idx[:keep_n]

    filtered_samples = [dataset['train'].samples[i] for i in keep_indices]
    if verbose:
        print(f"  Data Filtering: kept {len(filtered_samples)}/{len(dataset['train'])} samples")

    from src.data import ReasoningDataset
    pad_id = getattr(dataset['train'], 'pad_id', C.PAD)
    filtered_dataset = {**dataset, 'train': ReasoningDataset(filtered_samples, pad_id=pad_id)}
    return train_standard(model, filtered_dataset, epochs=epochs, device=device, verbose=verbose, cfg=cfg)


def train_focal_loss(model, dataset, epochs=None, device=C.device, verbose=True, cfg=None):
    """Focal Loss baseline.

    Same loop as train_standard but replaces CE with focal loss:
      FL(p_t) = (1 - p_t)^gamma * CE(p_t)
    """
    _c = cfg or {}
    epochs = epochs if epochs is not None else _c.get('epochs', C.epochs)
    bs = _c.get('batch_size', C.batch_size)
    lr = _c.get('lr', C.lr)
    wd = _c.get('weight_decay', C.weight_decay)
    gamma = _c.get('focal_gamma', C.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = get_dataloader(dataset['train'], batch_size=bs, shuffle=True)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            B, T, V = logits.shape
            ce_per_token = F.cross_entropy(
                logits.reshape(-1, V), target_ids.reshape(-1), reduction='none'
            ).reshape(B, T)

            # Focal modulation: (1 - p_t)^gamma
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                p_t = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            focal_weight = (1.0 - p_t) ** gamma

            focal_loss = focal_weight * ce_per_token * loss_mask
            loss = focal_loss.sum() / loss_mask.sum().clamp(min=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    return model


def train_jtt(model, dataset, epochs=None, device=C.device, verbose=True, cfg=None):
    """JTT (Just Train Twice) baseline.

    Phase 1: Warmup training on a copy of the model.
    Phase 2: Identify misclassified samples (answer token mismatch).
    Phase 3: Upsample those samples.
    Phase 4: Retrain original model on upsampled dataset.
    """
    _c = cfg or {}
    epochs = epochs if epochs is not None else _c.get('epochs', C.epochs)
    bs = _c.get('batch_size', C.batch_size)
    warmup_epochs = _c.get('jtt_warmup_epochs', C.jtt_warmup_epochs)
    upweight = _c.get('jtt_upweight_factor', C.jtt_upweight_factor)

    # Phase 1: Warmup on a copy
    if verbose:
        print(f"  JTT Phase 1: Warmup ({warmup_epochs} epochs)")
    warmup_model = copy.deepcopy(model)
    train_standard(warmup_model, dataset, epochs=warmup_epochs, device=device, verbose=False, cfg=cfg)

    # Phase 2: Identify misclassified samples
    warmup_model.eval()
    eval_loader = get_dataloader(dataset['train'], shuffle=False, batch_size=bs)
    error_flags = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            answer_mask = batch['answer_mask'].to(device)

            logits = warmup_model(input_ids)
            preds = logits.argmax(dim=-1)  # (B, T)

            for i in range(input_ids.size(0)):
                am = answer_mask[i].bool()
                if am.any():
                    is_error = (preds[i][am] != target_ids[i][am]).any().item()
                else:
                    is_error = False
                error_flags.append(is_error)

    del warmup_model
    n_errors = sum(error_flags)
    if verbose:
        print(f"  JTT Phase 2: {n_errors}/{len(error_flags)} error samples identified")

    # Phase 3: Upsample error samples
    original_samples = dataset['train'].samples
    upsampled = []
    for i, sample in enumerate(original_samples):
        upsampled.append(sample)
        if i < len(error_flags) and error_flags[i]:
            for _ in range(upweight - 1):
                upsampled.append(sample)

    if verbose:
        print(f"  JTT Phase 3: Upsampled {len(original_samples)} -> {len(upsampled)} samples")

    from src.data import ReasoningDataset
    pad_id = getattr(dataset['train'], 'pad_id', C.PAD)
    upsampled_dataset = {**dataset, 'train': ReasoningDataset(upsampled, pad_id=pad_id)}

    # Phase 4: Retrain
    if verbose:
        print(f"  JTT Phase 4: Retraining ({epochs} epochs)")
    return train_standard(model, upsampled_dataset, epochs=epochs, device=device, verbose=verbose, cfg=cfg)


def train_group_dro(model, dataset, epochs=None, device=C.device, verbose=True, cfg=None):
    """Group DRO baseline.

    Uses is_shortcut as group label. Maintains group weights q that are
    updated via exponentiated gradient to upweight the worst-performing group.
    """
    _c = cfg or {}
    epochs = epochs if epochs is not None else _c.get('epochs', C.epochs)
    bs = _c.get('batch_size', C.batch_size)
    lr = _c.get('lr', C.lr)
    wd = _c.get('weight_decay', C.weight_decay)
    eta = _c.get('gdro_eta', C.gdro_eta)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = get_dataloader(dataset['train'], batch_size=bs, shuffle=True)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    # Group weights: [non-shortcut, shortcut]
    q = torch.tensor([0.5, 0.5], device=device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            is_shortcut = batch['is_shortcut'].to(device)  # (B,)

            optimizer.zero_grad()
            logits = model(input_ids)

            B, T, V = logits.shape
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V), target_ids.reshape(-1), reduction='none'
            ).reshape(B, T)
            per_sample_loss = (loss_per_token * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)

            # Compute per-group losses
            group_losses = torch.zeros(2, device=device)
            for g in range(2):
                mask_g = (is_shortcut == g).float()
                if mask_g.sum() > 0:
                    group_losses[g] = (per_sample_loss * mask_g).sum() / mask_g.sum()

            # Update group weights
            q = q * torch.exp(eta * group_losses.detach())
            q = q / q.sum()

            loss = (q * group_losses).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}, "
                  f"q=[{q[0]:.3f}, {q[1]:.3f}]")

    return model


def _compute_sample_scores(model, dataset, device=C.device, cfg=None):
    """Compute ShortcutScores for training samples (one-time).

    Uses batched computation on server profile for efficiency.

    Returns list of scores, collected_data dict, and validation gradient g_V.
    """
    _c = cfg or {}
    max_samples = _c.get('score_max_samples', C.score_max_samples)
    score_bs = _c.get('score_batch_size', C.score_batch_size)
    bs = _c.get('batch_size', C.batch_size)

    val_loader = get_dataloader(dataset['val'], shuffle=False, batch_size=bs)
    g_V = compute_validation_gradient(model, val_loader, device)

    scores_all = []
    collected_data = {'scores': [], 'is_shortcut': [], 'alignments': [], 'concentrations': []}

    n_to_score = min(max_samples, len(dataset['train']))

    model.train()

    if score_bs > 1:
        # --- Batched per-sample gradient computation (server mode) ---
        score_loader = get_dataloader(dataset['train'], shuffle=False, batch_size=score_bs)
        n_scored = 0
        for batch in score_loader:
            if n_scored >= n_to_score:
                break
            actual_bs = batch['input_ids'].size(0)
            remaining = n_to_score - n_scored
            if actual_bs > remaining:
                # Trim batch
                batch = {k: v[:remaining] for k, v in batch.items()}
                actual_bs = remaining

            g_fulls, g_anss, g_reasons = compute_sample_gradients_batched(
                model, batch, device)

            batch_scores, _, _, batch_A, batch_R = compute_shortcut_scores_batched(
                g_fulls, g_anss, g_reasons, g_V)

            is_sc_list = batch['is_shortcut'].tolist()
            for i in range(len(batch_scores)):
                scores_all.append(batch_scores[i])
                collected_data['scores'].append(batch_scores[i])
                collected_data['is_shortcut'].append(is_sc_list[i])
                collected_data['alignments'].append(batch_A[i])
                collected_data['concentrations'].append(batch_R[i])

            n_scored += len(batch_scores)

            # Free GPU memory periodically
            if device == 'cuda':
                del g_fulls, g_anss, g_reasons
                torch.cuda.empty_cache()
    else:
        # --- Single-sample gradient computation (local mode) ---
        train_loader = get_dataloader(dataset['train'], shuffle=False, batch_size=1)
        for i, batch in enumerate(train_loader):
            if i >= n_to_score:
                break
            g_full, g_ans, g_reason = compute_sample_gradients(
                model, batch['input_ids'][0], batch['target_ids'][0],
                batch['loss_mask'][0], batch['answer_mask'][0],
                batch['reasoning_mask'][0], device
            )
            S, B_val, C_val, A_val, R_val = compute_shortcut_score(
                g_full, g_ans, g_reason, g_V)
            scores_all.append(S)
            collected_data['scores'].append(S)
            collected_data['is_shortcut'].append(batch['is_shortcut'][0].item())
            collected_data['alignments'].append(A_val)
            collected_data['concentrations'].append(R_val)

    # Extend with average score for remaining unscored samples
    avg_score = sum(scores_all) / max(len(scores_all), 1)
    all_scores = scores_all + [avg_score] * (len(dataset['train']) - len(scores_all))
    return all_scores, collected_data, g_V


def train_our_method(model, dataset, use_reweighting=True, use_gradient_surgery=True,
                     epochs=None, device=C.device, verbose=True, collect_scores=False, cfg=None):
    """Our method: Shortcut-aware Reweighting + Gradient Surgery.

    Stable two-phase approach:
      Phase 1: Warmup with standard training
      Phase 2: Compute ShortcutScores (one-time per-sample gradients)
      Phase 3: Weighted retraining with batch-level gradient surgery
    """
    _c = cfg or {}
    epochs = epochs if epochs is not None else _c.get('epochs', C.epochs)
    bs = _c.get('batch_size', C.batch_size)
    lr = _c.get('lr', C.lr)
    wd = _c.get('weight_decay', C.weight_decay)

    warmup_epochs = max(5, epochs // 6)
    main_epochs = epochs - warmup_epochs

    # Phase 1: Warmup
    if verbose:
        print(f"  Phase 1: Warmup ({warmup_epochs} epochs)")
    train_standard(model, dataset, epochs=warmup_epochs, device=device, verbose=False, cfg=cfg)

    # Phase 2: Compute ShortcutScores
    if verbose:
        print(f"  Phase 2: Computing ShortcutScores...")
    sample_scores, collected_data, g_V = _compute_sample_scores(model, dataset, device, cfg=cfg)

    # Compute per-sample weights
    sample_weights = []
    for S in sample_scores:
        w = compute_sample_weight(S).item() if use_reweighting else 1.0
        sample_weights.append(w)

    if verbose:
        n_scored = min(_c.get('score_max_samples', C.score_max_samples), len(sample_scores))
        avg_s = sum(sample_scores[:n_scored]) / n_scored
        avg_w = sum(sample_weights) / len(sample_weights)
        print(f"    Scored {n_scored}/{len(dataset['train'])} samples")
        print(f"    Avg ShortcutScore: {avg_s:.4f}, Avg weight: {avg_w:.4f}")

    # Phase 3: Weighted retraining with gradient surgery
    if verbose:
        print(f"  Phase 3: Weighted retraining ({main_epochs} epochs)")

    # Attach weights to samples
    weighted_samples = []
    for i, s in enumerate(dataset['train'].samples):
        ws = dict(s)
        ws['weight'] = sample_weights[i]
        weighted_samples.append(ws)

    from src.data import ReasoningDataset
    pad_id = getattr(dataset['train'], 'pad_id', C.PAD)
    weighted_ds = ReasoningDataset(weighted_samples, pad_id=pad_id)
    train_loader = get_dataloader(weighted_ds, batch_size=bs, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=wd)
    total_steps = main_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5)

    val_loader = get_dataloader(dataset['val'], shuffle=False, batch_size=bs)

    model.train()
    for epoch in range(main_epochs):
        total_loss = 0.0
        n_batches = 0

        # Periodically refresh validation gradient
        if use_gradient_surgery and epoch % 5 == 0:
            g_V = compute_validation_gradient(model, val_loader, device)
            model.train()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            weights = batch['weight'].to(device) if 'weight' in batch else None

            optimizer.zero_grad()
            logits = model(input_ids)

            # Weighted loss
            B, T, V = logits.shape
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V), target_ids.reshape(-1), reduction='none'
            ).reshape(B, T)
            masked_loss = loss_per_token * loss_mask

            if weights is not None and use_reweighting:
                per_sample_loss = masked_loss.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
                loss = (per_sample_loss * weights).mean()
            else:
                loss = masked_loss.sum() / loss_mask.sum().clamp(min=1)

            loss.backward()

            # Batch-level gradient surgery
            if use_gradient_surgery:
                batch_grad = get_grad_vector(model)
                norm_bg = batch_grad.norm()
                norm_gv = g_V.norm()
                if norm_bg > 1e-10 and norm_gv > 1e-10:
                    cos_sim = (batch_grad @ g_V) / (norm_bg * norm_gv)
                    if cos_sim.item() < C.tau_A:
                        g_mod = apply_gradient_surgery(
                            batch_grad, batch_grad, g_V,
                            C.tau_A - cos_sim.item(), 0.0)
                        # Preserve original gradient scale
                        g_mod = g_mod * (norm_bg / g_mod.norm().clamp(min=1e-10))
                        set_grad_vector(model, g_mod)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % max(1, main_epochs // 5) == 0:
            print(f"    Epoch {epoch+1}/{main_epochs}, Loss: {total_loss/n_batches:.4f}")

    if collect_scores:
        return model, collected_data
    return model


def self_consistency_predict(model, input_ids, eq_position, max_new_tokens=8,
                              num_samples=C.sc_num_samples, temperature=C.sc_temperature,
                              device=C.device, sep_id=C.SEP, eos_id=C.EOS):
    """Self-Consistency Decoding: sample multiple outputs, take majority vote.

    Returns:
        most_common_answer: list of token ids for the most voted answer
    """
    model.eval()
    prefix = input_ids[:eq_position + 1].unsqueeze(0).to(device)
    answers = []

    for _ in range(num_samples):
        generated = model.generate(prefix, max_new_tokens=max_new_tokens,
                                    temperature=temperature, greedy=False, eos_id=eos_id)
        # Extract answer tokens (after SEP)
        gen_tokens = generated[0].tolist()
        if sep_id in gen_tokens:
            sep_idx = gen_tokens.index(sep_id)
            ans = gen_tokens[sep_idx + 1:]
            if eos_id in ans:
                ans = ans[:ans.index(eos_id)]
            answers.append(tuple(ans))
        else:
            answers.append(tuple(gen_tokens[eq_position + 1:]))

    if not answers:
        return []

    # Majority vote
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]
    return list(most_common)
