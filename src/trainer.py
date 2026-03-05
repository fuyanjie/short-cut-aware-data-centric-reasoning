"""Training loops for all methods."""
import copy
import torch
import torch.nn.functional as F
from collections import Counter
from src.config import Config as C
from src.data import get_dataloader
from src.methods import (
    masked_ce_loss, compute_validation_gradient, compute_sample_gradients,
    compute_shortcut_score, compute_sample_weight, apply_gradient_surgery,
    get_grad_vector, set_grad_vector
)


def train_standard(model, dataset, epochs=C.epochs, device=C.device, verbose=True):
    """Standard supervised fine-tuning (Baseline a)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    train_loader = get_dataloader(dataset['train'], shuffle=True)
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

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    return model


def train_data_filtering(model, dataset, epochs=C.epochs, device=C.device, verbose=True):
    """Data Filtering baseline (Baseline c).

    Warm up for a few epochs, identify high-confidence samples as potential shortcuts,
    filter them out, then retrain.
    """
    # Phase 1: Warmup training
    warmup_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(warmup_model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    train_loader = get_dataloader(dataset['train'], shuffle=True, batch_size=C.batch_size)

    warmup_model.train()
    for epoch in range(C.df_warmup_epochs):
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
    keep_indices = []
    eval_loader = get_dataloader(dataset['train'], shuffle=False, batch_size=1)
    idx = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            answer_mask = batch['answer_mask'].to(device)

            logits = warmup_model(input_ids)
            probs = F.softmax(logits, dim=-1)

            # Check confidence on answer tokens
            B, T, V = probs.shape
            target_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            answer_conf = (target_probs * answer_mask).sum() / answer_mask.sum().clamp(min=1)

            if answer_conf.item() < C.df_confidence_threshold:
                keep_indices.append(idx)
            idx += 1

    # Phase 3: Retrain on filtered dataset
    filtered_samples = [dataset['train'].samples[i] for i in keep_indices]
    if verbose:
        print(f"  Data Filtering: kept {len(filtered_samples)}/{len(dataset['train'])} samples")

    from src.data import ReasoningDataset
    filtered_dataset = {**dataset, 'train': ReasoningDataset(filtered_samples)}
    return train_standard(model, filtered_dataset, epochs=epochs, device=device, verbose=verbose)


def train_our_method(model, dataset, use_reweighting=True, use_gradient_surgery=True,
                     epochs=C.epochs, device=C.device, verbose=True, collect_scores=False):
    """Our method: Shortcut-aware Reweighting + Gradient Surgery.

    Args:
        use_reweighting: enable Shortcut-aware Reweighting
        use_gradient_surgery: enable Gradient Surgery
        collect_scores: if True, collect ShortcutScores for analysis

    Returns:
        model, (optional) collected_data dict
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    train_loader = get_dataloader(dataset['train'], shuffle=True)
    val_loader = get_dataloader(dataset['val'], shuffle=False, batch_size=C.batch_size)

    # Initialize validation gradient
    g_V = compute_validation_gradient(model, val_loader, device)

    collected_data = {'scores': [], 'is_shortcut': [], 'alignments': [], 'concentrations': []}
    global_step = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            batch_size = batch['input_ids'].size(0)

            # Periodically update validation gradient
            if global_step % C.val_grad_interval == 0 and global_step > 0:
                g_V = compute_validation_gradient(model, val_loader, device)

            # Process each sample individually for per-sample gradients
            weighted_grads = []
            batch_loss = 0.0

            for i in range(batch_size):
                input_ids = batch['input_ids'][i]
                target_ids = batch['target_ids'][i]
                loss_mask = batch['loss_mask'][i]
                answer_mask = batch['answer_mask'][i]
                reasoning_mask = batch['reasoning_mask'][i]

                # Compute per-sample gradients
                g_full, g_ans, g_reason = compute_sample_gradients(
                    model, input_ids, target_ids, loss_mask, answer_mask,
                    reasoning_mask, device
                )

                # Compute ShortcutScore
                S, B_val, C_val, A_val, R_val = compute_shortcut_score(
                    g_full, g_ans, g_reason, g_V
                )

                if collect_scores:
                    collected_data['scores'].append(S)
                    collected_data['is_shortcut'].append(batch['is_shortcut'][i].item())
                    collected_data['alignments'].append(A_val)
                    collected_data['concentrations'].append(R_val)

                # Reweighting
                w = compute_sample_weight(S) if use_reweighting else torch.tensor(1.0)

                # Gradient Surgery
                if use_gradient_surgery:
                    g_mod = apply_gradient_surgery(g_full, g_ans, g_V, B_val, C_val)
                else:
                    g_mod = g_full

                weighted_grads.append(w * g_mod)

                # Track loss
                with torch.no_grad():
                    inp = input_ids.unsqueeze(0).to(device)
                    tgt = target_ids.unsqueeze(0).to(device)
                    lm = loss_mask.unsqueeze(0).to(device)
                    logits = model(inp)
                    sample_loss = masked_ce_loss(logits, tgt, lm)
                    batch_loss += sample_loss.item()

            # Aggregate gradients and update
            avg_grad = torch.stack(weighted_grads).mean(dim=0)
            optimizer.zero_grad()
            set_grad_vector(model, avg_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss
            n_samples += batch_size
            global_step += 1

        if verbose and (epoch + 1) % 5 == 0:
            avg_loss = total_loss / max(n_samples, 1)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    if collect_scores:
        return model, collected_data
    return model


def self_consistency_predict(model, input_ids, eq_position, max_new_tokens=8,
                              num_samples=C.sc_num_samples, temperature=C.sc_temperature,
                              device=C.device):
    """Self-Consistency Decoding: sample multiple outputs, take majority vote.

    Returns:
        most_common_answer: list of token ids for the most voted answer
    """
    model.eval()
    prefix = input_ids[:eq_position + 1].unsqueeze(0).to(device)
    answers = []

    for _ in range(num_samples):
        generated = model.generate(prefix, max_new_tokens=max_new_tokens,
                                    temperature=temperature, greedy=False)
        # Extract answer tokens (after SEP)
        gen_tokens = generated[0].tolist()
        if C.SEP in gen_tokens:
            sep_idx = gen_tokens.index(C.SEP)
            ans = gen_tokens[sep_idx + 1:]
            if C.EOS in ans:
                ans = ans[:ans.index(C.EOS)]
            answers.append(tuple(ans))
        else:
            answers.append(tuple(gen_tokens[eq_position + 1:]))

    if not answers:
        return []

    # Majority vote
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]
    return list(most_common)
