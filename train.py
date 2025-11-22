import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

def count_params(model: eqx.Module):
  num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
  num_millions = num_params / 1_000_000
  print(f"Model # of parameters: {num_millions:.2f}M")



# ------- loss function (matches your batched_forward pattern) -------
def loss_fn(model, state, x, y):
    """
    model: ResNet module (from make_with_state)
    state: BatchNorm/etc. state PyTree
    x:     (B, C, H, W)
    y:     (B,) integer labels
    """

    # Same pattern you used in the screenshot
    batched_forward = eqx.filter_vmap(
        model,
        in_axes=(0, None),
        axis_name="batch",
    )

    logits, state_batched = batched_forward(x, state)

    # Collapse the vmapped state back to a single state (your s[0] trick)
    new_state = jax.tree_util.tree_map(lambda s: s[0], state_batched)

    # Cross-entropy loss for integer labels
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    # has_aux=True will return (loss, new_state)
    return loss, new_state


# create optimizer once (or pass it in to train_step if you prefer)


# ------- single training step -------
@eqx.filter_jit
def train_step(model, state, opt_state,optimizer, x, y):
    """
    model:      ResNet module
    state:      BatchNorm/etc. state
    opt_state:  Optax optimizer state
    x, y:       training batch
    """

    # only `model` (arg 0) is differentiated; `state` is non-diff
    (loss, new_state), grads = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=True,
    )(model, state, x, y)

    updates, opt_state = optimizer.update(grads, opt_state, params=model)
    model = eqx.apply_updates(model, updates)

    return model, new_state, opt_state, loss



@eqx.filter_jit
def eval_step(model, state, x, y):
    """
    model: ResNet module (from eqx.nn.make_with_state)
    state: BatchNorm/etc. state (NOT updated here)
    x:     (B, C, H, W)
    y:     (B,) integer labels
    """

    # Put BN/Dropout etc. into inference mode: running stats are *used*,
    # but not updated.
    inference_model = eqx.nn.inference_mode(model)

    # Same pattern you used before: vmap over batch axis, share state
    batched_forward = eqx.filter_vmap(
        inference_model,
        in_axes=(0, None),       # x batched, state shared
        axis_name="batch",
    )

    # BN in inference_mode returns the same state for every example.
    logits, _ = batched_forward(x, state)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y)

    return loss, acc

def evaluate_model(model, state, data_loader):
    """
    Run eval over all batches and PRINT metrics (no return).
    """

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x_batch, y_batch in data_loader:
        x_batch, y_batch = jnp.array(x_batch), jnp.array(y_batch)
        loss, acc = eval_step(model, state, x_batch, y_batch)

        bsz = x_batch.shape[0]
        total_loss += float(loss) * bsz
        total_correct += float(acc) * bsz
        total_examples += bsz

    mean_loss = total_loss / total_examples
    mean_acc = total_correct / total_examples

    print(f"Eval — loss: {mean_loss:.4f}, acc: {mean_acc*100:.2f}%")



# ---------- eval over one epoch (no printing) ----------
def eval_epoch(model, state, val_loader):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x_batch, y_batch in val_loader:
        x_batch, y_batch = jnp.array(x_batch), jnp.array(y_batch)
        loss, acc = eval_step(model, state, x_batch, y_batch)

        bsz = x_batch.shape[0]
        total_loss += float(loss) * bsz
        total_correct += float(acc) * bsz
        total_examples += bsz

    mean_loss = total_loss / total_examples
    mean_acc = total_correct / total_examples
    return mean_loss, mean_acc


# ---------- main training loop with logging ----------
def train_and_log(
    model,
    state,
    opt_state,
    optimizer,
    train_loader,
    val_loader,
    num_epochs,
):
    """
    Returns:
      model, state, opt_state,
      train_loss_steps:   [(step, loss), ...]
      eval_loss_epochs:   [(epoch, val_loss), ...]
      eval_acc_epochs:    [(epoch, val_acc), ...]
    """

    train_loss_steps = []   # per-step training loss
    eval_loss_epochs = []   # per-epoch val loss
    eval_acc_epochs = []    # per-epoch val acc

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        # ---- training ----
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = jnp.array(x_batch), jnp.array(y_batch)
            model, state, opt_state, loss = train_step(
                model, state, opt_state,optimizer, x_batch, y_batch
            )

            global_step += 1
            train_loss_steps.append((global_step, float(loss)))

        # ---- evaluation for this epoch ----
        val_loss, val_acc = eval_epoch(model, state, val_loader)

        eval_loss_epochs.append((epoch, float(val_loss)))
        eval_acc_epochs.append((epoch, float(val_acc)))

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss(step {global_step}): {float(loss):.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}%"
        )

    return (
        model,
        state,
        opt_state,
        train_loss_steps,
        eval_loss_epochs,
        eval_acc_epochs,
    )

def plot_training_and_validation(steps, train_losses, epochs, val_losses, val_accs,Name):
    """
    steps:       list/array of global steps for training
    train_losses: list/array of training loss values (same length as steps)
    epochs:      list/array of epoch indices
    val_losses:  list/array of validation loss (per epoch)
    val_accs:    list/array of validation accuracy (per epoch) in [0, 1]
    """

    # 1) training loss over steps
    plt.figure()
    plt.plot(steps, train_losses, c="g")
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.title(f"{Name}: Training Loss vs Step")

    # 2) val loss & acc over epochs
    epochs = list(epochs)
    val_losses = list(val_losses)
    val_accs = list(val_accs)   # assume 0–1, will plot as %

    fig, ax1 = plt.subplots()

    # Left y-axis: loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.plot(epochs, val_losses, marker="o", color="r")
    ax1.tick_params(axis="y")

    # Right y-axis: accuracy
    ax2 = ax1.twinx()  # shares x-axis
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.plot(epochs, [a * 100 for a in val_accs], marker="x")
    ax2.tick_params(axis="y")

    fig.tight_layout()
    plt.title(f"{Name}: Validation Loss and Accuracy vs Epoch")
    plt.show()