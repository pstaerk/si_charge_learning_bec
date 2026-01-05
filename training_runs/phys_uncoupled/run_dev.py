def from_path(path, class_name, **kwargs):
    """
    Loads a class from a Python file and instantiates it.
    """
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(path).resolve()
    module_dir = str(path.parent)
    module_name = path.stem

    # Add the module's directory to the search path
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec for module at {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, class_name)

    return cls(**kwargs)


if __name__ == '__main__':
    from myrto.engine import read_yaml
    from pathlib import Path

    settings = read_yaml("settings.yaml")

    # -- settings --

    from marathon.data import datasets

    datasets = Path('')

    data_train_pbc = Path(settings["train_pbc"])
    data_valid_pbc = Path(settings["valid_pbc"])

    data_train_npbc = Path(settings["train_npbc"])
    data_valid_npbc = Path(settings["valid_npbc"])

    # {name: (source, save_predictions)}
    test_datasets = {
        "valid": (datasets / settings["valid_pbc"], False),
    }
    if "test_datasets" in settings:
        for k, (data, save) in settings["test_datasets"].items():
            test_datasets[k] = (datasets / data, save)

    # only support batch_shape for now
    batch_style = "batch_shape"
    # for the periodic case, chunk_size takes the role of num_graphs,
    # we vmap over chunks and accumulate gradients
    chunk_size = settings.get("chunk_size", 1)

    num_graphs = settings["num_graphs"]  # must be 2 for periodic
    num_nodes = settings["num_nodes"]
    num_edges = settings["num_edges"]

    loss_weights = settings.get("loss_weights", {"energy": 0.5, "forces": 0.5,
                                                 "apt":0.5})
    scale_by_variance = settings.get("scale_by_variance", False)

    start_learning_rate = float(settings.get("start_learning_rate", 1e-3))
    min_learning_rate = float(settings.get("min_learning_rate", 1e-6))

    max_epochs = settings.get("max_epochs", 2000)
    valid_every_epoch = settings.get("valid_every_epoch", 2)

    # lr decay
    decay_style = settings.get("decay_style", "linear")
    start_decay_after = settings.get("start_decay_after", 10)
    stop_decay_after = settings.get(
        "stop_decay_after", max_epochs
    )  # ignored for exponential

    seed = settings.get("seed", 0)
    print_model_summary = True
    benchmark_pipeline = settings.get("benchmark_pipeline", True)
    workdir = "run"

    use_wandb = settings.get("use_wandb", True)
    # used for wandb -- use folder names by default
    wandb_project = None
    wandb_name = None

    default_matmul_precision = settings.get("default_matmul_precision", "default")
    debug_nans = False  # ~50% slowdown, use with care

    # settings for grain
    worker_count = settings.get("worker_count", 4)
    worker_buffer_size = settings.get("worker_buffer_size", 2)

    neutralize_charge = settings.get("neutralize_charge", False)

    # -- imports & startup --

    import numpy as np
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_default_matmul_precision", default_matmul_precision)
    jax.config.update("jax_debug_nans", debug_nans)

    from pathlib import Path

    from marathon import comms

    reporter = comms.reporter()
    reporter.start("run")
    reporter.step("startup")

    # -- housekeeping based on settings --
    keys = list(loss_weights.keys())
    use_stress = "stress" in keys
    use_apt = "apt" in keys

    workdir = Path(workdir)

    # -- randomness --
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)

    # -- model --
    from myrto.engine import from_dict, read_yaml

    model_config = read_yaml("model.yaml")
    assert "model" in model_config
    assert "baseline" in model_config  # for compatibility w/ external models
    species_to_weight = model_config["baseline"]["elemental"]

    model_cfgs = model_config['model']
    path = model_cfgs.pop('path')
    class_name = model_cfgs.pop('model_name')

    # model = from_dict(model_config["model"])
    model = from_path(path, class_name, **model_cfgs)
    cutoff = model.cutoff

    params = model.init(init_key, *model.dummy_inputs())

    if print_model_summary:
        from flax import linen as nn

        msg = nn.tabulate(model, init_key)(*model.dummy_inputs())
        comms.state(msg.split("\n"), title="Model Summary")

    num_parameters = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    comms.state(f"Parameter count: {num_parameters}")

    # -- checkpointers --
    from marathon.emit import SummedMetric

    checkpointers = []

    name = "R2_" + "+".join([k[0].upper() for k in keys])
    checkpointers.append(SummedMetric(name, "r2", keys=keys))

    name = "MAE_" + "+".join([k[0].upper() for k in ["forces"]])
    checkpointers.append(SummedMetric(name, "mae", keys=["forces"]))

    checkpointers = tuple(checkpointers)

    # -- data loading --
    from marathon.evaluate.metrics import get_stats
    from marathon.extra.hermes import (
        DataLoader,
        DataSource,
        FilterEmpty,
        IndexSampler,
        ToStack,
        prefetch_to_device,
    )
    from marathon.extra.hermes.pain import Record, RecordMetadata
    from transforms import ToFixedShapeBatch, SetUpEwald, ToSample, SetUpEwaldFixedKgrid

    to_sample = ToSample(cutoff=cutoff, energy=True, forces=True,
                         stress=use_stress, apt=use_apt)

    def get_batcher():
        assert batch_style == "batch_shape"
        return ToFixedShapeBatch(
            num_graphs=num_graphs, num_edges=num_edges, num_nodes=num_nodes,
            keys=('energy', 'forces', 'apt'),
        )

    # source_train = DataSource(data_train, species_to_weight=species_to_weight)
    source_train_pbc = DataSource(data_train_pbc,
                                  species_to_weight=species_to_weight)
    source_train_npbc = DataSource(data_train_npbc,
                                   species_to_weight=species_to_weight)

    # source_valid = DataSource(data_valid, species_to_weight=species_to_weight)
    source_valid_pbc = DataSource(data_valid_pbc,
                                  species_to_weight=species_to_weight)
    source_valid_npbc = DataSource(data_valid_npbc,
                                   species_to_weight=species_to_weight)

    baseline = {"elemental": species_to_weight}
    n_train_pbc = len(source_train_pbc)
    n_valid_pbc = len(source_valid_pbc)

    n_train_npbc = len(source_train_npbc)
    n_valid_npbc = len(source_valid_npbc)

    # Strategy: Find out largest box and use the k_grid for all systems
    smearing = cutoff / 5.
    lr_wavelength = smearing / 0.5
    # only need to look into pbc, as pbc is solved with Ewald, npbc not
    cells_valid = [s.cell for s in source_valid_pbc] 
    cells_train = [s.cell for s in source_train_pbc]
    max_cell = np.max(np.concatenate((cells_valid, cells_train)), axis=0)
    ns = jnp.ceil(jnp.linalg.norm(max_cell, axis=-1) / lr_wavelength)
    k_grid = np.ones((int(ns[0]), int(ns[1]), int(ns[2])))

    reporter.step(f'Using {k_grid.shape=}')

    prepare_ewald = SetUpEwaldFixedKgrid(smearing=smearing, k_grid=k_grid)

    max_steps = max_epochs * (n_train_pbc + n_train_npbc)
    valid_every = valid_every_epoch * (n_train_pbc + n_train_npbc)
    comms.talk(f"run for {max_epochs} epochs, {max_steps} steps", full=True)
    comms.talk(
        f"validate every {valid_every_epoch} epochs, every {valid_every} steps",
        full=True,
    )

    reporter.step("loading validation set")

    # for now we assume that validation set fits into RAM easily
    valid_samples_pbc = []
    valid_samples_npbc = []
    batcher = get_batcher()

    def valid_iterator1():
        filterer = FilterEmpty()
        for i in range(n_valid_pbc):
            sample = to_sample.map(source_valid_pbc[i])
            if filterer.filter(sample):
                valid_samples_pbc.append(sample)
                yield Record(
                    data=sample, metadata=RecordMetadata(index=i, record_key=i)
                )
    def valid_iterator2():
        filterer = FilterEmpty()
        for i in range(n_valid_npbc):
            sample = to_sample.map(source_valid_npbc[i])
            if filterer.filter(sample):
                valid_samples_npbc.append(sample)
                yield Record(
                    data=sample, metadata=RecordMetadata(index=i, record_key=i)
                )

    # Ewald for the periodic, no Ewald for the non-periodic
    data_valid_pbc = [prepare_ewald.map(b.data) for b in batcher(valid_iterator1())]
    data_valid_npbc = [b.data for b in batcher(valid_iterator2())]

    valid_stats = get_stats(valid_samples_pbc, keys=keys)

    max_steps = max_epochs * (n_train_pbc + n_train_npbc)
    valid_every = valid_every_epoch * (n_train_pbc + n_train_npbc)
    comms.talk(f"run for {max_epochs} epochs, {max_steps} steps", full=True)
    comms.talk(
        f"validate every {valid_every_epoch} epochs, every {valid_every} steps",
        full=True,
    )

    reporter.step("loading validation set")

    valid_batch_sizes = np.array([batch.graph_mask.sum() for batch in data_valid_pbc])
    valid_batch_sizes = np.concatenate(
        (valid_batch_sizes, np.array([batch.graph_mask.sum() for batch in data_valid_npbc]))
    )
    median_valid_batch_size = int(np.median(valid_batch_sizes))

    if scale_by_variance:
        old_loss_weights = loss_weights

        loss_weights = {k: v / valid_stats[k]["var"]
                        for k, v in loss_weights.items()}

        msg = []
        for k, v in loss_weights.items():
            msg.append(f"{k}: {old_loss_weights[k]:.3f} -> {v:.3f}")
        comms.state(msg, title="variance scaled loss weights")

    del valid_samples_pbc
    del valid_samples_npbc

    reporter.step("setup training pipeline")

    def get_training_iterator(num_epochs):
        batchers = [get_batcher(), prepare_ewald]

        if chunk_size > 1:
            batchers.append(ToStack(batch_size=chunk_size, drop_remainder=True))

        return iter(
            DataLoader(
                data_source=source_train_pbc,
                sampler=IndexSampler(
                    n_train_pbc,
                    num_epochs=num_epochs,
                    seed=seed,
                ),
                operations=[
                    to_sample,
                    FilterEmpty(),
                    *batchers,
                ],
                worker_count=worker_count,
                worker_buffer_size=worker_buffer_size,
            )
        )

    # Second training iterator for non-periodic systems without Ewald
    def get_training_iterator_type2(num_epochs):
        batchers = [get_batcher(), prepare_ewald]

        if chunk_size > 1:
            batchers.append(ToStack(batch_size=chunk_size, drop_remainder=True))

        return iter(
            DataLoader(
                data_source=source_train_npbc,
                sampler=IndexSampler(
                    n_train_npbc,
                    num_epochs=num_epochs,
                    seed=seed + 1,  # Different seed for different data
                ),
                operations=[
                    to_sample,
                    FilterEmpty(),
                    *batchers,
                ],
                worker_count=worker_count,
                worker_buffer_size=worker_buffer_size,
            )
        )

    if benchmark_pipeline:
        from time import monotonic

        reporter.step("benchmark training pipeline", spin=False)

        @jax.jit
        def test_fn(batch):
            if chunk_size == 1:
                return (
                    batch.edge_mask.sum(),
                    batch.node_mask.sum(),
                    batch.graph_mask.sum(),
                    batch.edge_mask.shape[0],
                    batch.node_mask.shape[0],
                    batch.graph_mask.shape[0],
                )
            else:
                return (
                    batch.edge_mask.sum(),
                    batch.node_mask.sum(),
                    batch.graph_mask.sum(),
                    batch.edge_mask.shape[0] * batch.edge_mask.shape[1],
                    batch.node_mask.shape[0] * batch.node_mask.shape[1],
                    batch.graph_mask.shape[0] * batch.graph_mask.shape[1],
                )

        # trigger jit
        test_fn(next(get_training_iterator(1)))

        test_iter = prefetch_to_device(get_training_iterator(1), 2)

        results = []
        start = monotonic()
        for i, batch in enumerate(test_iter):
            reporter.tick(f"chunk {i}")
            results.append(test_fn(batch))
            del batch
        results = np.array(results)
        duration = monotonic() - start

        real_samples = results[:, 2].sum()
        util_edges = 100 * results[:, 0] / results[:, 3]
        util_nodes = 100 * results[:, 1] / results[:, 4]
        util_samples = 100 * results[:, 2] / results[:, 5]
        pipeline_speed = duration / real_samples

        unique_edges = np.unique(results[:, 3]).shape[0]
        unique_nodes = np.unique(results[:, 4]).shape[0]
        unique_samples = np.unique(results[:, 4]).shape[0]

        num_chunks = i + 1
        num_batches = num_chunks * chunk_size

        msg = []
        msg.append(f"speed       : {1e6*pipeline_speed:.0f}µs/sample")
        msg.append(f"              {worker_count} workers, buffer {worker_buffer_size}")
        msg.append(
            f"edges  : {np.mean(util_edges):.2f}% / {np.mean(results[:, 0]):.0f} mean"
        )
        msg.append(
            f"nodes  : {np.mean(util_nodes):.2f}% / {np.mean(results[:, 1]):.0f} mean"
        )
        msg.append(
            f"samples: {np.mean(util_samples):.2f}% / {np.mean(results[:, 2]):.0f} mean"
        )

        msg.append("")
        msg.append(
            f"unique shapes: {unique_edges} edges, {unique_nodes} nodes, {unique_samples} samples"
        )
        msg.append(
            f"... -> expecting {unique_edges*unique_nodes*unique_samples} compilations"
        )
        msg.append("")
        if chunk_size > 1:
            msg.append(f"num chunks: {num_chunks} containing {chunk_size} batches")
        msg.append(
            f"num batches: {num_batches} ({real_samples/num_batches:.0f} samples/batch)"
        )

        comms.state(msg, title="Training Pipeline Statistics")

        if np.mean(util_edges) < 50 or np.mean(util_nodes) < 50:
            comms.warn("Ratio of real to padded edges or nodes is TOO LOW (<50%). No!")
            comms.warn("I SHOULD REFUSE TO CONTINUE WITH THIS SICK JOB ...")

        median_train_batch_size = int(np.median(results[:, 2]) / chunk_size)

        median_batch_size = median_train_batch_size
        batches_per_epoch = num_batches
    else:
        pipeline_speed = 0.0
        median_batch_size = median_valid_batch_size
        batches_per_epoch = int(len(source_train) / median_batch_size)

    comms.talk(f"estimated samples/batch: {median_batch_size}")
    comms.talk(f"estimated batches/epoch: {batches_per_epoch}")

    iter_train = get_training_iterator(max_epochs)

    # -- optimizer --
    import optax

    reporter.step("setup optimizer")

    if decay_style == "linear":
        transition_steps = stop_decay_after * batches_per_epoch
        initial_steps = start_decay_after * batches_per_epoch
        scheduler = optax.schedules.linear_schedule(
            init_value=start_learning_rate,
            end_value=min_learning_rate,
            transition_begin=initial_steps,
            transition_steps=transition_steps - initial_steps,
        )

    elif decay_style == "exponential":
        transition_steps = max_epochs * batches_per_epoch
        initial_steps = start_decay_after * batches_per_epoch
        decay_rate = min_learning_rate / start_learning_rate
        scheduler = optax.schedules.exponential_decay(
            init_value=start_learning_rate,
            transition_steps=transition_steps - initial_steps,
            transition_begin=initial_steps,
            decay_rate=decay_rate,
            end_value=min_learning_rate,
        )

    @optax.inject_hyperparams
    def optimizer(learning_rate):
        return optax.lamb(learning_rate)

    optimizer = optimizer(scheduler)

    initial_opt_state = optimizer.init(params)

    # -- assemble state / handle restore --

    state = {
        "epoch": 0,
        "checkpointers": checkpointers,
        "opt_state": initial_opt_state,
        "iter_train": iter_train.get_state(),
    }

    if workdir.is_dir():
        from marathon.emit import get_latest

        comms.warn(
            f"found working directory {workdir}, will restore (only) model and optimisation state!"
        )
        reporter.step("restoring")

        items = get_latest(workdir, state)

        if items is None:
            comms.warn(f"failed to find checkpoints in workdir {workdir}, ignoring")
        else:
            params, state, new_model, _, _, _ = items

            comms.talk(f"restored step {state['epoch']}")

            # try to catch the most obvious error: editing the model config between restarts
            from myrto.engine.serialize import to_dict

            assert to_dict(new_model) == to_dict(model)

            iter_train.set_state(state["iter_train"])
    else:
        workdir.mkdir()

    opt_state = state["opt_state"]

    # -- loggers --
    from myrto.engine import to_dict

    from marathon.emit import Txt

    reporter.step("setup loggers")

    training_pipeline = {
        "style": "shape",
        "num_graphs": num_graphs,
        "num_edges": num_edges,
        "num_nodes": num_nodes,
    }

    if decay_style == "linear":
        lr_decay = {
            "style": "linear",
            "start_decay_after": start_decay_after,
            "stop_decay_after": stop_decay_after,
        }
    elif decay_style == "exponential":
        lr_decay = {"style": "exponential", "start_decay_after": start_decay_after}
    else:
        raise ValueError

    config = {
        "n_train": n_train_pbc+n_train_npbc,
        "n_valid": n_valid_pbc+n_valid_npbc,
        "loss_weights": loss_weights,
        "max_steps": max_steps,
        "start_learning_rate": start_learning_rate,
        "min_learning_rate": min_learning_rate,
        "lr_decay": lr_decay,
        "chunk_size": chunk_size,
        "training_pipeline": training_pipeline,
        "valid_every": valid_every,
        "model": to_dict(model),
        "num_parameters": num_parameters,
        "worker_count": worker_count,
        "worker_buffer_size": worker_buffer_size,
    }

    metrics = {key: ["r2", "mae", "rmse"] for key in keys}

    loggers = [Txt(metrics=metrics)]

    if use_wandb:
        import wandb

        from marathon.emit import WandB

        this_folder = Path.cwd()

        if wandb_project is None:
            wandb_project = (
                f"{this_folder.parent.parent.stem}.{this_folder.parent.stem}"
            )

        if wandb_name is None:
            wandb_name = f"{this_folder.stem}"

        run = wandb.init(config=config, name=wandb_name, project=wandb_project)

        config["wandb_id"] = run.id

        loggers.append(WandB(run, metrics=metrics))

    # -- setup actual training loop --
    from time import monotonic

    from marathon.emit import save_checkpoints
    from marathon.evaluate import get_metrics_fn #, get_predict_fn, get_loss_fn, 
    from marathon.utils import s_to_string, tree_concatenate, tree_stack

    from predict import get_predict_fn
    from loss import get_loss_fn

    reporter.step("setup training loop")

    # pred_fn = lambda params, batch: model.predict(params, batch, stress=use_stress)
    pred_fn = get_predict_fn(model.apply, stress=use_stress, electrostatics="ewald",
                             excess_charge_neutralization=neutralize_charge)

    _loss_fn = get_loss_fn(pred_fn, weights=loss_weights)

    if chunk_size > 1:

        def loss_fn(params, batch):
            losses, auxs = jax.vmap(lambda x: _loss_fn(params, x))(batch)
            loss = losses.mean()
            aux = jax.tree.map(lambda x: x.sum(axis=0), auxs)

            return loss, aux

    else:
        loss_fn = _loss_fn

    loss_fn = jax.jit(loss_fn)

    train_metrics_fn = get_metrics_fn(keys=keys)  # no stats
    valid_metrics_fn = get_metrics_fn(keys=keys, stats=valid_stats)

    # ... manager preamble

    def get_lr(opt_state):
        return float(opt_state.hyperparams["learning_rate"])

    def report_on_lr(opt_state):
        lr = get_lr(opt_state)
        return f"LR: {lr:.3e}"

    def format_metrics(metrics, keys=["energy", "forces"]):
        key_to_unit = {"energy": "meV/atom", "forces": "meV/Å", "stress": "meV", "apt": "e"}
        key_to_name = {"energy": "E", "forces": "F", "stress": "σ", "apt": "APT"}
        msg = []

        for key in keys:
            m = metrics[key]

            msg.append(f". {key_to_name[key]}")
            if "r2" in m:
                msg.append(f".. R2  : {m['r2']:.3f} %")
            msg.append(f".. MAE : {m['mae']:.3e} {key_to_unit[key]}")
            msg.append(f".. RMSE: {m['rmse']:.3e} {key_to_unit[key]}")

        return msg

    class Manager:
        def __init__(
            self, state, interval, loggers, workdir, model, baseline, max_epochs
        ):
            self.state = state
            self.interval = interval
            self.loggers = loggers
            self.workdir = workdir
            self.model = model
            self.baseline = baseline

            self.max_epochs = max_epochs

            self.start_epoch = state["epoch"]
            self.start_time = monotonic()

            self.cancel = False

        @property
        def done(self):
            return self.epoch >= self.max_epochs or self.cancel

        @property
        def epoch(self):
            return self.state["epoch"]

        @property
        def elapsed(self):
            return monotonic() - self.start_time

        @property
        def time_per_epoch(self):
            return self.elapsed / (self.epoch - self.start_epoch)

        @property
        def compute_time_per_epoch(self):
            return self.time_per_epoch - pipeline_speed

        @property
        def eta(self):
            return (self.max_epochs - self.epoch) * self.time_per_epoch

        def should_validate(self, epoch):
            return epoch >= self.epoch + self.interval

        def report(
            self,
            epoch,
            params,
            opt_state,
            train_state,
            train_loss,
            train_metrics,
            valid_loss,
            valid_metrics,
            info={},
        ):
            assert epoch > self.epoch  # always forward

            self.state["epoch"] = epoch
            self.state["opt_state"] = opt_state
            self.state["iter_train"] = train_state

            if jnp.isnan(train_loss):
                comms.warn(f"loss became NaN at epoch={self.epoch}, canceling training")
                self.cancel = True

            if get_lr(opt_state) < min_learning_rate:
                # sometimes we stop decay before max_epochs, in that case don't break
                if stop_decay_after == max_epochs:
                    comms.talk(
                        f"learning rate has reached minimum at epoch={self.epoch}, canceling"
                    )
                    self.cancel = True

            info = {
                "lr": get_lr(opt_state),
                "time_per_epoch": self.time_per_epoch,
                "compute_time_per_epoch": self.compute_time_per_epoch,
                **info,
            }

            for logger in self.loggers:
                logger(
                    self.state["epoch"],
                    train_loss,
                    train_metrics,
                    valid_loss,
                    valid_metrics,
                    other=info,
                )

            metrics = {"train": train_metrics, "valid": valid_metrics}
            metrics = jax.tree_util.tree_map(lambda x: np.array(x), metrics)

            save_checkpoints(
                metrics,
                params,
                self.state,
                self.model,
                self.baseline,
                self.workdir,
                config=config,
            )

            title = f"state at step: {self.epoch}"
            msg = []

            msg.append(f"train loss: {train_loss:.5e}")
            msg.append(f"valid loss: {valid_loss:.5e}")

            msg.append(report_on_lr(opt_state))

            msg.append("validation errors:")
            msg += format_metrics(metrics["valid"], keys=keys)

            msg.append("")
            msg.append(f"elapsed: {s_to_string(self.elapsed, 's')}")
            msg.append(
                f"timing: {s_to_string(self.time_per_epoch)}/step, {s_to_string(self.eta, 'm')} ETA"
            )

            msg.append("")
            comms.state(msg, title=title)

    manager = Manager(state, valid_every, loggers, workdir, model, baseline, max_steps)

    @jax.jit
    def compute_grads(params, batch):
        """Compute gradients without applying them."""
        loss_and_aux, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
            params, batch
        )
        loss, aux = loss_and_aux
        return grads, loss, aux

    @jax.jit
    def apply_accumulated_grads(params, opt_state, grads_list, losses):
        """Average accumulated gradients and apply update."""
        # Average the gradients
        avg_grads = jax.tree.map(lambda *gs: jnp.stack(gs).mean(axis=0), *grads_list)
        
        # Use mean loss for optimizer
        mean_loss = jnp.mean(jnp.array(losses))
        
        updates, opt_state = optimizer.update(avg_grads, opt_state, params, value=mean_loss)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state

    aggregate_loss = np.mean
    aggregate_aux = tree_stack

    # -- train! --
    import itertools

    reporter.step("🚄", spin=False)

    start = monotonic()

    # TODO: Create your second iterator here
    # iter_train_type2 = get_training_iterator_type2(max_epochs)
    iter_train_type2 = get_training_iterator_type2(max_epochs)
    
    iter_train_with_prefetch = prefetch_to_device(iter_train, 2)
    iter_train_type2_with_prefetch = prefetch_to_device(iter_train_type2, 2)
    iter_valid_with_prefetch = (
        prefetch_to_device(itertools.cycle(data_valid_pbc), 2)
    )
    iter_valid_with_prefetch_npbc = (
        prefetch_to_device(itertools.cycle(data_valid_npbc), 2)
    )

    ran_steps = 0
    train_aux = []
    train_loss = []
    report = None
    cache_size = 1
    accumulation_steps = 2  # Number of batches to accumulate before updating
    
    while True:
        if manager.done:
            break
            
        # Gradient accumulation loop
        grads_list = []
        batch_losses = []
        batch_auxs = []
        steps_in_accumulation = 0
        
        for acc_step in range(accumulation_steps):
            try:
                # Alternate between batch types
                if acc_step % 2 == 0:
                    batch = next(iter_train_with_prefetch)
                else:
                    # TODO: Uncomment when you have second iterator
                    batch = next(iter_train_type2_with_prefetch)
                    # batch = next(iter_train_with_prefetch)  # Placeholder
            except StopIteration:
                comms.talk("exhausted training iterator")
                manager.cancel = True
                break
            
            if manager.done:
                break
            
            # Compute gradients without applying
            grads, loss, aux = compute_grads(params, batch)
            
            grads_list.append(grads)
            batch_losses.append(loss)
            batch_auxs.append(aux)
            steps_in_accumulation += batch.graph_mask.sum()
            
            del batch
        
        if manager.done:
            break
        
        # Apply accumulated gradients
        if len(grads_list) > 0:
            params, opt_state = apply_accumulated_grads(params, opt_state, grads_list, batch_losses)
            
            ran_steps += steps_in_accumulation
            current_step = manager.epoch + ran_steps
            reporter.tick(f"{current_step}")
            
            if apply_accumulated_grads._cache_size() > cache_size:
                cache_size = apply_accumulated_grads._cache_size()
                comms.talk(f"recompiled at step={current_step} ({cache_size})")
            
            # Store metrics (average loss, all auxs)
            train_loss.append(np.mean([float(l) for l in batch_losses]))
            train_aux.extend(batch_auxs)

        if report is not None:
            manager.report(*report)
            report = None

        if manager.should_validate(manager.epoch + ran_steps):
            valid_aux = []
            valid_loss = []
            for i in range(len(data_valid_pbc)):
                batch = next(iter_valid_with_prefetch)
                reporter.tick(f"{current_step} (valid {i})")

                # for chunk_size > 1, we use the non-vmapped fn here
                loss, aux = jax.jit(_loss_fn)(params, batch)

                valid_aux.append(aux)
                valid_loss.append(loss)

            for i in range(len(data_valid_npbc)):
                batch = next(iter_valid_with_prefetch_npbc)
                reporter.tick(f"{current_step} (valid npbc {i})")

                # for chunk_size > 1, we use the non-vmapped fn here
                loss, aux = jax.jit(_loss_fn)(params, batch)

                valid_aux.append(aux)
                valid_loss.append(loss)

            train_aux = aggregate_aux(train_aux)
            train_metrics = train_metrics_fn(train_aux)

            valid_aux = tree_stack(valid_aux)
            valid_metrics = valid_metrics_fn(valid_aux)

            train_loss = aggregate_loss(train_loss)
            valid_loss = np.mean(valid_loss)

            report = (
                manager.epoch + ran_steps,
                params,
                opt_state,
                iter_train.get_state(),
                train_loss,
                train_metrics,
                valid_loss,
                valid_metrics,
                {
                    "compiles_compute_grads": compute_grads._cache_size(),
                    "compiles_apply_grads": apply_accumulated_grads._cache_size(),
                    "compiles_loss_fn": loss_fn._cache_size(),
                },
            )

            train_aux = []
            train_loss = []
            ran_steps = 0

    # -- wrap up --
    from marathon.emit import get_all, plot

    reporter.step("wrapup")

    pred_fn = jax.jit(pred_fn)

    def get_batcher():
        return ToFixedShapeBatch(num_graphs=2, num_edges=num_edges, num_nodes=num_nodes)

    test = {}
    for name, (source, save) in test_datasets.items():
        source = DataSource(source, species_to_weight=species_to_weight)
        batcher = get_batcher()

        def it():
            for i in range(len(source)):
                sample = to_sample.map(source[i])
                yield Record(
                    data=sample, metadata=RecordMetadata(index=i, record_key=i)
                )

        batches = [prepare_ewald.map(b.data) for b in batcher(it())]
        test[name] = (batches, save)

    def predict_and_collate(params, batches):
        predictions = {k: [] for k in keys}
        labels = {k: [] for k in keys}

        for batch in batches:
            preds = pred_fn(params, batch)

            for key in keys:
                mask = batch.labels[key + "_mask"]
                if mask.any():
                    predictions[key].append(preds[key][mask])
                    labels[key].append(batch.labels[key][mask])

        final_predictions = {}
        final_labels = {}

        for key in predictions.keys():
            if "energy" in key:
                final_predictions[key] = np.array(predictions[key]).flatten()
            if "forces" in key:
                final_predictions[key] = np.concatenate(predictions[key]).reshape(-1, 3)
            if "stress" in key:
                final_predictions[key] = np.array(predictions[key]).reshape(-1, 3, 3)

        for key in keys:
            if key == "energy":
                final_labels[key] = np.array(labels[key]).flatten()
            if key == "forces":
                final_labels[key] = np.concatenate(labels[key]).reshape(-1, 3)
            if key == "stress":
                final_labels[key] = np.array(labels[key]).reshape(-1, 3, 3)

        return final_labels, final_predictions

    for f, items in get_all(workdir, state):
        if f.suffix == ".backup":
            continue

        comms.talk(f"working on {f}")

        params, _, _, _, metrics, _ = items

        for name, (batches, save) in test.items():
            labels, predictions = predict_and_collate(params, batches)

            out = f / f"plot/{name}"
            out.mkdir(parents=True, exist_ok=True)

            plot(out, predictions, labels)

            if save:
                np.savez_compressed(out / "energy.npz", predictions["energy"])
                np.savez_compressed(out / "forces.npz", predictions["forces"])

    reporter.done()
    if use_wandb:
        run.finish()

    comms.talk("cleaning up")
    import shutil

    if use_wandb:
        wandb_dir = Path("wandb")
        if wandb_dir.is_dir():
            shutil.rmtree(wandb_dir)

    for f, items in get_all(workdir, state):
        if f.suffix == ".backup":
            shutil.rmtree(f)

    comms.state("done!")
