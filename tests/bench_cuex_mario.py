import os

import jax
import jax.numpy as jnp
import numpy as np
from mace_jax import modules
import haiku as hk

import e3nn_jax as e3j

import json

IMPLEMENTATION = os.environ["IMPLEMENTATION"]
FILENAME = "benchmark_results.json"


def main():
    # Dataset specifications
    num_species = 50
    num_graphs = 100
    avg_num_neighbors = 20

    # model_size = os.environ.get("MODEL_SIZE", "MP-S")
    model_size = "MP-S"

    if "MP" in model_size:
        num_atoms = 3_000
        num_edges = 160_000
    else:
        num_atoms = 4_000
        num_edges = 70_000

    num_features = int(os.environ["NUM_FEATURES"])

    # num_features = {
    #     "MP-S": 128,
    #     "MP-M": 128,
    #     "MP-L": 128,
    #     "OFF-S": 64 + 32,
    #     "OFF-M": 128,
    #     "OFF-L": 128 + 64,
    # }[model_size]

    hidden_irreps = {
        "MP-S": "0e",
        "MP-M": "0e+1o",
        "MP-L": "0e+1o+2e",
        "OFF-S": "0e",
        "OFF-M": "0e+1o",
        "OFF-L": "0e+1o+2e",
    }[model_size]

    @hk.without_apply_rng
    @hk.transform
    def model(batch_dict):
        return modules.MACE(
            output_irreps="1x0e",
            num_interactions=2,
            num_features=num_features,
            num_species=num_species,
            max_ell=3,
            correlation=3,
            radial_basis=lambda r, r_max: e3j.bessel(r, 8, r_max),
            radial_envelope=lambda r, r_max: e3j.poly_envelope(5 - 1, 2, r_max)(r),
            readout_mlp_irreps=e3j.Irreps("16x0e"),
            # num_radial_basis=8,
            interaction_irreps=e3j.Irreps("0e+1o+2e+3o"),
            hidden_irreps=e3j.Irreps(hidden_irreps),
            # offsets=np.zeros(num_species),
            r_max=5.0,
            avg_num_neighbors=avg_num_neighbors,
            skip_connection_first_layer=("MP" in model_size),
            # replicate_original_group=False,
            implementation=IMPLEMENTATION
        )(
            e3j.IrrepsArray("1o", batch_dict["nn_vecs"]),
            batch_dict["species"],
            batch_dict["inda"],
            batch_dict["indb"])

    import optax

    # Dummy data
    vecs = jax.random.normal(jax.random.key(0), (num_edges, 3))
    species = jax.random.randint(jax.random.key(0), (num_atoms,), 0, num_species)
    senders, receivers = jax.random.randint(
        jax.random.key(0), (2, num_edges), 0, num_atoms
    )
    graph_index = jax.random.randint(jax.random.key(0), (num_atoms,), 0, num_graphs)
    graph_index = jnp.sort(graph_index)
    target_E = jax.random.normal(jax.random.key(0), (num_graphs,))
    target_F = jax.random.normal(jax.random.key(0), (num_atoms, 3))
    nats = jnp.zeros((num_graphs,), dtype=jnp.int32).at[graph_index].add(1)
    mask = jnp.ones((num_edges,), dtype=bool)

    batch_dict = dict(
        nn_vecs=vecs,
        species=species,
        inda=senders,
        indb=receivers,
        inde=graph_index,
        nats=nats,
        mask=mask,
    )

    # Initialization
    w = jax.jit(model.init)(jax.random.key(0), batch_dict)
    opt = optax.adam(1e-2)
    opt_state = opt.init(w)
    step_count = 0

    # print_footprint(w)

    # Training
    @jax.jit
    def step(w, opt_state, batch_dict: dict, target_E: jax.Array, target_F: jax.Array):
        def loss_fn(w):
            # E, F = model.apply(w, batch_dict)
            E = model.apply(w, batch_dict).array.sum()
            return jnp.mean((E - target_E) ** 2)

            # return jnp.mean((E - target_E) ** 2) + jnp.mean((F - target_F) ** 2)

        grad = jax.grad(loss_fn)(w)
        updates, opt_state = opt.update(grad, opt_state)
        w = optax.apply_updates(w, updates)
        return w, opt_state

    # compilation
    _ = step(w, opt_state, batch_dict, target_E, target_F)

    import time

    t0 = time.perf_counter()

    for i in range(10):
        (w, opt_state) = step(w, opt_state, batch_dict, target_E, target_F)
        step_count += 1
        # print_footprint(w)

    jax.block_until_ready(w)
    t1 = time.perf_counter()

    delta_t = t1 - t0
    runtime_per_step = 1e3 * (delta_t) / 10

    try:
        with open(FILENAME) as fd:
            results = json.load(fd)
    except FileNotFoundError:
        results = {}

    lmax = e3j.Irreps(hidden_irreps).lmax
    res = {
        f"mul_{num_features}_lmax_2": {
            "runtimes": {
                "forward": delta_t,
            },
            "memory_mb": 12345.6789
        }
    }

    convention = {"e3j": "bm_tp_e3j", "cuex": "bm_tp_e3jtocuex"}
    name = convention.get(IMPLEMENTATION, "bm_tp_e3jtocuex")
    impl = results.get(name, {})
    impl.update(res)
    results[name] = impl
    with open(FILENAME, "w") as fd:
        json.dump(results, fd, indent=4)

    print(f"{num_features},{runtime_per_step:.0f},{hidden_irreps}")

    # print_footprint(w)


def print_footprint(w):
    print(", ".join([f"{np.sum(x):.3f}" for x in jax.tree.leaves(w)]))


if __name__ == "__main__":
    main()
