from collections import namedtuple

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import jraph

from mace_jax import modules

Node = namedtuple("Node", ["positions", "attrs"])
Edge = namedtuple("Edge", ["shifts"])
Globals = namedtuple("Globals", ["cell"])


def test_basics():
    graph = jraph.GraphsTuple(
        nodes=Node(
            positions=positions.array,
            attrs=jax.nn.one_hot(jnp.array([0, 1]), 2),
        ),
        edges=Edge(shifts=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        globals=Globals(cell=None),
        senders=jnp.array([0, 1]),
        receivers=jnp.array([1, 0]),
        n_edge=jnp.array([2]),
        n_node=jnp.array([2]),
    )

    modules.message_passing_cuex_translation.MessagePassingConvolution(
        avg_num_neighbors=1.0,
        target_irreps=
    )

    return e3nn.IrrepsArray("0e", energy)

    positions = e3nn.normal("1o", jax.random.PRNGKey(1), (2,))
    assert_equivariant(wrapper, jax.random.PRNGKey(1), args_in=(positions,))
