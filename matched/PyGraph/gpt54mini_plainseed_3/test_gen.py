import pytest

from py_graph_t.Graph import Graph
from py_graph_t.SimpleGraph import SimpleGraph
from py_graph_t.exceptions.SimpleGraphException import (
    VertexNotExistsException,
    EdgeNotFoundException,
    VertexDuplicatedException,
    EdgeNameExistsException,
    EdgeDuplicatedException,
    LoopDetectedException,
)


@pytest.fixture
def graph():
    g = Graph()
    for v in ["A", "B", "C", "D"]:
        g.add_vertex(v)
    return g


@pytest.fixture
def simple_graph():
    g = SimpleGraph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    return g


def test_add_vertex_and_duplicate_raises():
    g = Graph()
    v = g.add_vertex("A")
    assert g.vertex_exists("A") is True
    assert v.get_value() == "A"
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex("A")


def test_add_edge_and_show_edge_returns_same_edge(graph):
    edge = graph.add_edge("A", "B", name="e1")
    found = graph.show_edge("A", "B")
    assert found == edge
    assert found.get_name() == "e1"
    assert graph.num_edges() == 1


def test_add_edge_with_missing_vertex_raises(graph):
    with pytest.raises(VertexNotExistsException):
        graph.add_edge("A", "Z", name="bad")


def test_edge_name_must_be_unique(graph):
    graph.add_edge("A", "B", name="e1")
    with pytest.raises(EdgeNameExistsException):
        graph.add_edge("B", "C", name="e1")


def test_delete_edge_removes_edge_and_returns_removed_edge(graph):
    graph.add_edge("A", "B", name="e1")
    removed = graph.delete_edge("A", "B")
    assert removed.get_name() == "e1"
    assert graph.num_edges() == 0
    assert graph.edge_exists("A", "B") is False


def test_delete_nonexistent_edge_raises(graph):
    with pytest.raises(EdgeNotFoundException):
        graph.delete_edge("A", "B")


def test_delete_vertex_removes_incident_edges(graph):
    graph.add_edge("A", "B", name="ab")
    graph.add_edge("A", "C", name="ac")
    graph.add_edge("B", "D", name="bd")
    removed = graph.delete_vertex("A")
    assert removed.get_value() == "A"
    assert graph.vertex_exists("A") is False
    assert graph.num_edges() == 1
    assert graph.list_graph_edges() == ["bd"]


def test_vertex_neighbors_and_degree(graph):
    graph.add_edge("A", "B", name="ab")
    graph.add_edge("A", "C", name="ac")
    neigh = graph.vertex_neighbors("A")
    assert {v.get_value() for v in neigh} == {"B", "C"}
    assert graph.vertex_degree("A") == 2


def test_is_vertices_adjacents(graph):
    graph.add_edge("A", "B", name="ab")
    assert graph.is_vertices_adjacents("A", "B") is True
    assert graph.is_vertices_adjacents("A", "C") is False


def test_list_graph_vertices_and_edges(graph):
    graph.add_edge("A", "B", name="ab")
    graph.add_edge("B", "C", name="bc")
    assert graph.list_graph_vertices() == ["A", "B", "C", "D"]
    assert graph.list_graph_edges() == ["ab", "bc"]


def test_adjacency_matrix_is_symmetric_and_counts_edges(graph):
    graph.add_edge("A", "B", name="ab")
    graph.add_edge("A", "C", name="ac")
    matrix = graph.adjacency_matrix()
    assert matrix["A"]["B"] == 1
    assert matrix["B"]["A"] == 1
    assert matrix["A"]["C"] == 1
    assert matrix["C"]["A"] == 1
    assert matrix["B"]["C"] == 0


def test_has_cycle_and_has_loop(graph):
    graph.add_edge("A", "B", name="ab")
    graph.add_edge("B", "C", name="bc")
    graph.add_edge("C", "A", name="ca")
    assert graph.has_cycle() is True
    assert graph.has_loop() is False


def test_check_regular_graph_true_and_false():
    regular = Graph()
    for v in ["A", "B", "C"]:
        regular.add_vertex(v)
    regular.add_edge("A", "B", name="ab")
    regular.add_edge("B", "C", name="bc")
    regular.add_edge("C", "A", name="ca")
    assert regular.check_regular_graph() is True

    non_regular = Graph()
    for v in ["A", "B", "C", "D"]:
        non_regular.add_vertex(v)
    non_regular.add_edge("A", "B", name="ab")
    non_regular.add_edge("A", "C", name="ac")
    assert non_regular.check_regular_graph() is False


def test_simple_graph_rejects_loops(simple_graph):
    with pytest.raises(LoopDetectedException):
        simple_graph.add_edge("A", "A", name="aa")


def test_simple_graph_rejects_parallel_edges(simple_graph):
    simple_graph.add_edge("A", "B", name="ab")
    with pytest.raises(EdgeDuplicatedException):
        simple_graph.add_edge("B", "A", name="ab2")


def test_simple_graph_requires_existing_vertices(simple_graph):
    with pytest.raises(VertexNotExistsException):
        simple_graph.add_edge("A", "Z", name="az")


def test_simple_graph_add_edge_and_show_edge(simple_graph):
    edge = simple_graph.add_edge("A", "B", name="ab")
    assert edge.get_name() == "ab"
    assert simple_graph.num_edges() == 1
    assert simple_graph.show_edge("A", "B") == edge
