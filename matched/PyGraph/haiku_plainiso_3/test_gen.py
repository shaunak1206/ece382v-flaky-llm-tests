import pytest

from py_graph_t.Graph import Graph
from py_graph_t.SimpleGraph import SimpleGraph
from py_graph_t.exceptions.SimpleGraphException import (
    VertexNotExistsException,
    EdgeDuplicatedException,
    LoopDetectedException,
    EdgeNotFoundException,
    VertexDuplicatedException,
    EdgeNameExistsException,
)


@pytest.fixture
def graph():
    return Graph()


@pytest.fixture
def simple_graph():
    return SimpleGraph()


@pytest.fixture
def triangle():
    g = SimpleGraph()
    for v in ("a", "b", "c"):
        g.add_vertex(v)
    g.add_edge("a", "b", "ab")
    g.add_edge("b", "c", "bc")
    g.add_edge("c", "a", "ca")
    return g


@pytest.fixture
def path():
    g = SimpleGraph()
    for v in ("a", "b", "c"):
        g.add_vertex(v)
    g.add_edge("a", "b", "ab")
    g.add_edge("b", "c", "bc")
    return g


# ---------- Graph: vertices ----------

def test_add_vertex_returns_vertex_and_registers_it(graph):
    vertex = graph.add_vertex("a")
    assert vertex is not None
    assert vertex.get_value() == "a"
    assert graph.vertex_exists("a")
    assert graph.num_vertex() == 1


def test_add_duplicate_vertex_raises(graph):
    graph.add_vertex("a")
    with pytest.raises(VertexDuplicatedException):
        graph.add_vertex("a")
    assert graph.num_vertex() == 1


def test_vertex_exists_false_for_unknown_vertex(graph):
    assert graph.vertex_exists("missing") is False


def test_list_graph_vertices_contains_all_identifiers(graph):
    for v in ("a", "b", "c"):
        graph.add_vertex(v)
    assert sorted(graph.list_graph_vertices()) == ["a", "b", "c"]


def test_delete_vertex_removes_incident_edges_only(triangle):
    removed = triangle.delete_vertex("a")
    assert removed.get_value() == "a"
    assert not triangle.vertex_exists("a")
    assert triangle.num_vertex() == 2
    # Only the edge "bc" does not touch "a"
    assert triangle.list_graph_edges() == ["bc"]
    assert triangle.num_edges() == 1


def test_delete_missing_vertex_raises(graph):
    with pytest.raises(VertexNotExistsException):
        graph.delete_vertex("ghost")


# ---------- Graph: edges ----------

def test_add_edge_with_missing_vertex_raises(graph):
    graph.add_vertex("a")
    with pytest.raises(VertexNotExistsException):
        graph.add_edge("a", "b", "ab")
    assert graph.num_edges() == 0


def test_add_edge_with_duplicate_name_raises(graph):
    graph.add_vertex("a")
    graph.add_vertex("b")
    graph.add_vertex("c")
    graph.add_edge("a", "b", "same")
    with pytest.raises(EdgeNameExistsException):
        graph.add_edge("b", "c", "same")


def test_add_edge_returns_edge_and_stores_name(graph):
    graph.add_vertex("a")
    graph.add_vertex("b")
    edge = graph.add_edge("a", "b", "ab")
    assert edge.get_name() == "ab"
    assert graph.num_edges() == 1
    assert graph.list_graph_edges() == ["ab"]


def test_show_edge_returns_existing_edge(triangle):
    edge = triangle.show_edge("a", "b")
    assert edge.get_name() == "ab"


def test_show_edge_raises_when_no_edge_between_vertices(path):
    with pytest.raises(EdgeNotFoundException):
        path.show_edge("a", "c")


def test_delete_edge_removes_and_returns_edge(triangle):
    removed = triangle.delete_edge("a", "b")
    assert removed.get_name() == "ab"
    assert triangle.num_edges() == 2
    assert triangle.edge_exists("a", "b") is False


def test_delete_missing_edge_raises(path):
    with pytest.raises(EdgeNotFoundException):
        path.delete_edge("a", "c")


def test_edge_name_exists(triangle):
    assert triangle.edge_name_exists("ab") is True
    assert triangle.edge_name_exists("zz") is False


# ---------- Graph: neighborhood / structure ----------

def test_vertex_neighbors_and_degree_on_triangle(triangle):
    neighbors = triangle.vertex_neighbors("a")
    assert sorted(v.get_value() for v in neighbors) == ["b", "c"]
    assert triangle.vertex_degree("a") == 2


def test_vertex_neighbors_missing_vertex_raises(triangle):
    with pytest.raises(VertexNotExistsException):
        triangle.vertex_neighbors("ghost")


def test_is_vertices_adjacents_true_and_false(path):
    assert path.is_vertices_adjacents("a", "b") is True
    assert path.is_vertices_adjacents("a", "c") is False


def test_adjacency_matrix_is_symmetric_with_edge_counts(path):
    matrix = path.adjacency_matrix()
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["a"] == 1
    assert matrix["b"]["c"] == 1
    assert matrix["c"]["b"] == 1
    assert matrix["a"]["c"] == 0
    assert matrix["a"]["a"] == 0


def test_has_cycle_detects_triangle_and_ignores_path(triangle, path):
    assert triangle.has_cycle() is True
    assert path.has_cycle() is False


def test_has_loop_is_false_for_loopless_graph(path):
    assert path.has_loop() is False


def test_has_loop_detects_self_edge_in_base_graph(graph):
    graph.add_vertex("a")
    graph.add_edge("a", "a", "self")
    assert graph.has_loop() is True


def test_str_of_empty_graph_is_empty_string(graph):
    assert str(graph) == ""


# ---------- SimpleGraph ----------

def test_simple_graph_is_a_graph_subclass(simple_graph):
    assert isinstance(simple_graph, Graph)


def test_simple_graph_add_edge_rejects_loop(simple_graph):
    simple_graph.add_vertex("a")
    with pytest.raises(LoopDetectedException):
        simple_graph.add_edge("a", "a", "loop")
    assert simple_graph.num_edges() == 0


def test_simple_graph_add_edge_with_missing_vertex_raises(simple_graph):
    simple_graph.add_vertex("a")
    with pytest.raises(VertexNotExistsException):
        simple_graph.add_edge("a", "b", "ab")
    assert simple_graph.num_edges() == 0


def test_simple_graph_add_duplicate_edge_raises(simple_graph):
    simple_graph.add_vertex("a")
    simple_graph.add_vertex("b")
    simple_graph.add_edge("a", "b", "ab")
    with pytest.raises(EdgeDuplicatedException):
        simple_graph.add_edge("a", "b", "ab2")
    assert simple_graph.num_edges() == 1


def test_simple_graph_add_edge_returns_created_edge(simple_graph):
    simple_graph.add_vertex("a")
    simple_graph.add_vertex("b")
    edge = simple_graph.add_edge("a", "b", "ab")
    assert edge.get_name() == "ab"
    assert simple_graph.edge_exists("a", "b") is True
