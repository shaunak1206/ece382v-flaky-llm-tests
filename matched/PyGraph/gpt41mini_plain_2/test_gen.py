import pytest

from py_graph_t.SimpleGraph import SimpleGraph
from py_graph_t.Graph import Graph
from py_graph_t.exceptions.SimpleGraphException import (
    VertexNotExistsException,
    EdgeDuplicatedException,
    LoopDetectedException,
    VertexDuplicatedException,
    EdgeNotFoundException,
    EdgeNameExistsException
)


@pytest.fixture
def simple_graph():
    g = SimpleGraph()
    return g

@pytest.fixture
def base_graph():
    return Graph()


def test_add_vertex_success(simple_graph):
    v = simple_graph.add_vertex("A")
    assert v.value == "A"
    assert "A" in simple_graph.vertices

def test_add_vertex_duplicate_raises(simple_graph):
    simple_graph.add_vertex("A")
    with pytest.raises(VertexDuplicatedException):
        simple_graph.add_vertex("A")

def test_add_edge_success(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    edge = simple_graph.add_edge("A", "B", name="edge1")
    assert edge.name == "edge1"
    assert edge.vertex_a.value == "A"
    assert edge.vertex_b.value == "B"
    assert simple_graph.edge_exists("A", "B")

def test_add_edge_loop_raises(simple_graph):
    simple_graph.add_vertex("A")
    with pytest.raises(LoopDetectedException):
        simple_graph.add_edge("A", "A", name="loop_edge")

def test_add_edge_vertex_not_exists_raises(simple_graph):
    simple_graph.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        simple_graph.add_edge("A", "B", name="missing_vertex_edge")

def test_add_edge_duplicate_raises(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_edge("A", "B", name="edge1")
    with pytest.raises(EdgeDuplicatedException):
        simple_graph.add_edge("A", "B", name="edge2")

def test_delete_vertex_removes_edges(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_vertex("C")
    simple_graph.add_edge("A", "B", name="edgeAB")
    simple_graph.add_edge("A", "C", name="edgeAC")
    assert simple_graph.num_edges() == 2
    removed = simple_graph.delete_vertex("A")
    assert removed.value == "A"
    assert "A" not in simple_graph.vertices
    # all edges involving A removed
    assert simple_graph.num_edges() == 0

def test_delete_vertex_not_exists_raises(simple_graph):
    with pytest.raises(VertexNotExistsException):
        simple_graph.delete_vertex("X")

def test_delete_edge_success(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_edge("A", "B", name="edgeAB")
    deleted_edge = simple_graph.delete_edge("A", "B")
    assert deleted_edge.name == "edgeAB"
    assert not simple_graph.edge_exists("A", "B")

def test_delete_edge_not_found_raises(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    with pytest.raises(EdgeNotFoundException):
        simple_graph.delete_edge("A", "B")

def test_vertex_neighbors(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_vertex("C")
    simple_graph.add_edge("A", "B", name="edgeAB")
    simple_graph.add_edge("B", "C", name="edgeBC")
    neighbors_of_B = simple_graph.vertex_neighbors("B")
    neighbor_values = {v.value for v in neighbors_of_B}
    assert neighbor_values == {"A", "C"}

def test_vertex_neighbors_nonexistent_vertex_raises(simple_graph):
    with pytest.raises(VertexNotExistsException):
        simple_graph.vertex_neighbors("X")

def test_vertex_degree(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_vertex("C")
    simple_graph.add_edge("A", "B", name="edgeAB")
    simple_graph.add_edge("A", "C", name="edgeAC")
    degree = simple_graph.vertex_degree("A")
    assert degree == 2

def test_vertex_degree_nonexistent_raises(simple_graph):
    with pytest.raises(VertexNotExistsException):
        simple_graph.vertex_degree("Z")

def test_is_vertices_adjacents(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_edge("A", "B", name="edgeAB")
    assert simple_graph.is_vertices_adjacents("A", "B") is True
    assert simple_graph.is_vertices_adjacents("B", "A") is True
    simple_graph.add_vertex("C")
    assert simple_graph.is_vertices_adjacents("A", "C") is False

def test_has_cycle_false(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_vertex("C")
    simple_graph.add_edge("A", "B", name="edgeAB")
    simple_graph.add_edge("B", "C", name="edgeBC")
    assert simple_graph.has_cycle() is False

def test_has_cycle_true(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_vertex("C")
    simple_graph.add_edge("A", "B", name="edgeAB")
    simple_graph.add_edge("B", "C", name="edgeBC")
    simple_graph.add_edge("C", "A", name="edgeCA")
    assert simple_graph.has_cycle() is True

def test_has_loop_false(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_edge("A", "B", name="edgeAB")
    assert simple_graph.has_loop() is False

def test_adjacency_matrix(simple_graph):
    simple_graph.add_vertex("X")
    simple_graph.add_vertex("Y")
    simple_graph.add_edge("X", "Y", name="edgeXY")
    mat = simple_graph.adjacency_matrix()
    assert mat["X"]["Y"] == 1
    assert mat["Y"]["X"] == 1
    assert mat["X"]["X"] == 0
    assert mat["Y"]["Y"] == 0

def test_str_representation_contains_edges(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_edge("A", "B", name="edgeAB")
    s = str(simple_graph)
    assert "edgeAB" in s
    assert "A" in s or "B" in s  # edges __str__ likely shows vertices

def test_edge_name_exists_raises_in_base_graph(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_edge("A", "B", name="edge1")
    with pytest.raises(EdgeNameExistsException):
        base_graph.add_edge("B", "A", name="edge1")
