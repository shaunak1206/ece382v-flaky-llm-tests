import pytest

from py_graph_t.SimpleGraph import SimpleGraph
from py_graph_t.Graph import Graph
from py_graph_t.exceptions.SimpleGraphException import (
    VertexNotExistsException,
    EdgeDuplicatedException,
    LoopDetectedException,
    VertexDuplicatedException,
    EdgeNotFoundException,
    EdgeNameExistsException,
)


@pytest.fixture
def simple_graph():
    g = SimpleGraph()
    return g


@pytest.fixture
def base_graph():
    g = Graph()
    return g


def test_add_vertex(simple_graph):
    v = simple_graph.add_vertex("A")
    assert v.value == "A"
    assert simple_graph.vertex_exists("A")
    assert simple_graph.num_vertex() == 1


def test_add_duplicate_vertex_raises(simple_graph):
    simple_graph.add_vertex("A")
    with pytest.raises(VertexDuplicatedException):
        simple_graph.add_vertex("A")


def test_add_edge_simplegraph_and_edge_exists(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    edge = simple_graph.add_edge("A", "B", name="e1")
    assert simple_graph.edge_exists("A", "B")
    assert edge.name == "e1"

    # Check that edge is returned by show_edge
    edge_shown = simple_graph.show_edge("A", "B")
    assert edge_shown == edge


def test_add_edge_loop_raises(simple_graph):
    simple_graph.add_vertex("A")
    with pytest.raises(LoopDetectedException):
        simple_graph.add_edge("A", "A", name="loopedge")


def test_add_edge_vertex_not_exists_raises(simple_graph):
    simple_graph.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        simple_graph.add_edge("A", "B", name="e1")
    with pytest.raises(VertexNotExistsException):
        simple_graph.add_edge("B", "A", name="e2")


def test_add_duplicate_edge_raises(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_edge("A", "B", name="e1")
    with pytest.raises(EdgeDuplicatedException):
        simple_graph.add_edge("A", "B", name="e1")
    with pytest.raises(EdgeDuplicatedException):
        simple_graph.add_edge("B", "A", name="e1")


def test_delete_vertex_removes_edges(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    simple_graph.add_vertex("C")
    simple_graph.add_edge("A", "B", name="e1")
    simple_graph.add_edge("B", "C", name="e2")
    assert simple_graph.num_edges() == 2
    simple_graph.delete_vertex("B")
    assert not simple_graph.vertex_exists("B")
    assert simple_graph.num_edges() == 0
    assert simple_graph.num_vertex() == 2


def test_delete_nonexistent_vertex_raises(simple_graph):
    with pytest.raises(VertexNotExistsException):
        simple_graph.delete_vertex("X")


def test_delete_edge(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_edge("A", "B", name="e1")
    assert base_graph.edge_exists("A", "B")
    deleted_edge = base_graph.delete_edge("A", "B")
    assert deleted_edge.name == "e1"
    assert not base_graph.edge_exists("A", "B")


def test_delete_nonexistent_edge_raises(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    with pytest.raises(EdgeNotFoundException):
        base_graph.delete_edge("A", "B")


def test_edge_name_exists(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_edge("A", "B", name="e1")
    assert base_graph.edge_name_exists("e1")
    assert not base_graph.edge_name_exists("eX")


def test_vertex_neighbors_and_degree(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_vertex("C")
    base_graph.add_edge("A", "B", name="e1")
    base_graph.add_edge("A", "C", name="e2")
    neighbors = base_graph.vertex_neighbors("A")
    neighbor_values = {v.value for v in neighbors}
    assert neighbor_values == {"B", "C"}
    degree = base_graph.vertex_degree("A")
    assert degree == 2

    with pytest.raises(VertexNotExistsException):
        base_graph.vertex_neighbors("X")
    with pytest.raises(VertexNotExistsException):
        base_graph.vertex_degree("X")


def test_is_vertices_adjacents(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_vertex("C")
    base_graph.add_edge("A", "B", name="e1")
    assert base_graph.is_vertices_adjacents("A", "B")
    assert base_graph.is_vertices_adjacents("B", "A")
    assert not base_graph.is_vertices_adjacents("A", "C")


def test_get_all_and_list_graph_vertices_edges(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_edge("A", "B", name="e1")
    vertex_dict = base_graph.get_all_vertex()
    assert "A" in vertex_dict and "B" in vertex_dict
    verts = base_graph.list_graph_vertices()
    assert set(verts) == {"A", "B"}
    edges = base_graph.list_graph_edges()
    assert edges == ["e1"]


def test_has_cycle_and_has_loop(base_graph):
    # Create a cycle graph A-B-C-A
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_vertex("C")
    base_graph.add_edge("A", "B", name="e1")
    base_graph.add_edge("B", "C", name="e2")
    base_graph.add_edge("C", "A", name="e3")
    assert base_graph.has_cycle() is True
    assert base_graph.has_loop() is False

    # Add loop edge and test has_loop True
    base_graph.add_edge("A", "A", name="loopedge")  # Graph allows loops here
    assert base_graph.has_loop() is True

    # For SimpleGraph, adding a loop should raise LoopDetectedException
    sg = SimpleGraph()
    sg.add_vertex("X")
    with pytest.raises(LoopDetectedException):
        sg.add_edge("X", "X", name="loopedge")


def test_check_regular_graph(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_vertex("C")
    # no edges so regular
    assert base_graph.check_regular_graph() is True
    base_graph.add_edge("A", "B", name="e1")
    base_graph.add_edge("B", "C", name="e2")
    base_graph.add_edge("C", "A", name="e3")
    # every vertex degree 2, regular
    assert base_graph.check_regular_graph() is True
    base_graph.add_vertex("D")
    base_graph.add_edge("A", "D", name="e4")
    # degrees now differ, not regular
    assert base_graph.check_regular_graph() is False


def test_incidence_list(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_edge("A", "B", name="e1")
    inc_list = base_graph.incidence_list()
    # Should have 2 vertices * 1 edge => 2 or 4 entries? The code produces one per vertex per edge
    # For each vertex-edge pair we get a ValueBinding with incidence count 1 if edge incident to vertex
    values = {(vb.vertex_value, vb.edge_name, vb.incidence) for vb in inc_list}
    assert ("A", "e1", 1) in values
    assert ("B", "e1", 1) in values


def test_adjacency_matrix(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_vertex("C")
    base_graph.add_edge("A", "B", name="e1")
    base_graph.add_edge("B", "C", name="e2")
    matrix = base_graph.adjacency_matrix()
    assert matrix["A"]["A"] == 0
    assert matrix["A"]["B"] == 1
    assert matrix["B"]["A"] == 1
    assert matrix["B"]["C"] == 1
    assert matrix["C"]["B"] == 1
    assert matrix["A"]["C"] == 0


def test_str_representation(base_graph):
    base_graph.add_vertex("A")
    base_graph.add_vertex("B")
    base_graph.add_edge("A", "B", name="e1")
    s = str(base_graph)
    assert "e1" in s
    # Should contain 1 line (1 edge)
    lines = s.strip().split("\n")
    assert len(lines) == 1


def test_simplegraph_add_edge_returns_same_as_show_edge(simple_graph):
    simple_graph.add_vertex("A")
    simple_graph.add_vertex("B")
    returned_edge = simple_graph.add_edge("A", "B", name="e1")
    shown_edge = simple_graph.show_edge("A", "B")
    assert returned_edge == shown_edge
