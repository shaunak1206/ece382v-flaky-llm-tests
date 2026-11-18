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


def test_add_and_get_vertex():
    g = Graph()
    v = g.add_vertex("A")
    assert v.value == "A"
    assert g.vertex_exists("A")
    assert g.num_vertex() == 1


def test_add_duplicate_vertex_raises():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex("A")


def test_delete_existing_vertex_removes_edges():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="edge1")
    assert g.num_edges() == 1
    removed_vertex = g.delete_vertex("A")
    assert removed_vertex.value == "A"
    assert not g.vertex_exists("A")
    assert g.num_edges() == 0


def test_delete_nonexistent_vertex_raises():
    g = Graph()
    with pytest.raises(VertexNotExistsException):
        g.delete_vertex("X")


def test_add_edge_and_show_edge():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    edge = g.add_edge("A", "B", name="edge1")
    assert edge.name == "edge1"
    assert g.edge_exists("A", "B")
    # show_edge should return the same edge
    edge2 = g.show_edge("A", "B")
    assert edge == edge2


def test_add_edge_with_nonexistent_vertex_raises():
    g = Graph()
    g.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        g.add_edge("A", "X", name="edge1")


def test_add_edge_duplicate_name_raises():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="edge1")
    g.add_vertex("C")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge("B", "C", name="edge1")


def test_delete_existing_edge():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="edge1")
    removed_edge = g.delete_edge("A", "B")
    assert removed_edge.name == "edge1"
    assert not g.edge_exists("A", "B")


def test_delete_nonexistent_edge_raises():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    with pytest.raises(EdgeNotFoundException):
        g.delete_edge("A", "B")


def test_vertex_neighbors_and_degree():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")
    g.add_edge("A", "B", name="e1")
    g.add_edge("A", "C", name="e2")
    neighbors = g.vertex_neighbors("A")
    neighbor_values = {v.value for v in neighbors}
    assert neighbor_values == {"B", "C"}
    degree = g.vertex_degree("A")
    assert degree == 2


def test_vertex_neighbors_nonexistent_vertex_raises():
    g = Graph()
    with pytest.raises(VertexNotExistsException):
        g.vertex_neighbors("X")


def test_is_vertices_adjacents_true_false():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")
    g.add_edge("A", "B", name="e1")
    assert g.is_vertices_adjacents("A", "B") is True
    assert g.is_vertices_adjacents("A", "C") is False


def test_list_graph_vertices_and_edges():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="edge1")
    vertices_list = g.list_graph_vertices()
    assert set(vertices_list) == {"A", "B"}
    edges_list = g.list_graph_edges()
    assert edges_list == ["edge1"]


def test_has_cycle_detected_and_no_cycle():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")

    g.add_edge("A", "B", name="e1")
    g.add_edge("B", "C", name="e2")

    assert g.has_cycle() is False

    # adding edge to create cycle
    g.add_edge("C", "A", name="e3")
    assert g.has_cycle() is True


def test_has_loop_returns_false_for_simple_graph():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="e1")
    assert g.has_loop() is False


def test_simplegraph_add_edge_prevents_loops():
    sg = SimpleGraph()
    sg.add_vertex("A")
    with pytest.raises(LoopDetectedException):
        sg.add_edge("A", "A", name="loop_edge")


def test_simplegraph_add_edge_raises_if_vertex_not_exists():
    sg = SimpleGraph()
    sg.add_vertex("A")
    with pytest.raises(VertexNotExistsException):
        sg.add_edge("A", "B", name="e1")


def test_simplegraph_add_edge_prevents_duplicate_edges():
    sg = SimpleGraph()
    sg.add_vertex("A")
    sg.add_vertex("B")
    sg.add_edge("A", "B", name="e1")
    with pytest.raises(EdgeDuplicatedException):
        sg.add_edge("A", "B", name="e2")


def test_adjacency_matrix_contents():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")
    g.add_edge("A", "B", name="e1")
    matrix = g.adjacency_matrix()
    assert matrix["A"]["B"] == 1
    assert matrix["B"]["A"] == 1
    assert matrix["A"]["C"] == 0
    assert matrix["C"]["A"] == 0


def test_check_regular_graph_true_false():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")
    # All vertices degree 2 (triangle)
    g.add_edge("A", "B", name="e1")
    g.add_edge("B", "C", name="e2")
    g.add_edge("C", "A", name="e3")
    assert g.check_regular_graph() is True

    g2 = Graph()
    g2.add_vertex("A")
    g2.add_vertex("B")
    g2.add_vertex("C")
    g2.add_edge("A", "B", name="e1")
    g2.add_edge("B", "C", name="e2")
    # Vertex C has degree 1, A has degree 1, B has degree 2: not regular
    assert g2.check_regular_graph() is False


def test_incidence_list_produces_entries():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="e1")
    incidence = g.incidence_list()
    # There should be 2 vertices x 1 edge = 2 entries
    assert len(incidence) == 2
    vals = {binding.vertex for binding in incidence}
    assert vals == {"A", "B"}
    # incidence values should be 1 for each since edge connects different vertices
    values = {binding.value for binding in incidence}
    assert "e1" in values
    for b in incidence:
        assert b.binding in (0, 1, 2)


def test_graph_str_contains_edges():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="e1")
    s = str(g)
    assert "e1" in s

