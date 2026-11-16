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


def make_simple(values=(1, 2, 3)):
    g = SimpleGraph()
    for v in values:
        g.add_vertex(v)
    return g


def make_triangle():
    g = make_simple((1, 2, 3))
    g.add_edge(1, 2, "a")
    g.add_edge(2, 3, "b")
    g.add_edge(1, 3, "c")
    return g


def test_add_vertex_returns_vertex_and_counts():
    g = Graph()
    vertex = g.add_vertex("x")
    assert vertex.value == "x"
    assert g.num_vertex() == 1
    assert g.vertex_exists("x")
    assert not g.vertex_exists("y")


def test_add_duplicate_vertex_raises():
    g = Graph()
    g.add_vertex(1)
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex(1)
    assert g.num_vertex() == 1


def test_get_all_vertex_and_list_vertices():
    g = make_simple((1, 2, 3))
    assert set(g.get_all_vertex().keys()) == {1, 2, 3}
    assert sorted(g.list_graph_vertices()) == [1, 2, 3]


def test_delete_vertex_removes_incident_edges():
    g = make_triangle()
    removed = g.delete_vertex(2)
    assert removed.value == 2
    assert not g.vertex_exists(2)
    assert g.num_vertex() == 2
    assert g.num_edges() == 1
    assert g.edge_exists(1, 3)


def test_delete_missing_vertex_raises():
    g = make_simple((1,))
    with pytest.raises(VertexNotExistsException):
        g.delete_vertex(99)


def test_simple_graph_add_edge_returns_edge_and_registers():
    g = make_simple((1, 2))
    edge = g.add_edge(1, 2, "e1")
    assert edge is not None
    assert g.num_edges() == 1
    assert g.edge_exists(1, 2)
    assert g.show_edge(1, 2) is edge


def test_simple_graph_add_edge_missing_vertex_raises():
    g = make_simple((1, 2))
    with pytest.raises(VertexNotExistsException):
        g.add_edge(1, 99, "e1")
    with pytest.raises(VertexNotExistsException):
        g.add_edge(99, 2, "e1")
    assert g.num_edges() == 0


def test_simple_graph_rejects_loop():
    g = make_simple((1, 2))
    with pytest.raises(LoopDetectedException):
        g.add_edge(1, 1, "loop")
    assert g.num_edges() == 0


def test_simple_graph_rejects_parallel_edge():
    g = make_simple((1, 2))
    g.add_edge(1, 2, "e1")
    with pytest.raises(EdgeDuplicatedException):
        g.add_edge(1, 2, "e2")
    assert g.num_edges() == 1


def test_base_graph_duplicate_edge_name_raises():
    g = Graph()
    g.add_vertex(1)
    g.add_vertex(2)
    g.add_vertex(3)
    g.add_edge(1, 2, "same")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge(2, 3, "same")
    assert g.num_edges() == 1


def test_base_graph_add_edge_missing_vertex_raises():
    g = Graph()
    g.add_vertex(1)
    with pytest.raises(VertexNotExistsException):
        g.add_edge(1, 2, "e")


def test_show_edge_errors():
    g = make_simple((1, 2, 3))
    g.add_edge(1, 2, "e1")
    with pytest.raises(EdgeNotFoundException):
        g.show_edge(1, 3)
    with pytest.raises(VertexNotExistsException):
        g.show_edge(1, 99)


def test_delete_edge():
    g = make_simple((1, 2))
    edge = g.add_edge(1, 2, "e1")
    removed = g.delete_edge(1, 2)
    assert removed is edge
    assert g.num_edges() == 0
    assert not g.edge_exists(1, 2)


def test_delete_missing_edge_raises():
    g = make_simple((1, 2))
    with pytest.raises(EdgeNotFoundException):
        g.delete_edge(1, 2)


def test_neighbors_degree_and_adjacency():
    g = make_simple((1, 2, 3, 4))
    g.add_edge(1, 2, "a")
    g.add_edge(1, 3, "b")
    neighbors = g.vertex_neighbors(1)
    assert {v.value for v in neighbors} == {2, 3}
    assert g.vertex_degree(1) == 2
    assert g.vertex_degree(4) == 0
    assert g.is_vertices_adjacents(1, 2) is True
    assert g.is_vertices_adjacents(1, 4) is False
    with pytest.raises(VertexNotExistsException):
        g.vertex_neighbors(99)
    with pytest.raises(VertexNotExistsException):
        g.vertex_degree(99)


def test_is_terminal():
    g = make_simple((1, 2, 3))
    edge = g.add_edge(1, 2, "a")
    assert g.is_terminal(edge, 1) is True
    assert g.is_terminal(edge, 2) is True
    assert g.is_terminal(edge, 3) is False


def test_has_cycle_triangle_vs_path():
    assert make_triangle().has_cycle() is True

    path = make_simple((1, 2, 3))
    path.add_edge(1, 2, "a")
    path.add_edge(2, 3, "b")
    assert path.has_cycle() is False


def test_has_loop_false_for_simple_graph():
    assert make_triangle().has_loop() is False
    assert SimpleGraph().has_loop() is False


def test_check_regular_graph():
    assert make_triangle().check_regular_graph() is True
    assert SimpleGraph().check_regular_graph() is True

    path = make_simple((1, 2, 3))
    path.add_edge(1, 2, "a")
    path.add_edge(2, 3, "b")
    assert path.check_regular_graph() is False


def test_adjacency_matrix():
    g = make_simple((1, 2, 3))
    g.add_edge(1, 2, "a")
    matrix = g.adjacency_matrix()
    assert matrix[1][2] == 1
    assert matrix[2][1] == 1
    assert matrix[1][1] == 0
    assert matrix[1][3] == 0
    assert matrix[3][2] == 0


def test_list_edges_and_incidence_list_size():
    g = make_triangle()
    assert sorted(g.list_graph_edges()) == ["a", "b", "c"]
    assert len(g.incidence_list()) == 3 * 3


def test_str_representation():
    assert str(SimpleGraph()) == ""
    g = make_simple((1, 2))
    g.add_edge(1, 2, "a")
    text = str(g)
    assert text.endswith("\n")
    assert len(text.strip()) > 0
