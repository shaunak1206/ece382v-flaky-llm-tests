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


def make_graph(cls, values):
    g = cls()
    for v in values:
        g.add_vertex(v)
    return g


def test_add_vertex_and_counts():
    g = Graph()
    vertex = g.add_vertex(1)
    assert vertex.value == 1
    g.add_vertex(2)
    assert g.num_vertex() == 2
    assert g.vertex_exists(1)
    assert not g.vertex_exists(5)
    assert sorted(g.list_graph_vertices()) == [1, 2]
    assert set(g.get_all_vertex().keys()) == {1, 2}


def test_add_duplicate_vertex_raises():
    g = make_graph(Graph, [1])
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex(1)
    assert g.num_vertex() == 1


def test_delete_vertex_returns_vertex_and_removes_it():
    g = make_graph(Graph, [1, 2])
    removed = g.delete_vertex(1)
    assert removed.value == 1
    assert not g.vertex_exists(1)
    assert g.num_vertex() == 1


def test_delete_vertex_removes_incident_edges():
    g = make_graph(Graph, [1, 2, 3])
    g.add_edge(1, 2, "e1")
    g.add_edge(2, 3, "e2")
    g.delete_vertex(2)
    assert g.num_edges() == 0
    assert g.num_vertex() == 2


def test_delete_vertex_keeps_unrelated_edges():
    g = make_graph(Graph, [1, 2, 3, 4])
    g.add_edge(1, 2, "e1")
    g.add_edge(3, 4, "e2")
    g.delete_vertex(1)
    assert g.num_edges() == 1
    assert g.list_graph_edges() == ["e2"]


def test_delete_missing_vertex_raises():
    g = Graph()
    with pytest.raises(VertexNotExistsException):
        g.delete_vertex(42)


def test_graph_add_edge_and_show_edge():
    g = make_graph(Graph, [1, 2])
    edge = g.add_edge(1, 2, "e1")
    assert edge.get_name() == "e1"
    assert g.num_edges() == 1
    assert g.edge_exists(1, 2)
    assert g.show_edge(1, 2).get_name() == "e1"
    assert g.list_graph_edges() == ["e1"]


def test_graph_add_edge_missing_vertex_raises():
    g = make_graph(Graph, [1])
    with pytest.raises(VertexNotExistsException):
        g.add_edge(1, 2, "e1")
    assert g.num_edges() == 0


def test_graph_add_edge_duplicate_name_raises():
    g = make_graph(Graph, [1, 2, 3])
    g.add_edge(1, 2, "same")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge(2, 3, "same")
    assert g.num_edges() == 1


def test_show_edge_errors():
    g = make_graph(Graph, [1, 2, 3])
    g.add_edge(1, 2, "e1")
    with pytest.raises(EdgeNotFoundException):
        g.show_edge(1, 3)
    with pytest.raises(VertexNotExistsException):
        g.show_edge(1, 99)


def test_delete_edge():
    g = make_graph(Graph, [1, 2, 3])
    g.add_edge(1, 2, "e1")
    removed = g.delete_edge(1, 2)
    assert removed.get_name() == "e1"
    assert g.num_edges() == 0
    assert not g.edge_exists(1, 2)
    with pytest.raises(EdgeNotFoundException):
        g.delete_edge(1, 3)


def test_neighbors_degree_and_adjacency():
    g = make_graph(SimpleGraph, [1, 2, 3, 4])
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    neighbors = g.vertex_neighbors(1)
    assert sorted(v.value for v in neighbors) == [2, 3]
    assert g.vertex_degree(1) == 2
    assert g.vertex_degree(4) == 0
    assert g.is_vertices_adjacents(1, 2)
    assert not g.is_vertices_adjacents(1, 4)
    with pytest.raises(VertexNotExistsException):
        g.vertex_neighbors(99)
    with pytest.raises(VertexNotExistsException):
        g.vertex_degree(99)


def test_simple_graph_add_edge_success():
    g = make_graph(SimpleGraph, [1, 2])
    edge = g.add_edge(1, 2)
    assert edge is not None
    assert g.num_edges() == 1
    assert g.edge_exists(1, 2)


def test_simple_graph_rejects_loop():
    g = make_graph(SimpleGraph, [1])
    with pytest.raises(LoopDetectedException):
        g.add_edge(1, 1)
    assert g.num_edges() == 0
    assert not g.has_loop()


def test_simple_graph_rejects_missing_vertex():
    g = make_graph(SimpleGraph, [1])
    with pytest.raises(VertexNotExistsException):
        g.add_edge(1, 2)
    assert g.num_edges() == 0


def test_simple_graph_rejects_duplicate_edge():
    g = make_graph(SimpleGraph, [1, 2])
    g.add_edge(1, 2)
    with pytest.raises(EdgeDuplicatedException):
        g.add_edge(1, 2)
    assert g.num_edges() == 1


def test_has_cycle_triangle_vs_path():
    triangle = make_graph(SimpleGraph, [1, 2, 3])
    triangle.add_edge(1, 2)
    triangle.add_edge(2, 3)
    triangle.add_edge(3, 1)
    assert triangle.has_cycle() is True

    path = make_graph(SimpleGraph, [1, 2, 3])
    path.add_edge(1, 2)
    path.add_edge(2, 3)
    assert path.has_cycle() is False


def test_check_regular_graph():
    empty = Graph()
    assert empty.check_regular_graph() is True

    triangle = make_graph(SimpleGraph, [1, 2, 3])
    triangle.add_edge(1, 2)
    triangle.add_edge(2, 3)
    triangle.add_edge(3, 1)
    assert triangle.check_regular_graph() is True

    path = make_graph(SimpleGraph, [1, 2, 3])
    path.add_edge(1, 2)
    path.add_edge(2, 3)
    assert path.check_regular_graph() is False


def test_adjacency_matrix():
    g = make_graph(SimpleGraph, [1, 2, 3])
    g.add_edge(1, 2)
    matrix = g.adjacency_matrix()
    assert matrix[1][2] == 1
    assert matrix[2][1] == 1
    assert matrix[1][3] == 0
    assert matrix[3][3] == 0
    assert set(matrix.keys()) == {1, 2, 3}


def test_graph_has_loop_when_loop_edge_added():
    g = make_graph(Graph, [1, 2])
    assert g.has_loop() is False
    g.add_edge(1, 1, "loop")
    assert g.has_loop() is True


def test_incidence_list_size_and_str():
    g = make_graph(SimpleGraph, [1, 2, 3])
    assert str(g) == ""
    g.add_edge(1, 2, "a")
    g.add_edge(2, 3, "b")
    assert len(g.incidence_list()) == 3 * 2
    text = str(g)
    assert text.endswith("\n")
    assert len(text.strip().split("\n")) == 2
