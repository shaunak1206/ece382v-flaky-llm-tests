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
def triangle():
    """Simple graph with vertices 1, 2, 3 connected as a triangle."""
    g = SimpleGraph()
    for v in (1, 2, 3):
        g.add_vertex(v)
    g.add_edge(1, 2, "a")
    g.add_edge(2, 3, "b")
    g.add_edge(3, 1, "c")
    return g


@pytest.fixture
def path():
    """Simple graph with vertices 1-2-3 forming a path."""
    g = SimpleGraph()
    for v in (1, 2, 3):
        g.add_vertex(v)
    g.add_edge(1, 2, "e1")
    g.add_edge(2, 3, "e2")
    return g


def test_simple_graph_is_a_graph_subclass():
    assert issubclass(SimpleGraph, Graph)


def test_new_graph_is_empty():
    g = SimpleGraph()
    assert g.num_vertex() == 0
    assert g.num_edges() == 0
    assert g.get_all_vertex() == {}
    assert str(g) == ""


def test_add_vertex_returns_vertex_and_registers_it():
    g = SimpleGraph()
    vertex = g.add_vertex("x")
    assert vertex is not None
    assert g.vertex_exists("x")
    assert g.num_vertex() == 1


def test_add_duplicate_vertex_raises():
    g = SimpleGraph()
    g.add_vertex(1)
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex(1)
    assert g.num_vertex() == 1


def test_add_edge_creates_edge_between_existing_vertices():
    g = SimpleGraph()
    g.add_vertex(1)
    g.add_vertex(2)
    edge = g.add_edge(1, 2, "e")
    assert edge is not None
    assert g.num_edges() == 1
    assert g.edge_exists(1, 2)


def test_add_edge_with_missing_vertex_raises():
    g = SimpleGraph()
    g.add_vertex(1)
    with pytest.raises(VertexNotExistsException):
        g.add_edge(1, 99, "e")
    assert g.num_edges() == 0


def test_add_loop_is_rejected_by_simple_graph():
    g = SimpleGraph()
    g.add_vertex(1)
    with pytest.raises(LoopDetectedException):
        g.add_edge(1, 1, "loop")
    assert g.num_edges() == 0


def test_add_parallel_edge_is_rejected_by_simple_graph(path):
    with pytest.raises(EdgeDuplicatedException):
        path.add_edge(1, 2, "parallel")
    assert path.num_edges() == 2


def test_base_graph_rejects_duplicate_edge_name():
    g = Graph()
    g.add_vertex(1)
    g.add_vertex(2)
    g.add_vertex(3)
    g.add_edge(1, 2, "same")
    with pytest.raises(EdgeNameExistsException):
        g.add_edge(2, 3, "same")
    assert g.edge_name_exists("same")
    assert not g.edge_name_exists("other")


def test_vertex_neighbors_and_degree(triangle):
    neighbors = sorted(v.get_value() for v in triangle.vertex_neighbors(1))
    assert neighbors == [2, 3]
    assert triangle.vertex_degree(1) == 2


def test_vertex_neighbors_of_missing_vertex_raises(triangle):
    with pytest.raises(VertexNotExistsException):
        triangle.vertex_neighbors(42)
    with pytest.raises(VertexNotExistsException):
        triangle.vertex_degree(42)


def test_is_vertices_adjacents(path):
    assert path.is_vertices_adjacents(1, 2) is True
    assert path.is_vertices_adjacents(1, 3) is False


def test_has_cycle_detects_triangle(triangle):
    assert triangle.has_cycle() is True


def test_has_cycle_is_false_for_path(path):
    assert path.has_cycle() is False


def test_has_loop_is_false_for_simple_graph(triangle):
    assert triangle.has_loop() is False


def test_check_regular_graph_on_triangle_and_path(triangle, path):
    assert triangle.check_regular_graph() is True
    assert path.check_regular_graph() is False


def test_adjacency_matrix_is_symmetric_with_zero_diagonal(path):
    matrix = path.adjacency_matrix()
    assert matrix[1][2] == 1
    assert matrix[2][1] == 1
    assert matrix[1][3] == 0
    assert matrix[2][3] == 1
    assert matrix[3][3] == 0


def test_list_graph_vertices_and_edges(triangle):
    assert sorted(triangle.list_graph_vertices()) == [1, 2, 3]
    assert sorted(triangle.list_graph_edges()) == ["a", "b", "c"]


def test_delete_edge_removes_and_returns_edge(triangle):
    before = triangle.num_edges()
    removed = triangle.delete_edge(1, 2)
    assert removed is not None
    assert triangle.num_edges() == before - 1
    assert not triangle.edge_exists(1, 2)


def test_delete_missing_edge_raises(path):
    with pytest.raises(EdgeNotFoundException):
        path.delete_edge(1, 3)


def test_delete_vertex_removes_incident_edges(triangle):
    removed = triangle.delete_vertex(1)
    assert removed is not None
    assert not triangle.vertex_exists(1)
    assert triangle.num_vertex() == 2
    assert triangle.num_edges() == 1
    assert triangle.edge_exists(2, 3)


def test_delete_missing_vertex_raises(triangle):
    with pytest.raises(VertexNotExistsException):
        triangle.delete_vertex(99)


def test_show_edge_returns_existing_edge_and_raises_when_absent(path):
    edge = path.show_edge(1, 2)
    assert edge is not None
    with pytest.raises(EdgeNotFoundException):
        path.show_edge(1, 3)
    with pytest.raises(VertexNotExistsException):
        path.show_edge(1, 99)


def test_str_has_one_line_per_edge(path):
    text = str(path)
    assert text.count("\n") == path.num_edges()
