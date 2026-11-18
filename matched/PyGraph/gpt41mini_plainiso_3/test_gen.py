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


def test_add_vertex_simplegraph_and_graph():
    g = SimpleGraph()
    v = g.add_vertex("A")
    assert v.value == "A"
    assert g.vertex_exists("A")
    # Duplicate vertex raises
    with pytest.raises(VertexDuplicatedException):
        g.add_vertex("A")


def test_delete_vertex_removes_edges():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B")
    assert g.num_edges() == 1
    removed_vertex = g.delete_vertex("A")
    assert removed_vertex.value == "A"
    # Edge connected to A was removed
    assert g.num_edges() == 0
    with pytest.raises(VertexNotExistsException):
        g.delete_vertex("A")


def test_add_edge_simplegraph_no_loops_no_parallel_edges():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")

    # Add edge works
    e = g.add_edge("A", "B", name="e1")
    assert e.vertex_a.value == "A"
    assert e.vertex_b.value == "B"
    assert e.name == "e1"

    # Adding loop edge raises LoopDetectedException
    with pytest.raises(LoopDetectedException):
        g.add_edge("A", "A", name="loop")

    # Adding edge with non-existent vertex raises VertexNotExistsException
    with pytest.raises(VertexNotExistsException):
        g.add_edge("A", "C", name="e2")

    # Adding duplicate edge raises EdgeDuplicatedException
    with pytest.raises(EdgeDuplicatedException):
        g.add_edge("B", "A", name="e3")  # since edge is undirected


def test_add_edge_graph_allows_parallel_edges_by_name_conflict():
    g = Graph()
    g.add_vertex("A")
    g.add_vertex("B")

    e1 = g.add_edge("A", "B", name="e1")
    assert e1.name == "e1"

    # Adding edge with existing name raises EdgeNameExistsException
    with pytest.raises(EdgeNameExistsException):
        g.add_edge("B", "A", name="e1")

    # Adding parallel edge with different name works
    e2 = g.add_edge("B", "A", name="e2")
    assert e2.name == "e2"


def test_show_edge_and_edge_not_found():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="e1")

    edge = g.show_edge("A", "B")
    assert edge.name == "e1"

    # Raises VertexNotExistsException if vertex missing
    with pytest.raises(VertexNotExistsException):
        g.show_edge("A", "C")

    # Raises EdgeNotFoundException if edge not exists
    with pytest.raises(EdgeNotFoundException):
        g.show_edge("A", "A")


def test_delete_edge_and_edge_not_found():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="e1")

    edge = g.delete_edge("A", "B")
    assert edge.name == "e1"
    assert g.num_edges() == 0

    # Deleting non-existent edge raises EdgeNotFoundException
    with pytest.raises(EdgeNotFoundException):
        g.delete_edge("A", "B")


def test_vertex_neighbors_and_degree():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_vertex("C")

    g.add_edge("A", "B", name="e1")
    g.add_edge("A", "C", name="e2")

    neighbors = g.vertex_neighbors("A")
    neighbors_values = {v.value for v in neighbors}
    assert neighbors_values == {"B", "C"}

    degree = g.vertex_degree("A")
    assert degree == 2

    # vertex_neighbors for non-existent vertex raises
    with pytest.raises(VertexNotExistsException):
        g.vertex_neighbors("D")

    with pytest.raises(VertexNotExistsException):
        g.vertex_degree("D")


def test_is_vertices_adjacents_true_and_false():
    g = SimpleGraph()
    g.add_vertex("X")
    g.add_vertex("Y")
    g.add_vertex("Z")

    g.add_edge("X", "Y", name="edge1")

    assert g.is_vertices_adjacents("X", "Y") is True
    assert g.is_vertices_adjacents("Y", "X") is True
    assert g.is_vertices_adjacents("X", "Z") is False


def test_has_cycle_detection():
    g = SimpleGraph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)
    g.add_edge("A", "B", name="e1")
    g.add_edge("B", "C", name="e2")
    assert g.has_cycle() is False

    # Add cycle
    g.add_edge("C", "A", name="e3")
    assert g.has_cycle() is True


def test_has_loop_detection_in_graph():
    g = SimpleGraph()
    g.add_vertex("A")
    # Adding edge loop in SimpleGraph raises LoopDetectedException
    with pytest.raises(LoopDetectedException):
        g.add_edge("A", "A", name="loop")

    # But if forcibly add loop edge through Graph (bypassing SimpleGraph)
    g2 = Graph()
    g2.vertices = g.vertices
    g2.edges = []
    loop_edge = g2.add_edge("A", "A", name="loop_edge")
    assert g2.has_loop() is True
    assert g.has_loop() is False  # Because SimpleGraph does not allow loops


def test_check_regular_graph_true_and_false():
    g = SimpleGraph()
    for v in ["A", "B", "C", "D"]:
        g.add_vertex(v)

    # Regular graph: square
    g.add_edge("A", "B", name="e1")
    g.add_edge("B", "C", name="e2")
    g.add_edge("C", "D", name="e3")
    g.add_edge("D", "A", name="e4")

    assert g.check_regular_graph() is True

    # Add an edge to break regularity
    g.add_edge("A", "C", name="e5")
    assert g.check_regular_graph() is False


def test_incidence_list_values_counts():
    g = SimpleGraph()
    for v in ["A", "B"]:
        g.add_vertex(v)

    g.add_edge("A", "B", name="e1")
    inc_list = g.incidence_list()

    # Since two vertices and one edge, total 2 entries with count 1, 0 no others
    counts = [ib.count for ib in inc_list]
    assert counts.count(1) == 2
    assert counts.count(0) == 0  # No loops or disconnected incidence


def test_adjacency_matrix_structure_and_contents():
    g = SimpleGraph()
    for v in ["A", "B", "C"]:
        g.add_vertex(v)

    g.add_edge("A", "B", name="e1")
    g.add_edge("B", "C", name="e2")
    matrix = g.adjacency_matrix()

    # Matrix keys == vertex values
    assert set(matrix.keys()) == {"A", "B", "C"}
    for row in matrix.values():
        assert set(row.keys()) == {"A", "B", "C"}

    # Check adjacency counts
    assert matrix["A"]["B"] == 1
    assert matrix["B"]["A"] == 1
    assert matrix["B"]["C"] == 1
    assert matrix["C"]["B"] == 1
    assert matrix["A"]["C"] == 0


def test_str_representation_of_graph_contains_edges():
    g = SimpleGraph()
    g.add_vertex("A")
    g.add_vertex("B")
    g.add_edge("A", "B", name="e1")
    string_repr = str(g)
    # The string must contain edge name and vertex values
    assert "e1" in string_repr
    assert "A" in string_repr
    assert "B" in string_repr
    

def test_num_vertex_and_num_edges_reporting():
    g = SimpleGraph()
    g.add_vertex("X")
    g.add_vertex("Y")
    g.add_vertex("Z")
    assert g.num_vertex() == 3
    assert g.num_edges() == 0
    g.add_edge("X", "Y", name="e1")
    assert g.num_edges() == 1


def test_list_graph_vertices_and_edges():
    g = SimpleGraph()
    for v in ["A", "B"]:
        g.add_vertex(v)
    g.add_edge("A", "B", name="edgeAB")

    vertices_list = g.list_graph_vertices()
    assert set(vertices_list) == {"A", "B"}

    edges_list = g.list_graph_edges()
    assert edges_list == ["edgeAB"]
