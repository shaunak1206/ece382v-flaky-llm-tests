import pytest

from darkchess.src.board import Board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.gamengine import GameEngine
from darkchess.src.king import King
from darkchess.src.knight import Knight
from darkchess.src.queen import Queen
from darkchess.src.rook import Rook
from darkchess.src.utils import clean_board


def empty_board():
    return Board(starting_board=clean_board())


def test_knight_name_value_and_symbol():
    white = Knight((3, 3), "white")
    black = Knight((3, 3), "black")
    assert white.get_name() == "knight"
    assert white.get_value() == 3
    assert white.get_symbol() == "wk "
    assert black.get_symbol() == "bk "
    assert white.symbol == "wk "


def test_knight_rejects_unknown_team():
    with pytest.raises(ValueError):
        Knight((0, 0), "red")


def test_knight_centre_has_eight_moves_on_empty_board():
    board = empty_board()
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight
    moves = knight.get_moves([board])
    expected_targets = {(1, 2), (1, 4), (2, 1), (2, 5),
                        (4, 1), (4, 5), (5, 2), (5, 4)}
    assert len(moves) == 8
    assert {m[1] for m in moves} == expected_targets
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_in_corner_has_two_moves():
    board = empty_board()
    knight = Knight((0, 0), "white")
    board.board[0, 0] = knight
    moves = knight.get_moves([board])
    assert set(moves) == {((0, 0), (2, 1), "base"), ((0, 0), (1, 2), "base")}


def test_black_knight_captures_enemy_and_is_blocked_by_own_team():
    board = empty_board()
    knight = Knight((3, 3), "black")
    board.board[3, 3] = knight
    board.board[1, 2] = Knight((1, 2), "white")
    board.board[1, 4] = Knight((1, 4), "black")
    moves = knight.get_moves([board])
    targets = [m[1] for m in moves]
    assert (1, 2) in targets
    assert (1, 4) not in targets
    assert len(moves) == 7


def test_default_board_layout():
    board = Board()
    assert board.turn == "white"
    assert board.board.shape == (8, 8)
    occupied = [cell for cell in board.board.flat if cell.team is not None]
    assert len(occupied) == 32
    assert isinstance(board.board[0, 1], Knight)
    assert board.board[0, 1].team == "white"
    assert isinstance(board.board[7, 6], Knight)
    assert board.board[7, 6].team == "black"
    assert board.board[0, 4].get_name() == "king"
    assert board.board[3, 3].team is None


def test_base_move_relocates_piece_and_passes_turn():
    board = Board()
    board.update_board(((0, 1), (2, 0), "base"), "white")
    knight = board.board[2, 0]
    assert isinstance(knight, Knight)
    assert knight.team == "white"
    assert knight.get_position() == (2, 0)
    assert knight.moved is True
    assert board.board[0, 1].team is None
    assert board.turn == "black"
    # a None keyword is treated as a base move, and the turn flips back
    board.update_board(((7, 1), (5, 0), None), "black")
    assert board.board[5, 0].team == "black"
    assert board.turn == "white"


def test_enpasse_removes_piece_beside_destination():
    board = empty_board()
    knight = Knight((4, 3), "white")
    board.board[4, 3] = knight
    board.board[4, 2] = Knight((4, 2), "black")
    board.update_board(((4, 3), (5, 2), "enpasse"), "white")
    assert board.board[5, 2] is knight
    assert knight.get_position() == (5, 2)
    assert board.board[4, 2].team is None
    assert board.board[4, 3].team is None


def test_castling_kingside_moves_rook_too():
    board = empty_board()
    king = King((0, 4), "white")
    rook = Rook((0, 7), "white")
    board.board[0, 4] = king
    board.board[0, 7] = rook
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert board.board[0, 6] is king
    assert king.get_position() == (0, 6)
    assert king.moved is True
    assert board.board[0, 5] is rook
    assert rook.get_position() == (0, 5)
    assert rook.moved is True
    assert board.board[0, 4].team is None
    assert board.board[0, 7].team is None


def test_evolve_promotes_to_queen():
    board = empty_board()
    board.board[6, 0] = Knight((6, 0), "white")
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    promoted = board.board[7, 0]
    assert isinstance(promoted, Queen)
    assert promoted.team == "white"
    assert board.board[6, 0].team is None


def test_update_board_rejects_unknown_turn():
    board = empty_board()
    board.board[0, 0] = Knight((0, 0), "white")
    board.turn = "purple"
    with pytest.raises(ValueError):
        board.update_board(((0, 0), (1, 2), "base"), "white")


def test_copy_is_independent_and_keeps_coordinates():
    board = Board()
    board.turn = "black"
    clone = board.copy()
    assert clone.turn == "black"
    assert clone.equal(board)
    assert clone.board[2, 3].get_position() == (2, 3)
    clone.board[0, 1] = Dummy((0, 1), None)
    assert board.board[0, 1].team == "white"
    assert clone.equal(board) is False


def test_view_board_prints_text_representation(capsys):
    board = Board()
    board.view_board()
    out = capsys.readouterr().out
    assert out == board.__str__() + "\n"
    assert out.splitlines()[0].strip().startswith("A")


def test_clean_board_is_fresh_grid_of_dummies():
    first = clean_board()
    second = clean_board()
    assert first.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(first[i, j], Dummy)
            assert first[i, j].team is None
            assert first[i, j].get_position() == (i, j)
    first[0, 0] = Knight((0, 0), "white")
    assert isinstance(second[0, 0], Dummy)


def test_engine_defaults_and_turn_order():
    board = Board()
    engine = GameEngine(board)
    assert engine.move_counter == 0
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.white_check is False
    assert engine.black_check is False
    assert len(engine.game_states) == 1
    assert engine.game_states[0] is not board
    assert engine.game_states[0].equal(board)
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")
    assert GameEngine(Board(), "computer").get_turn() == ("white", "computer")


def test_get_moves_returns_piece_moves():
    engine = GameEngine(Board())
    assert set(engine.get_moves((0, 1))) == {
        ((0, 1), (2, 0), "base"),
        ((0, 1), (2, 2), "base"),
    }
    assert engine.get_moves((3, 3)) == []


def test_num_alph_converts_to_board_names():
    result = GameEngine.num_alph({
        1: ((0, 1), (2, 0), "base"),
        2: ((7, 7), (5, 6), "base"),
    })
    assert result == {1: ["B1", "A3"], 2: ["H8", "G6"]}


def test_is_check_flags_white_king_attacked_by_black_knight():
    board = empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[2, 3] = Knight((2, 3), "black")
    engine = GameEngine(board)
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_is_check_flags_black_king_attacked_by_white_knight():
    board = empty_board()
    board.board[7, 4] = King((7, 4), "black")
    board.board[5, 3] = Knight((5, 3), "white")
    engine = GameEngine(board)
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_all_moves_includes_knight_moves_for_white():
    engine = GameEngine(Board())
    moves = engine.all_moves()
    assert ((0, 1), (2, 0), "base") in moves
    assert ((0, 1), (2, 2), "base") in moves
    assert all(engine.Board.board[m[0]].team == "white" for m in moves)


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    engine.Board.update_board(((0, 1), (2, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(Board())
    assert engine.Board.turn == "white"


def test_computer_choose_move_applies_move():
    engine = GameEngine(Board(), "computer", "computer")
    engine.choose_move(((0, 1), (2, 0), "base"))
    assert engine.move_counter == 1
    assert isinstance(engine.Board.board[2, 0], Knight)
    assert engine.Board.board[0, 1].team is None
    assert len(engine.game_states) == 2
    assert engine.get_turn()[0] == "black"


def test_stalemate_no_move_exits_when_team_has_no_moves():
    engine = GameEngine(Board(clean_board()))
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_stalemate_3states_exits_on_repetition():
    engine = GameEngine(Board())
    engine.game_states = [engine.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")
