import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.queen import Queen
from darkchess.src.king import King


def build(pieces):
    """Return a numpy board from clean_board() with the given pieces placed."""
    arr = clean_board()
    for piece in pieces:
        arr[piece.get_position()] = piece
    return arr


def states_for(arr):
    """Minimal game_states list: knight/engine read game_states[-1].board."""
    return [Board(arr)]


# ---------------------------------------------------------------- utils

def test_clean_board_is_8x8_of_dummies_without_team():
    arr = clean_board()
    assert arr.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            cell = arr[i, j]
            assert isinstance(cell, Dummy)
            assert cell.team is None


# ---------------------------------------------------------------- Board

def test_default_board_layout_and_turn():
    board = Board()
    assert board.turn == "white"
    assert isinstance(board.board[1, 0], Pawn)
    assert board.board[1, 0].team == "white"
    assert isinstance(board.board[0, 4], King)
    assert board.board[0, 4].team == "white"
    assert isinstance(board.board[7, 3], Queen)
    assert board.board[7, 3].team == "black"
    assert isinstance(board.board[3, 3], Dummy)


def test_str_has_header_footer_and_eight_rows():
    lines = str(Board()).splitlines()
    assert len(lines) == 10
    assert lines[0].startswith("  A")
    assert lines[-1].startswith("  A")
    assert lines[1].startswith("8 ")
    assert lines[8].startswith("1 ")


def test_update_board_base_move_moves_piece_and_switches_turn():
    board = Board()
    board.update_board(((1, 4), (3, 4), "base"), "white")
    moved = board.board[3, 4]
    assert isinstance(moved, Pawn)
    assert moved.get_position() == (3, 4)
    assert moved.moved is True
    assert isinstance(board.board[1, 4], Dummy)
    assert board.turn == "black"


def test_update_board_invalid_turn_raises():
    board = Board()
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((1, 0), (2, 0), "base"), "white")


def test_update_board_enpassant_removes_captured_pawn():
    white_pawn = Pawn((4, 4), "white")
    black_pawn = Pawn((4, 3), "black")
    board = Board(build([white_pawn, black_pawn]))
    board.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(board.board[5, 3], Pawn)
    assert board.board[5, 3].get_position() == (5, 3)
    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 3], Dummy)


def test_update_board_castling_kingside_moves_rook():
    king = King((0, 4), "white")
    rook = Rook((0, 7), "white")
    board = Board(build([king, rook]))
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Rook)
    assert board.board[0, 5].get_position() == (0, 5)
    assert board.board[0, 5].moved is True
    assert isinstance(board.board[0, 7], Dummy)
    assert isinstance(board.board[0, 4], Dummy)


def test_update_board_castling_queenside_moves_rook():
    king = King((0, 4), "white")
    rook = Rook((0, 0), "white")
    board = Board(build([king, rook]))
    board.update_board(((0, 4), (0, 2), "castling"), "white")
    assert isinstance(board.board[0, 2], King)
    assert isinstance(board.board[0, 3], Rook)
    assert isinstance(board.board[0, 0], Dummy)


def test_update_board_evolve_promotes_pawn_to_queen():
    pawn = Pawn((6, 0), "white")
    board = Board(build([pawn]))
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    promoted = board.board[7, 0]
    assert isinstance(promoted, Queen)
    assert promoted.team == "white"
    assert isinstance(board.board[6, 0], Dummy)


def test_copy_is_independent_and_equal_detects_differences():
    original = Board()
    dup = original.copy()
    assert dup is not original
    assert dup.equal(original)
    assert original.equal(dup)
    assert dup.turn == original.turn
    assert dup.board[1, 0] is not original.board[1, 0]

    dup.board[1, 0] = Dummy((1, 0), None)
    assert not dup.equal(original)
    assert not original.equal(dup)
    assert isinstance(original.board[1, 0], Pawn)


# ---------------------------------------------------------------- GameEngine

def test_engine_defaults():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.white_check is False
    assert engine.black_check is False


def test_get_turn_alternates_between_players():
    engine = GameEngine(Board(), player1="human", player2="computer")
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")


def test_num_alph_converts_tuples_to_chess_notation():
    result = GameEngine.num_alph({1: ((0, 1), (2, 0), "base")})
    assert result == {1: ["B1", "A3"]}


def test_get_moves_for_knight_at_start_excludes_own_pieces():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    ends = {m[1] for m in moves}
    assert ends == {(2, 0), (2, 2)}
    assert all(m[0] == (0, 1) and m[2] == "base" for m in moves)


def test_is_check_detects_king_attacked_by_knight():
    king = King((0, 4), "white")
    black_knight = Knight((2, 5), "black")  # attacks (0, 4)
    engine = GameEngine(Board(build([king, black_knight])))
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_stalemate_3states_exits_when_history_repeats():
    engine = GameEngine(Board(clean_board()))
    engine.game_states = [engine.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


def test_stalemate_no_move_exits_when_no_moves_and_not_in_check():
    engine = GameEngine(Board(clean_board()))
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_choose_move_computer_plays_move_and_advances_counter():
    engine = GameEngine(Board(), player1="computer", player2="computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert isinstance(engine.Board.board[3, 4], Pawn)
    assert engine.get_turn() == ("black", "computer")


# ---------------------------------------------------------------- Knight

def test_knight_name_value_and_symbols():
    white = Knight((0, 1), "white")
    black = Knight((7, 1), "black")
    assert white.get_name() == "knight"
    assert white.get_value() == 3
    assert white.get_symbol() == "wk "
    assert black.get_symbol() == "bk "


def test_knight_invalid_team_raises():
    knight = Knight((0, 0), "green")
    with pytest.raises(ValueError):
        knight.get_symbol()
    with pytest.raises(ValueError):
        knight.get_moves(states_for(clean_board()))


def test_knight_in_center_has_eight_moves_on_empty_board():
    knight = Knight((3, 3), "white")
    arr = build([knight])
    moves = knight.get_moves(states_for(arr))
    ends = {m[1] for m in moves}
    assert len(moves) == 8
    assert ends == {(1, 2), (1, 4), (2, 1), (2, 5),
                    (4, 1), (4, 5), (5, 2), (5, 4)}


def test_knight_in_corner_blocked_by_own_piece():
    knight = Knight((0, 0), "white")
    blocker = Knight((1, 2), "white")
    arr = build([knight, blocker])
    moves = knight.get_moves(states_for(arr))
    assert {m[1] for m in moves} == {(2, 1)}


def test_knight_can_capture_enemy_piece():
    knight = Knight((0, 0), "white")
    enemy = Knight((2, 1), "black")
    arr = build([knight, enemy])
    moves = knight.get_moves(states_for(arr))
    assert {m[1] for m in moves} == {(2, 1), (1, 2)}


def test_knight_copy_preserves_state():
    knight = Knight((2, 2), "black")
    knight.moved = True
    dup = knight.copy()
    assert isinstance(dup, Knight)
    assert dup is not knight
    assert dup.get_position() == (2, 2)
    assert dup.team == "black"
    assert dup.moved is True
