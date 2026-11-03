import pytest

from darkchess.src.board import Board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.gamengine import GameEngine
from darkchess.src.king import King
from darkchess.src.knight import Knight
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen
from darkchess.src.rook import Rook
from darkchess.src.utils import clean_board


def empty_board():
    """Board with no pieces on it."""
    return Board(starting_board=clean_board())


def put(board, piece):
    """Place a piece on the board at its own position."""
    board.board[piece.pos] = piece
    return board


# ---------------------------------------------------------------- utils.clean_board


def test_clean_board_is_8x8_array_of_empty_dummies():
    grid = clean_board()
    assert grid.shape == (8, 8)
    assert grid.dtype == object
    for i in range(8):
        for j in range(8):
            tile = grid[i, j]
            assert isinstance(tile, Dummy)
            assert tile.team is None
            assert tile.get_position() == (i, j)
            assert tile.symbol == "   "


def test_clean_board_returns_independent_arrays():
    first = clean_board()
    second = clean_board()
    assert first is not second
    first[0, 0] = Knight((0, 0), "white")
    assert isinstance(second[0, 0], Dummy)
    assert second[0, 0].team is None


# ---------------------------------------------------------------- Knight


def test_knight_name_value_and_symbols():
    white = Knight((0, 1), "white")
    black = Knight((7, 1), "black")
    assert white.get_name() == "knight"
    assert white.get_value() == 3
    assert white.get_symbol() == "wk "
    assert black.get_symbol() == "bk "


def test_knight_rejects_unknown_team():
    with pytest.raises(ValueError):
        Knight((0, 0), "green")


def test_knight_in_centre_has_eight_moves_on_empty_board():
    board = empty_board()
    knight = Knight((3, 3), "white")
    put(board, knight)
    moves = knight.get_moves([board])
    ends = {end for _, end, _ in moves}
    assert len(moves) == 8
    assert ends == {(1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)}
    assert all(start == (3, 3) and kind == "base" for start, _, kind in moves)


def test_black_knight_in_corner_has_two_moves():
    board = empty_board()
    knight = Knight((7, 7), "black")
    put(board, knight)
    ends = {end for _, end, _ in knight.get_moves([board])}
    assert ends == {(5, 6), (6, 5)}


def test_knight_skips_own_piece_and_captures_enemy():
    board = empty_board()
    knight = Knight((3, 3), "white")
    put(board, knight)
    put(board, Knight((1, 2), "white"))
    put(board, Knight((1, 4), "black"))
    moves = knight.get_moves([board])
    ends = {end for _, end, _ in moves}
    assert len(moves) == 7
    assert (1, 2) not in ends
    assert (1, 4) in ends


def test_knight_reads_blockers_from_latest_state():
    old = empty_board()
    blocked = empty_board()
    put(blocked, Knight((1, 2), "white"))
    knight = Knight((3, 3), "white")
    assert len(knight.get_moves([old])) == 8
    moves = knight.get_moves([old, blocked])
    ends = {end for _, end, _ in moves}
    assert len(moves) == 7
    assert (1, 2) not in ends


# ---------------------------------------------------------------- Board


def test_default_board_layout():
    board = Board()
    grid = board.board
    assert board.turn == "white"
    assert isinstance(grid[0, 0], Rook) and grid[0, 0].team == "white"
    assert isinstance(grid[0, 1], Knight) and grid[0, 1].team == "white"
    assert isinstance(grid[0, 3], Queen) and grid[0, 3].team == "white"
    assert isinstance(grid[0, 4], King) and grid[0, 4].team == "white"
    assert all(isinstance(grid[1, j], Pawn) and grid[1, j].team == "white" for j in range(8))
    assert all(isinstance(grid[6, j], Pawn) and grid[6, j].team == "black" for j in range(8))
    assert isinstance(grid[7, 4], King) and grid[7, 4].team == "black"
    assert isinstance(grid[7, 3], Queen) and grid[7, 3].team == "black"
    assert isinstance(grid[2, 0], Dummy) and grid[2, 0].team is None
    for i in range(8):
        for j in range(8):
            assert grid[i, j].get_position() == (i, j)


def test_board_text_has_coordinates_and_eight_ranks():
    text = str(Board())
    lines = text.splitlines()
    assert len(lines) == 10
    assert lines[0].split() == list("ABCDEFGH")
    assert lines[-1].split() == list("ABCDEFGH")
    for row, rank in enumerate(range(8, 0, -1), start=1):
        assert lines[row].startswith(str(rank) + " ")
        assert lines[row].endswith(" " + str(rank))
    assert "wk " in text


def test_equal_detects_a_changed_layout():
    first = Board()
    second = Board()
    assert first.equal(second)
    second.update_board(((1, 4), (3, 4), "base"), "white")
    assert not first.equal(second)


def test_base_move_relocates_piece_and_flips_turn():
    board = empty_board()
    put(board, Knight((0, 1), "white"))
    board.update_board(((0, 1), (2, 0), "base"), "white")
    knight = board.board[2, 0]
    assert isinstance(knight, Knight)
    assert knight.get_position() == (2, 0)
    assert knight.moved is True
    assert isinstance(board.board[0, 1], Dummy)
    assert board.board[0, 1].get_position() == (0, 1)
    assert board.turn == "black"
    # a None keyword word is treated as a plain move
    board.update_board(((2, 0), (0, 1), None), "white")
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[2, 0], Dummy)
    assert board.turn == "white"


def test_enpassant_removes_captured_pawn():
    board = empty_board()
    put(board, Pawn((4, 3), "white"))
    put(board, Pawn((4, 4), "black"))
    board.update_board(((4, 3), (5, 4), "enpasse"), "white")
    assert isinstance(board.board[5, 4], Pawn)
    assert board.board[5, 4].team == "white"
    assert board.board[5, 4].get_position() == (5, 4)
    assert board.board[4, 3].team is None
    assert board.board[4, 4].team is None


def test_castling_moves_rook_on_both_wings():
    kingside = empty_board()
    put(kingside, King((0, 4), "white"))
    put(kingside, Rook((0, 7), "white"))
    kingside.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(kingside.board[0, 6], King)
    assert kingside.board[0, 6].get_position() == (0, 6)
    assert isinstance(kingside.board[0, 5], Rook)
    assert kingside.board[0, 5].get_position() == (0, 5)
    assert kingside.board[0, 5].moved is True
    assert kingside.board[0, 4].team is None
    assert kingside.board[0, 7].team is None

    queenside = empty_board()
    put(queenside, King((0, 4), "white"))
    put(queenside, Rook((0, 0), "white"))
    queenside.update_board(((0, 4), (0, 2), "castling"), "white")
    assert isinstance(queenside.board[0, 2], King)
    assert queenside.board[0, 2].get_position() == (0, 2)
    assert isinstance(queenside.board[0, 3], Rook)
    assert queenside.board[0, 3].get_position() == (0, 3)
    assert queenside.board[0, 3].moved is True
    assert queenside.board[0, 0].team is None


def test_evolve_turns_pawn_into_queen():
    board = empty_board()
    put(board, Pawn((6, 0), "white"))
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    promoted = board.board[7, 0]
    assert isinstance(promoted, Queen)
    assert promoted.team == "white"
    assert promoted.get_position() == (7, 0)
    assert board.board[6, 0].team is None


def test_update_board_rejects_unknown_turn():
    board = empty_board()
    put(board, Knight((0, 1), "white"))
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((0, 1), (2, 0), "base"), "white")


def test_copy_is_independent_of_original():
    original = Board()
    clone = original.copy()
    assert clone.equal(original)
    assert clone.turn == original.turn
    assert clone.board[1, 0] is not original.board[1, 0]
    assert clone.board[1, 0].get_position() == (1, 0)
    clone.update_board(((1, 0), (2, 0), "base"), "white")
    assert isinstance(original.board[1, 0], Pawn)
    assert isinstance(original.board[2, 0], Dummy)
    assert original.turn == "white"
    assert clone.turn == "black"


# ---------------------------------------------------------------- GameEngine


def test_engine_defaults_and_turn_order():
    board = Board()
    engine = GameEngine(board)
    assert engine.move_counter == 0
    assert engine.player1 == "human" and engine.player2 == "human"
    assert engine.white_check is False and engine.black_check is False
    assert len(engine.game_states) == 1
    assert engine.game_states[0] is not board
    assert engine.game_states[0].equal(board)
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")
    assert GameEngine(Board(), player1="computer").get_turn() == ("white", "computer")


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    before = engine.Board.copy()
    engine.Board.update_board(((1, 0), (2, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(before)
    assert engine.Board.turn == "white"
    assert isinstance(engine.Board.board[1, 0], Pawn)
    assert engine.Board.board[1, 0].team == "white"


def test_is_check_flags_both_kings_attacked_by_knights():
    board = empty_board()
    put(board, King((0, 4), "white"))
    put(board, Knight((2, 3), "black"))  # attacks (0, 4)
    put(board, King((7, 4), "black"))
    put(board, Knight((5, 3), "white"))  # attacks (7, 4)
    engine = GameEngine(board)
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is True


def test_side_with_no_moves_and_not_in_check_ends_game():
    board = empty_board()
    put(board, Knight((3, 3), "black"))
    engine = GameEngine(board)
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")
    # the starting position still has white moves, so the game goes on
    assert GameEngine(Board()).is_stalemate_no_move("white") is None


def test_repeated_states_end_game_only_when_full_window_matches():
    engine = GameEngine(Board())
    engine.game_states = [engine.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")

    engine.game_states = [engine.Board.copy() for _ in range(8)]
    assert engine.is_stalemate_3states("white") is None

    engine.game_states = [Board(starting_board=clean_board())] + [
        engine.Board.copy() for _ in range(8)
    ]
    assert engine.is_stalemate_3states("white") is None


def test_choose_move_computer_applies_given_move():
    engine = GameEngine(Board(), player1="computer")
    engine.choose_move(((0, 1), (2, 0), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert isinstance(engine.Board.board[2, 0], Knight)
    assert engine.get_turn() == ("black", "human")


def test_human_mode_retries_until_a_valid_piece_and_move_are_chosen(monkeypatch):
    # "C3" is an empty tile and "B8" is a black piece, both rejected; then "B1" and move 1
    answers = iter(["C3", "B8", "B1", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    engine = GameEngine(Board())
    engine.choose_move()
    assert engine.move_counter == 1
    assert isinstance(engine.Board.board[2, 0], Knight)
    assert engine.Board.board[2, 0].team == "white"
    assert isinstance(engine.Board.board[0, 1], Dummy)
