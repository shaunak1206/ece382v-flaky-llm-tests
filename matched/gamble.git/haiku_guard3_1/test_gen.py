"""
Unit tests for gamble.models.cards (Card and Deck).
"""
import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck


def _names(cards):
    return [str(card) for card in cards]


# ---------------------------------------------------------------------------
# Card tests
# ---------------------------------------------------------------------------


def test_default_card_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_get_parses_text_case_insensitively():
    upper = Card.get("TH")
    lower = Card.get("th")
    assert upper.value is Card.Values.TEN
    assert upper.suit is Card.Suits.HEARTS
    assert lower == upper


def test_get_rejects_wrong_length():
    with pytest.raises(InvalidCard):
        Card.get("AS2")
    with pytest.raises(InvalidCard):
        Card.get("A")
    with pytest.raises(InvalidCard):
        Card.get("")


def test_get_rejects_invalid_value_or_suit():
    with pytest.raises(InvalidCard):
        Card.get("1S")  # invalid value character
    with pytest.raises(InvalidCard):
        Card.get("AX")  # invalid suit character


def test_str_and_repr():
    card = Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    assert str(card) == "A\u2660"
    assert repr(card) == "<Card:A\u2660>"


def test_full_name():
    assert Card.get("QD").full_name == "queen of diamonds"
    assert Card.get("2C").full_name == "two of clubs"


def test_color_and_red_black_properties():
    hearts = Card.get("KH")
    diamonds = Card.get("2D")
    spades = Card.get("AS")
    clubs = Card.get("9C")

    assert hearts.color == Card.RED
    assert diamonds.color == Card.RED
    assert hearts.is_red is True
    assert hearts.is_black is False

    assert spades.color == Card.BLACK
    assert clubs.color == Card.BLACK
    assert clubs.is_black is True
    assert clubs.is_red is False


def test_ordering_by_value_including_le_and_ge():
    ace = Card.get("AS")
    two = Card.get("2H")
    king = Card.get("KD")
    queen = Card.get("QC")
    ace_other_suit = Card.get("AH")

    assert ace < two
    assert king > queen
    assert queen < king
    assert ace <= ace_other_suit
    assert ace >= ace_other_suit
    assert king >= queen
    assert not (king <= queen)


def test_equality_requires_same_value_and_suit():
    assert Card.get("5H") == Card.get("5H")
    assert Card.get("5H") != Card.get("5S")
    assert Card.get("5H") != Card.get("6H")
    assert (Card.get("5H") == "5H") is False


def test_unicode_skips_knight_for_queen_and_king():
    base = 127136  # spades base code point
    # Ace of spades: base + 1
    assert Card.get("AS").unicode == chr(base + 1)
    # Ten of spades: no skip
    assert Card.get("TS").unicode == chr(base + 10)
    # Queen and king: skip the knight, so one extra offset
    assert Card.get("QS").unicode == chr(base + 12 + 1)
    assert Card.get("KS").unicode == chr(base + 13 + 1)


def test_suits_all_sorted_and_dict_keys():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    suit_dict = Card.Suits.dict()
    assert set(suit_dict.keys()) == {"S", "C", "D", "H"}
    assert suit_dict["H"] is Card.Suits.HEARTS


def test_values_all_sorted_and_dict_keys():
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0] is Card.Values.ACE
    assert values[-1] is Card.Values.KING
    value_dict = Card.Values.dict()
    assert len(value_dict) == 13
    assert value_dict["T"] is Card.Values.TEN


# ---------------------------------------------------------------------------
# Deck tests
# ---------------------------------------------------------------------------


def test_default_deck_has_52_unique_cards_and_repr():
    random.seed(1001)
    deck = Deck()
    assert deck.cards_left == 52
    assert len(set(_names(deck.cards))) == 52
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_unshuffled_deck_top_and_bottom():
    random.seed(1002)
    deck = Deck(shuffle=False)
    # Built as suits in order, values ace..king, then reversed.
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_draw_single_card_updates_state():
    random.seed(1003)
    deck = Deck(shuffle=False)
    top = deck.top
    drawn = deck.draw()
    assert drawn == top
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_cards_returns_list():
    random.seed(1004)
    deck = Deck(shuffle=False)
    expected = _names(reversed(deck.cards[-3:]))
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert len(drawn) == 3
    assert _names(drawn) == expected
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_hand_default_size_is_five():
    random.seed(1005)
    deck = Deck()
    hand = deck.draw_hand()
    assert len(hand) == 5
    assert deck.cards_left == 47
    assert deck.draws == 5


def test_draw_hand_size_one_still_returns_hand():
    random.seed(1006)
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=1)
    assert len(hand) == 1
    assert deck.cards_left == 51


def test_draw_from_empty_deck_raises():
    random.seed(1007)
    deck = Deck(shuffle=False)
    deck.draw(times=52)
    assert deck.cards_left == 0
    with pytest.raises(IndexError):
        deck.draw()


def test_shuffle_same_seed_gives_same_order():
    random.seed(2024)
    first = Deck()
    random.seed(2024)
    second = Deck()
    assert _names(first.cards) == _names(second.cards)


def test_shuffle_preserves_cards_and_counts_shuffles():
    random.seed(3003)
    deck = Deck(shuffle=False)
    before = sorted(_names(deck.cards))
    assert deck.shuffles == 0

    deck.shuffle(times=3)

    assert deck.shuffles == 3
    assert sorted(_names(deck.cards)) == before
    assert deck.cards_left == 52


def test_shuffle_calls_random_shuffle_on_deck_cards(monkeypatch):
    random.seed(4004)
    calls = []

    def fake_shuffle(seq):
        calls.append(seq)

    monkeypatch.setattr(random, "shuffle", fake_shuffle)

    deck = Deck(shuffle=False)
    deck.shuffle(times=2)

    assert len(calls) == 2
    assert all(seq is deck.cards for seq in calls)


def test_contains_card_and_non_card():
    random.seed(5005)
    deck = Deck(cards=[Card.get("AS"), Card.get("9H")], shuffle=False)
    assert Card.get("AS") in deck
    assert Card.get("9H") in deck
    assert Card.get("KH") not in deck
    assert "AS" not in deck


def test_euchre_deck_contains_only_euchre_values():
    random.seed(6006)
    deck = EuchreDeck()
    assert deck.cards_left == 24
    value_chars = {card.value.char for card in deck.cards}
    assert value_chars == {"9", "T", "J", "Q", "K", "A"}
    assert len(set(_names(deck.cards))) == 24
