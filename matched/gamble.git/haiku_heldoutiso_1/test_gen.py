import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


# ---------------------------------------------------------------------------
# Card: construction and text representation
# ---------------------------------------------------------------------------

def test_default_card_is_ace_of_spades_and_str_repr():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A\u2660"
    assert repr(card) == "<Card:A\u2660>"


@pytest.mark.parametrize(
    "text, value, suit",
    [
        ("AS", Card.Values.ACE, Card.Suits.SPADES),
        ("TH", Card.Values.TEN, Card.Suits.HEARTS),
        ("2D", Card.Values.TWO, Card.Suits.DIAMONDS),
        ("kc", Card.Values.KING, Card.Suits.CLUBS),
        ("ts", Card.Values.TEN, Card.Suits.SPADES),
    ],
)
def test_get_parses_text_case_insensitively(text, value, suit):
    card = Card.get(text)
    assert card.value is value
    assert card.suit is suit


@pytest.mark.parametrize("text", ["", "A", "AS2", "ASDF"])
def test_get_rejects_wrong_length(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_get_rejects_invalid_value():
    with pytest.raises(InvalidCard):
        Card.get("1S")


def test_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard):
        Card.get("AX")


# ---------------------------------------------------------------------------
# Card: equality and ordering
# ---------------------------------------------------------------------------

def test_equality_requires_same_value_and_suit():
    assert Card.get("KH") == Card.get("kh")
    assert Card.get("KH") != Card.get("KS")
    assert Card.get("KH") != Card.get("QH")


def test_equality_with_non_card_is_false():
    assert (Card() == "AS") is False
    assert (Card() == None) is False  # noqa: E711


def test_ordering_places_ace_lowest_and_king_highest():
    ace = Card.get("AS")
    two = Card.get("2S")
    king = Card.get("KS")
    assert ace < two
    assert two < king
    assert king > ace
    assert not (king < ace)


def test_le_and_ge_include_equal_cards():
    a = Card.get("7H")
    b = Card.get("7C")  # same value, different suit
    assert a <= b
    assert a >= b
    assert Card.get("9H") >= Card.get("7H")
    assert Card.get("7H") <= Card.get("9H")
    assert not (Card.get("9H") <= Card.get("7H"))


# ---------------------------------------------------------------------------
# Card: properties
# ---------------------------------------------------------------------------

def test_color_and_is_black_is_red():
    spades = Card.get("AS")
    hearts = Card.get("AH")
    assert spades.color == Card.BLACK
    assert spades.is_black is True
    assert spades.is_red is False
    assert hearts.color == Card.RED
    assert hearts.is_red is True
    assert hearts.is_black is False


def test_full_name():
    assert Card.get("QD").full_name == "queen of diamonds"


def test_unicode_glyphs_skip_knight_for_queen_and_king():
    assert Card.get("AS").unicode == "\U0001F0A1"
    assert Card.get("TS").unicode == "\U0001F0AA"
    assert Card.get("JS").unicode == "\U0001F0AB"
    # Queen and King skip the knight glyph (+1 offset)
    assert Card.get("QS").unicode == "\U0001F0AD"
    assert Card.get("KS").unicode == "\U0001F0AE"
    assert Card.get("QH").unicode == "\U0001F0BD"


# ---------------------------------------------------------------------------
# Card.Suits and Card.Values enums
# ---------------------------------------------------------------------------

def test_suits_all_sorted_by_value_and_dict_keys():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.value for s in suits] == sorted(s.value for s in suits)
    assert set(Card.Suits.dict().keys()) == {"S", "C", "D", "H"}


def test_values_all_sorted_ace_to_king_and_dict_keys():
    values = Card.Values.all()
    assert len(values) == 13
    assert values[0] is Card.Values.ACE
    assert values[-1] is Card.Values.KING
    assert [v.value for v in values] == list(range(1, 14))
    assert set(Card.Values.dict().keys()) == set("A23456789TJQK")


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    pairs = {(c.value.name, c.suit.name) for c in deck.cards}
    assert len(pairs) == 52


def test_unshuffled_deck_top_is_ace_of_spades_and_bottom_is_king_of_hearts():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")
    assert deck.shuffles == 0
    assert deck.draws == 0


def test_shuffle_on_construction_counts_one_shuffle():
    assert Deck().shuffles == 1
    assert Deck(shuffle=False).shuffles == 0


def test_shuffle_calls_random_shuffle_and_counts_times(monkeypatch):
    calls = []

    def fake_shuffle(seq):
        calls.append(len(seq))

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    deck = Deck(shuffle=False)
    deck.shuffle(times=3)
    assert calls == [52, 52, 52]
    assert deck.shuffles == 3


def test_draw_single_card_returns_card_and_updates_counters():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert isinstance(card, Card)
    assert card == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_cards_returns_list():
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert all(isinstance(c, Card) for c in cards)
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert hand.size == 5
    assert len(hand) == 5
    assert deck.cards_left == 47
    assert deck.draws == 5


def test_draw_hand_of_size_one_still_returns_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=1)
    assert isinstance(hand, Hand)
    assert len(hand) == 1
    assert deck.cards_left == 51


def test_contains_tracks_cards_in_deck():
    deck = Deck(shuffle=False)
    top = deck.top
    assert top in deck
    assert "AS" not in deck
    deck.draw()
    assert top not in deck


def test_draw_from_empty_deck_raises_index_error():
    deck = Deck(cards=[Card()], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_str_and_repr_show_cards_left():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw(times=2)
    assert str(deck) == "<Deck[50]>"


def test_euchre_deck_contains_only_nine_through_ace_of_each_suit():
    deck = EuchreDeck()
    assert isinstance(deck, Deck)
    assert deck.cards_left == 24
    names = {c.value.name for c in deck.cards}
    assert names == {"nine", "ten", "jack", "queen", "king", "ace"}
    assert all(c.value.name != "two" for c in deck.cards)
    suits = {c.suit.name for c in deck.cards}
    assert suits == {"spades", "clubs", "diamonds", "hearts"}
