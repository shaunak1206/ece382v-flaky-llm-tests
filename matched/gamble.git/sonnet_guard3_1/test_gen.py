import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck


def _card_strs(cards):
    return sorted(str(c) for c in cards)


def test_suits_all_sorted_by_value():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_values_all_has_thirteen_sorted():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0].char == "A"
    assert values[-1].char == "K"


def test_suits_and_values_dict_keys():
    assert set(Card.Suits.dict().keys()) == {"S", "C", "D", "H"}
    assert set(Card.Values.dict().keys()) == set("A23456789TJQK")
    assert Card.Suits.dict()["H"] is Card.Suits.HEARTS
    assert Card.Values.dict()["T"] is Card.Values.TEN


def test_card_defaults_to_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES
    assert card.full_name == "ace of spades"


def test_card_get_parses_case_insensitively():
    card = Card.get("kh")
    assert card.value == Card.Values.KING
    assert card.suit == Card.Suits.HEARTS
    assert card == Card.get("KH")


@pytest.mark.parametrize("text", ["", "A", "10H", "ASX"])
def test_card_get_wrong_length_raises(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_get_invalid_value_or_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("1S")
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_card_colors():
    black = Card.get("AS")
    red = Card.get("AH")
    assert black.is_black and not black.is_red
    assert red.is_red and not red.is_black
    assert black.color == Card.BLACK
    assert red.color == Card.RED
    assert Card.get("2C").is_black
    assert Card.get("2D").is_red


def test_card_str_and_repr():
    card = Card.get("TD")
    assert str(card) == "T\u2666"
    assert repr(card) == "<Card:T\u2666>"


def test_card_unicode_skips_knight():
    assert Card.get("AS").unicode == chr(127137)
    assert Card.get("JS").unicode == chr(127136 + 11)
    # queen and king skip the knight code point
    assert Card.get("QS").unicode == chr(127136 + 12 + 1)
    assert Card.get("KS").unicode == chr(127136 + 13 + 1)


def test_card_ordering_by_value_only():
    two = Card.get("2S")
    king = Card.get("KH")
    ace = Card.get("AS")
    assert two < king
    assert king > two
    assert ace < two  # ace is low
    assert two <= king
    assert king >= two


def test_card_same_value_different_suit():
    a = Card.get("5S")
    b = Card.get("5H")
    assert a != b
    assert not a < b
    assert not a > b
    # <= and >= also require equality, which fails across suits
    assert not a <= b
    assert not a >= b


def test_card_equality():
    assert Card.get("9C") == Card.get("9c")
    assert Card.get("9C") != Card.get("9D")
    assert Card.get("9C") != "9C"
    assert not (Card.get("9C") == 9)
    assert Card.get("9C") <= Card.get("9C")
    assert Card.get("9C") >= Card.get("9C")


def test_default_deck_unshuffled_order_and_contents():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")
    assert len(set(_card_strs(deck.cards))) == 52
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_shuffle_preserves_cards_and_counts():
    random.seed(1234)
    deck = Deck(shuffle=False)
    before = _card_strs(deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert deck.cards_left == 52
    assert _card_strs(deck.cards) == before


def test_deck_shuffles_on_construction_by_default():
    random.seed(99)
    deck = Deck()
    assert deck.shuffles == 1
    assert deck.cards_left == 52
    assert len(set(_card_strs(deck.cards))) == 52


def test_deck_draw_single_and_multiple():
    random.seed(5)
    deck = Deck(shuffle=False)
    first = deck.draw()
    assert isinstance(first, Card)
    assert first == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1

    several = deck.draw(times=3)
    assert isinstance(several, list)
    assert several == [Card.get("2S"), Card.get("3S"), Card.get("4S")]
    assert deck.cards_left == 48
    assert deck.draws == 4


def test_deck_draw_zero_returns_empty_list():
    deck = Deck(shuffle=False)
    assert deck.draw(times=0) == []
    assert deck.cards_left == 52
    assert deck.draws == 0


def test_deck_draw_from_empty_raises():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
    assert deck.cards_left == 0
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_contains():
    deck = Deck(shuffle=False)
    ace = Card.get("AS")
    assert ace in deck
    assert "AS" not in deck
    assert 5 not in deck
    drawn = deck.draw()
    assert drawn == ace
    assert ace not in deck


def test_deck_custom_cards_used_and_empty_list_falls_back():
    custom = [Card.get("2H"), Card.get("3H")]
    deck = Deck(cards=custom, shuffle=False)
    assert deck.cards_left == 2
    assert deck.top == Card.get("3H")
    assert deck.bottom == Card.get("2H")
    assert Deck(cards=[], shuffle=False).cards_left == 52


def test_deck_draw_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert len(hand) == 5
    assert deck.cards_left == 47
    assert deck.draws == 5
    single = deck.draw_hand(size=1)
    assert len(single) == 1
    assert deck.cards_left == 46


def test_euchre_deck_contents():
    random.seed(42)
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert {c.value.char for c in deck.cards} == set("9TJQKA")
    assert len(set(_card_strs(deck.cards))) == 24
    assert Card.get("2S") not in deck
    assert Card.get("9H") in deck
