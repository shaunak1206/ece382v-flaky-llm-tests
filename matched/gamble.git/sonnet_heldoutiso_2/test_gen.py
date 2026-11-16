import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_suits_all_sorted_and_complete():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_values_all_and_dict():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    mapping = Card.Values.dict()
    assert mapping["A"] is Card.Values.ACE
    assert mapping["T"] is Card.Values.TEN
    assert mapping["K"] is Card.Values.KING
    assert Card.Suits.dict()["H"] is Card.Suits.HEARTS


def test_card_default_constructor():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_card_get_valid_and_case_insensitive():
    card = Card.get("kh")
    assert card.value is Card.Values.KING
    assert card.suit is Card.Suits.HEARTS
    assert Card.get("TD") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)


@pytest.mark.parametrize("text", ["", "A", "10S", "ASD"])
def test_card_get_wrong_length_raises(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_get_invalid_value_raises():
    with pytest.raises(InvalidCard):
        Card.get("1S")


def test_card_get_invalid_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_card_color_properties():
    spade = Card.get("AS")
    club = Card.get("2C")
    diamond = Card.get("3D")
    heart = Card.get("4H")
    assert spade.is_black and club.is_black
    assert not spade.is_red and not club.is_red
    assert diamond.is_red and heart.is_red
    assert not diamond.is_black and not heart.is_black
    assert spade.color == Card.BLACK
    assert heart.color == Card.RED


def test_card_names_and_string_forms():
    card = Card.get("AS")
    assert card.full_name == "ace of spades"
    assert str(card) == "A\u2660"
    assert repr(card) == "<Card:A\u2660>"
    assert str(Card.get("TH")) == "T\u2665"


def test_card_unicode_skips_knight():
    ace = Card.get("AS")
    assert ace.unicode == chr(127136 + 1)
    jack = Card.get("JS")
    assert jack.unicode == chr(127136 + 11)
    queen = Card.get("QS")
    assert queen.unicode == chr(127136 + 13)
    king = Card.get("KS")
    assert king.unicode == chr(127136 + 14)


def test_card_comparisons_by_value_only():
    low = Card.get("2S")
    high = Card.get("KH")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    assert not high < low
    assert not low > high


def test_card_same_value_different_suit_not_equal():
    a = Card.get("5S")
    b = Card.get("5H")
    assert a != b
    assert not a < b and not a > b
    assert not a <= b and not a >= b
    c = Card.get("5S")
    assert a == c
    assert a <= c and a >= c


def test_card_equality_with_non_card():
    assert Card.get("AS") != "AS"
    assert not (Card.get("AS") == 1)


def test_deck_default_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    assert len({str(c) for c in deck.cards}) == 52
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_unshuffled_order_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_shuffle_counts_and_preserves_cards():
    deck = Deck()
    assert deck.shuffles == 1
    before = sorted(str(c) for c in deck.cards)
    random.seed(1234)
    deck.shuffle(times=3)
    assert deck.shuffles == 4
    assert deck.cards_left == 52
    assert sorted(str(c) for c in deck.cards) == before


def test_deck_draw_single_card():
    deck = Deck(shuffle=False)
    top = deck.top
    card = deck.draw()
    assert isinstance(card, Card)
    assert card == top
    assert deck.cards_left == 51
    assert deck.draws == 1
    assert card not in deck


def test_deck_draw_multiple_returns_list():
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert cards == [Card.get("AS"), Card.get("2S"), Card.get("3S")]
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_deck_draw_from_empty_raises():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_contains():
    deck = Deck(cards=[Card.get("AS"), Card.get("2H")], shuffle=False)
    assert Card.get("AS") in deck
    assert Card.get("2H") in deck
    assert Card.get("3C") not in deck
    assert "AS" not in deck
    assert None not in deck


def test_deck_custom_cards():
    cards = [Card.get("2S"), Card.get("3S"), Card.get("4S")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 3
    assert deck.top == Card.get("4S")
    assert deck.bottom == Card.get("2S")


def test_deck_draw_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47
    small = deck.draw_hand(size=2)
    assert len(small) == 2
    assert deck.cards_left == 45


def test_euchre_deck_contents():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    allowed = {1, 9, 10, 11, 12, 13}
    assert {c.value.value for c in deck.cards} == allowed
    assert len({str(c) for c in deck.cards}) == 24
    assert Card.get("2S") not in deck
    assert Card.get("9H") in deck
