import random
from collections import Counter

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES
    assert str(card) == "A♠"


def test_suits_all_sorted_by_value():
    suits = Card.Suits.all()
    assert [s.char for s in suits] == ["S", "C", "D", "H"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_values_all_and_dict():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    mapping = Card.Values.dict()
    assert mapping["T"] is Card.Values.TEN
    assert mapping["A"] is Card.Values.ACE
    assert set(mapping) == set("A23456789TJQK")


def test_card_get_valid_and_case_insensitive():
    card = Card.get("kh")
    assert card.value == Card.Values.KING
    assert card.suit == Card.Suits.HEARTS
    assert card == Card.get("KH")


@pytest.mark.parametrize("text", ["", "A", "ASS", "10S"])
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
    black = Card.get("AS")
    red = Card.get("AH")
    assert black.is_black and not black.is_red
    assert red.is_red and not red.is_black
    assert black.color == Card.BLACK
    assert red.color == Card.RED
    assert Card.get("2C").is_black
    assert Card.get("2D").is_red


def test_card_full_name_str_and_repr():
    card = Card.get("QD")
    assert card.full_name == "queen of diamonds"
    assert str(card) == "Q♦"
    assert repr(card) == "<Card:Q♦>"


def test_card_unicode_skips_knight():
    assert Card.get("AS").unicode == chr(0x1F0A1)
    assert Card.get("JS").unicode == chr(0x1F0AB)
    # queen and king skip the knight code point
    assert Card.get("QS").unicode == chr(0x1F0AD)
    assert Card.get("KS").unicode == chr(0x1F0AE)


def test_card_equality():
    assert Card.get("5H") == Card.get("5H")
    assert Card.get("5H") != Card.get("5D")
    assert Card.get("5H") != Card.get("6H")
    assert Card.get("5H") != "5H"


def test_card_ordering_by_value():
    two = Card.get("2S")
    king = Card.get("KH")
    assert two < king
    assert king > two
    assert two <= king
    assert king >= two
    assert not king < two
    assert two <= Card.get("2S")
    assert two >= Card.get("2S")


def test_deck_default_has_52_unique_cards():
    deck = Deck()
    assert deck.cards_left == 52
    counts = Counter(str(c) for c in deck.cards)
    assert len(counts) == 52
    assert all(n == 1 for n in counts.values())


def test_deck_unshuffled_order_and_top_bottom():
    deck = Deck(shuffle=False)
    assert deck.shuffles == 0
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_shuffle_counts_and_preserves_cards():
    random.seed(1234)
    deck = Deck(shuffle=False)
    before = sorted(str(c) for c in deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert sorted(str(c) for c in deck.cards) == before
    assert deck.cards_left == 52


def test_deck_default_constructor_shuffles_once():
    deck = Deck()
    assert deck.shuffles == 1
    assert deck.draws == 0


def test_deck_draw_single_and_multiple():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert isinstance(card, Card)
    assert card == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1

    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert cards == [Card.get("2S"), Card.get("3S"), Card.get("4S")]
    assert deck.cards_left == 48
    assert deck.draws == 4


def test_deck_draw_from_empty_raises():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_custom_cards_used():
    cards = [Card.get("2S"), Card.get("3S")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 2
    assert deck.top == Card.get("3S")
    assert deck.bottom == Card.get("2S")


def test_deck_contains():
    deck = Deck(shuffle=False)
    card = Card.get("9C")
    assert card in deck
    assert "9C" not in deck
    deck.draw(times=52)
    assert card not in deck


def test_deck_str_and_repr():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw(times=2)
    assert str(deck) == "<Deck[50]>"


def test_deck_draw_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47
    single = deck.draw_hand(size=1)
    assert len(single) == 1
    assert deck.cards_left == 46


def test_euchre_deck_contents():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert len({str(c) for c in deck.cards}) == 24
    allowed = {1, 9, 10, 11, 12, 13}
    assert {c.value.value for c in deck.cards} == allowed
    assert Card.get("2S") not in deck
    assert Card.get("AH") in deck
