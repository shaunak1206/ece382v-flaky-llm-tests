import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"


def test_suits_and_values_all_sorted_and_complete():
    suits = Card.Suits.all()
    values = Card.Values.all()
    assert [s.char for s in suits] == ["S", "C", "D", "H"]
    assert [v.value for v in values] == list(range(1, 14))
    assert set(Card.Suits.dict().keys()) == {"S", "C", "D", "H"}
    assert Card.Values.dict()["T"].name == "ten"


def test_card_get_valid_parsing():
    card = Card.get("KH")
    assert card.value == Card.Values.KING
    assert card.suit == Card.Suits.HEARTS
    assert card.full_name == "king of hearts"


def test_card_get_is_case_insensitive():
    assert Card.get("td") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)


@pytest.mark.parametrize("text", ["", "A", "10S", "ASS"])
def test_card_get_wrong_length_raises(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_get_invalid_value_raises():
    with pytest.raises(InvalidCard):
        Card.get("1S")


def test_card_get_invalid_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_card_colors():
    assert Card.get("AS").is_black and not Card.get("AS").is_red
    assert Card.get("2C").is_black
    assert Card.get("3D").is_red and not Card.get("3D").is_black
    assert Card.get("4H").color == Card.RED


def test_card_unicode_characters():
    assert Card.get("AS").unicode == chr(0x1F0A1)
    # queen and king skip the knight card
    assert Card.get("QS").unicode == chr(0x1F0AD)
    assert Card.get("KS").unicode == chr(0x1F0AE)
    assert Card.get("AH").unicode == chr(0x1F0B1)


def test_card_equality():
    assert Card.get("5H") == Card.get("5H")
    assert Card.get("5H") != Card.get("5D")
    assert Card.get("5H") != Card.get("6H")
    assert Card.get("5H") != "5H"


def test_card_ordering_by_value_only():
    low = Card.get("2S")
    high = Card.get("KD")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    # same value, different suit: neither < nor >, but not equal either
    a, b = Card.get("7S"), Card.get("7H")
    assert not a < b and not a > b
    assert not a <= b and not a >= b
    assert a <= Card.get("7S")
    assert a >= Card.get("7S")


def test_deck_default_has_52_unique_cards():
    deck = Deck()
    assert deck.cards_left == 52
    assert len({str(c) for c in deck.cards}) == 52


def test_deck_unshuffled_order_and_top_bottom():
    deck = Deck(shuffle=False)
    assert deck.shuffles == 0
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_default_shuffles_once():
    deck = Deck()
    assert deck.shuffles == 1
    assert deck.draws == 0


def test_deck_shuffle_multiple_times_preserves_cards():
    deck = Deck(shuffle=False)
    before = sorted(str(c) for c in deck.cards)
    random.seed(1234)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert sorted(str(c) for c in deck.cards) == before
    assert deck.cards_left == 52


def test_deck_draw_single_card():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert isinstance(card, Card)
    assert card == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1
    assert card not in deck


def test_deck_draw_multiple_cards_returns_list():
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


def test_deck_custom_cards_and_contains():
    cards = [Card.get("2S"), Card.get("3H")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 2
    assert Card.get("2S") in deck
    assert Card.get("4C") not in deck
    assert "2S" not in deck


def test_deck_str_and_repr():
    deck = Deck()
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_deck_draw_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47
    small = deck.draw_hand(size=1)
    assert len(small) == 1
    assert deck.cards_left == 46


def test_euchre_deck_contents():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert Card.get("9S") in deck
    assert Card.get("AH") in deck
    assert Card.get("KD") in deck
    assert Card.get("8S") not in deck
    assert Card.get("2C") not in deck
