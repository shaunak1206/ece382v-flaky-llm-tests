import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck


def _card_keys(cards):
    return sorted(str(c) for c in cards)


def test_card_defaults_to_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES
    assert card.full_name == "ace of spades"


def test_suits_and_values_all_are_sorted_and_complete():
    suits = Card.Suits.all()
    values = Card.Values.all()
    assert [s.char for s in suits] == ["S", "C", "D", "H"]
    assert [v.value for v in values] == list(range(1, 14))
    assert Card.Suits.dict()["H"] is Card.Suits.HEARTS
    assert Card.Values.dict()["T"] is Card.Values.TEN


def test_card_get_parses_text_case_insensitively():
    card = Card.get("as")
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES
    assert Card.get("TD") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)


@pytest.mark.parametrize("text", ["", "A", "ASS", "10S"])
def test_card_get_wrong_length_raises(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_get_invalid_value_or_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("1S")
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_card_colors():
    assert Card.get("AS").is_black
    assert Card.get("2C").is_black
    assert Card.get("3D").is_red
    assert Card.get("KH").is_red
    assert not Card.get("KH").is_black
    assert Card.get("AS").color == Card.BLACK
    assert Card.get("AH").color == Card.RED


def test_card_str_repr_and_full_name():
    card = Card.get("QH")
    assert str(card) == "Q\u2665"
    assert repr(card) == "<Card:Q\u2665>"
    assert card.full_name == "queen of hearts"


def test_card_unicode_skips_knight():
    assert Card.get("AS").unicode == chr(0x1F0A1)
    assert Card.get("JS").unicode == chr(0x1F0AB)
    # queen and king skip the knight code point
    assert Card.get("QS").unicode == chr(0x1F0AD)
    assert Card.get("KS").unicode == chr(0x1F0AE)


def test_card_equality():
    assert Card.get("5H") == Card.get("5h")
    assert Card.get("5H") != Card.get("5S")
    assert Card.get("5H") != Card.get("6H")
    assert Card.get("5H") != "5H"


def test_card_ordering_by_value_only():
    low = Card.get("2S")
    high = Card.get("KD")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    assert not high < low
    # same value, different suit: neither < nor > and not equal
    assert not Card.get("2S") < Card.get("2H")
    assert not Card.get("2S") > Card.get("2H")
    assert not Card.get("2S") <= Card.get("2H")
    # ace is low
    assert Card.get("AS") < Card.get("2S")
    assert Card.get("7C") <= Card.get("7C")
    assert Card.get("7C") >= Card.get("7C")


def test_default_deck_unshuffled_order_and_contents():
    random.seed(0)
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")
    assert len(set(str(c) for c in deck.cards)) == 52
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_default_deck_shuffled_keeps_all_cards():
    random.seed(1234)
    deck = Deck()
    assert deck.shuffles == 1
    assert deck.cards_left == 52
    reference = Deck(shuffle=False)
    assert _card_keys(deck.cards) == _card_keys(reference.cards)


def test_shuffle_is_deterministic_with_same_seed_and_counts_times():
    random.seed(42)
    first = Deck()
    first.shuffle(times=3)
    assert first.shuffles == 4

    random.seed(42)
    second = Deck()
    second.shuffle(times=3)

    assert [str(c) for c in first.cards] == [str(c) for c in second.cards]


def test_empty_card_list_builds_default_deck():
    random.seed(0)
    deck = Deck(cards=[], shuffle=False)
    assert deck.cards_left == 52


def test_custom_cards_without_shuffle_preserve_order():
    random.seed(0)
    cards = [Card.get("2S"), Card.get("3H"), Card.get("4D")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 3
    assert deck.bottom == Card.get("2S")
    assert deck.top == Card.get("4D")


def test_draw_single_and_multiple():
    random.seed(0)
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


def test_draw_from_empty_deck_raises():
    random.seed(0)
    deck = Deck(cards=[Card.get("2S")], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_contains():
    random.seed(0)
    deck = Deck(shuffle=False)
    ace = Card.get("AS")
    assert ace in deck
    assert "AS" not in deck
    assert None not in deck
    deck.draw()
    assert ace not in deck


def test_draw_hand_default_and_custom_size():
    random.seed(0)
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert len(hand) == 5
    assert hand.size == 5
    assert deck.cards_left == 47
    # A-5 of spades is a straight flush
    assert hand.rank.name == "straight flush"

    single = deck.draw_hand(size=1)
    assert len(single) == 1
    assert deck.cards_left == 46


def test_euchre_deck_contents():
    random.seed(7)
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert deck.shuffles == 1
    expected = [
        Card(value=v, suit=s)
        for s in Card.Suits.all()
        for v in Card.Values.all()
        if v.value >= 9 or v.value == 1
    ]
    assert _card_keys(deck.cards) == _card_keys(expected)
    assert Card.get("2S") not in deck
    assert Card.get("9S") in deck
    assert Card.get("AH") in deck
