import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES


def test_card_get_parses_valid_text():
    card = Card.get("KH")
    assert card.value == Card.Values.KING
    assert card.suit == Card.Suits.HEARTS


def test_card_get_is_case_insensitive():
    assert Card.get("td") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)


def test_card_get_wrong_length_raises():
    with pytest.raises(InvalidCard):
        Card.get("10S")
    with pytest.raises(InvalidCard):
        Card.get("A")
    with pytest.raises(InvalidCard):
        Card.get("")


def test_card_get_invalid_value_raises():
    with pytest.raises(InvalidCard):
        Card.get("XS")


def test_card_get_invalid_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_card_color_properties():
    assert Card.get("AS").is_black
    assert not Card.get("AS").is_red
    assert Card.get("2C").is_black
    assert Card.get("3D").is_red
    assert Card.get("4H").is_red
    assert not Card.get("4H").is_black
    assert Card.get("3D").color == Card.RED
    assert Card.get("2C").color == Card.BLACK


def test_card_full_name_str_repr():
    card = Card.get("QH")
    assert card.full_name == "queen of hearts"
    assert str(card) == "Q♥"
    assert repr(card) == "<Card:Q♥>"


def test_card_unicode_ace_and_skip_knight():
    assert Card.get("AS").unicode == chr(127137)
    assert Card.get("JS").unicode == chr(127136 + 11)
    # queen and king skip the knight card
    assert Card.get("QS").unicode == chr(127136 + 13)
    assert Card.get("KS").unicode == chr(127136 + 14)


def test_card_comparisons_by_value_only():
    low = Card.get("2S")
    high = Card.get("KD")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    assert not high < low
    assert not low > high


def test_card_equality_requires_suit_and_value():
    assert Card.get("5C") == Card.get("5C")
    assert Card.get("5C") != Card.get("5D")
    assert Card.get("5C") != Card.get("6C")
    assert Card.get("5C") != "5C"
    # same value, different suit still satisfies <= and >=
    assert Card.get("5C") <= Card.get("5D")
    assert Card.get("5C") >= Card.get("5D")


def test_suits_and_values_enumerations():
    assert [s.char for s in Card.Suits.all()] == ["S", "C", "D", "H"]
    assert [v.value for v in Card.Values.all()] == list(range(1, 14))
    assert set(Card.Suits.dict()) == {"S", "C", "D", "H"}
    assert Card.Values.dict()["T"] == Card.Values.TEN
    assert len(Card.Values.dict()) == 13


def test_deck_default_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(set((c.suit.char, c.value.char) for c in deck.cards)) == 52
    assert deck.shuffles == 0
    assert deck.draws == 0


def test_deck_unshuffled_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_shuffle_flag_and_counter():
    deck = Deck()
    assert deck.shuffles == 1
    assert deck.cards_left == 52
    deck.shuffle(times=3)
    assert deck.shuffles == 4
    assert deck.cards_left == 52


def test_deck_shuffle_preserves_cards():
    plain = Deck(shuffle=False)
    shuffled = Deck(shuffle=True)
    key = lambda c: (c.suit.value, c.value.value)  # noqa: E731
    assert sorted(plain.cards, key=key) == sorted(shuffled.cards, key=key)


def test_deck_draw_single_card():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert isinstance(card, Card)
    assert card == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1
    assert card not in deck


def test_deck_draw_multiple_cards():
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert cards == [Card.get("AS"), Card.get("2S"), Card.get("3S")]
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_deck_draw_from_empty_deck_raises():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
    assert deck.cards_left == 0
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_draw_hand_default_and_custom_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert len(hand) == 5
    assert deck.cards_left == 47
    small = deck.draw_hand(size=2)
    assert len(small) == 2
    assert deck.cards_left == 45
    assert deck.draws == 7


def test_deck_custom_cards_and_empty_list_falls_back():
    custom = [Card.get("2H"), Card.get("3H")]
    deck = Deck(cards=custom, shuffle=False)
    assert deck.cards_left == 2
    assert deck.top == Card.get("3H")
    assert deck.bottom == Card.get("2H")
    assert Deck(cards=[], shuffle=False).cards_left == 52


def test_deck_contains_str_repr():
    deck = Deck(shuffle=False)
    assert Card.get("QD") in deck
    assert "QD" not in deck
    assert None not in deck
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"
