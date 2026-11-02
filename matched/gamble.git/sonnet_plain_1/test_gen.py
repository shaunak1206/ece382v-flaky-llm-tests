import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck


def test_suits_all_sorted_by_value():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_values_all_has_thirteen_sorted_values():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0].char == "A"
    assert values[-1].char == "K"


def test_suits_and_values_dict_keyed_by_char():
    assert set(Card.Suits.dict().keys()) == {"S", "C", "D", "H"}
    assert Card.Suits.dict()["H"] is Card.Suits.HEARTS
    assert set(Card.Values.dict().keys()) == set("A23456789TJQK")
    assert Card.Values.dict()["T"] is Card.Values.TEN


def test_card_default_constructor_is_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES


def test_card_get_parses_text_case_insensitively():
    card = Card.get("kh")
    assert card.value == Card.Values.KING
    assert card.suit == Card.Suits.HEARTS
    assert Card.get("TD") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)


def test_card_get_wrong_length_raises():
    with pytest.raises(InvalidCard):
        Card.get("AAS")
    with pytest.raises(InvalidCard):
        Card.get("A")
    with pytest.raises(InvalidCard):
        Card.get("")


def test_card_get_invalid_value_or_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("1S")
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_card_color_properties():
    assert Card.get("AS").is_black
    assert Card.get("2C").is_black
    assert Card.get("3D").is_red
    assert Card.get("4H").is_red
    assert not Card.get("4H").is_black
    assert Card.get("AS").color == Card.BLACK
    assert Card.get("AH").color == Card.RED


def test_card_full_name_str_and_repr():
    card = Card.get("QH")
    assert card.full_name == "queen of hearts"
    assert str(card) == "Q♥"
    assert repr(card) == "<Card:Q♥>"


def test_card_unicode_skips_knight():
    assert Card.get("AS").unicode == chr(127137)
    assert Card.get("JS").unicode == chr(127136 + 11)
    # queen and king are shifted by one to skip the knight card
    assert Card.get("QS").unicode == chr(127136 + 12 + 1)
    assert Card.get("KS").unicode == chr(127136 + 13 + 1)


def test_card_equality_and_non_card_comparison():
    assert Card.get("5C") == Card.get("5C")
    assert Card.get("5C") != Card.get("5D")
    assert Card.get("5C") != Card.get("6C")
    assert Card.get("5C") != "5C"
    assert Card.get("5C") != 5


def test_card_ordering_compares_value_only():
    low = Card.get("2S")
    high = Card.get("KH")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    # same value, different suit: neither strictly less nor greater
    a, b = Card.get("7S"), Card.get("7H")
    assert not a < b
    assert not a > b
    assert not a <= b
    assert not a >= b
    assert a <= Card.get("7S")
    assert a >= Card.get("7S")


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
    shuffled = Deck()
    assert shuffled.shuffles == 1
    assert shuffled.cards_left == 52
    deck = Deck(shuffle=False)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert deck.cards_left == 52
    assert Deck(shuffle=False).shuffles == 0


def test_deck_shuffle_preserves_cards():
    plain = Deck(shuffle=False)
    shuffled = Deck(shuffle=False)
    shuffled.shuffle()
    key = lambda c: (c.suit.value, c.value.value)
    assert sorted(plain.cards, key=key) == sorted(shuffled.cards, key=key)


def test_deck_draw_single_card_removes_top():
    deck = Deck(shuffle=False)
    top = deck.top
    drawn = deck.draw()
    assert isinstance(drawn, Card)
    assert drawn == top
    assert deck.cards_left == 51
    assert deck.draws == 1
    assert drawn not in deck


def test_deck_draw_multiple_returns_list_in_order():
    deck = Deck(shuffle=False)
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert drawn == [Card.get("AS"), Card.get("2S"), Card.get("3S")]
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_deck_draw_from_empty_raises():
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
    small = deck.draw_hand(size=1)
    assert len(small) == 1
    assert deck.cards_left == 46


def test_deck_custom_cards_and_empty_list_falls_back_to_default():
    cards = [Card.get("2S"), Card.get("3S")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 2
    assert deck.top == Card.get("3S")
    assert deck.bottom == Card.get("2S")
    assert Deck(cards=[], shuffle=False).cards_left == 52


def test_deck_contains_str_and_repr():
    deck = Deck(shuffle=False)
    assert Card.get("9D") in deck
    assert "9D" not in deck
    assert 5 not in deck
    small = Deck(cards=[Card.get("2S")], shuffle=False)
    assert Card.get("3S") not in small
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"
