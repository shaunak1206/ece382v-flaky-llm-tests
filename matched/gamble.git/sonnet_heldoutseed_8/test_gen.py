import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def card_strings(cards):
    return sorted(str(c) for c in cards)


# ---------------------------- Card ----------------------------


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES
    assert str(card) == "A♠"


def test_suits_all_sorted_and_complete():
    suits = Card.Suits.all()
    assert [s.char for s in suits] == ["S", "C", "D", "H"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_values_all_sorted_and_complete():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0].char == "A"
    assert values[-1].char == "K"


def test_suits_and_values_dict_keyed_by_char():
    suits = Card.Suits.dict()
    values = Card.Values.dict()
    assert set(suits) == {"S", "C", "D", "H"}
    assert suits["H"] is Card.Suits.HEARTS
    assert len(values) == 13
    assert values["T"] is Card.Values.TEN
    assert values["A"] is Card.Values.ACE


def test_card_get_valid_and_case_insensitive():
    card = Card.get("kh")
    assert card.value == Card.Values.KING
    assert card.suit == Card.Suits.HEARTS
    assert Card.get("TD") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)


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


def test_card_color_properties():
    black = Card.get("AS")
    red = Card.get("AH")
    assert black.color == Card.BLACK
    assert black.is_black and not black.is_red
    assert red.color == Card.RED
    assert red.is_red and not red.is_black
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
    assert Card.get("QS").unicode == chr(0x1F0AD)
    assert Card.get("KS").unicode == chr(0x1F0AE)
    assert Card.get("AH").unicode == chr(0x1F0B1)


def test_card_comparisons_by_value_only():
    low = Card.get("2S")
    high = Card.get("KH")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    assert not low > high
    assert not high < low
    # same value, different suit: neither less nor greater
    assert not Card.get("5S") < Card.get("5H")
    assert not Card.get("5S") > Card.get("5H")


def test_card_equality():
    assert Card.get("5S") == Card.get("5s")
    assert Card.get("5S") != Card.get("5H")
    assert Card.get("5S") != Card.get("6S")
    assert Card.get("5S") != "5S"
    assert Card.get("5S") <= Card.get("5S")
    assert Card.get("5S") >= Card.get("5S")


# ---------------------------- Deck ----------------------------


def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(set(card_strings(deck.cards))) == 52
    assert deck.shuffles == 0
    assert deck.draws == 0


def test_unshuffled_deck_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_shuffle_default_counts_and_preserves_cards():
    deck = Deck()
    assert deck.shuffles == 1
    before = card_strings(deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 4
    assert deck.cards_left == 52
    assert card_strings(deck.cards) == before


def test_deck_draw_single_returns_card_and_removes_it():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert isinstance(card, Card)
    assert card == Card.get("AS")
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
    assert deck.cards_left == 0
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_contains():
    deck = Deck(shuffle=False)
    assert Card.get("9C") in deck
    assert "9C" not in deck
    assert 5 not in deck
    small = Deck(cards=[Card.get("2S")], shuffle=False)
    assert Card.get("2S") in small
    assert Card.get("3S") not in small


def test_deck_custom_cards_and_str_repr():
    cards = [Card.get("2S"), Card.get("3S"), Card.get("4S")]
    deck = Deck(cards=list(cards), shuffle=False)
    assert deck.cards_left == 3
    assert str(deck) == "<Deck[3]>"
    assert repr(deck) == "<Deck[3]>"
    assert deck.top == Card.get("4S")
    assert deck.bottom == Card.get("2S")


def test_deck_draw_hand_default_and_custom_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47
    single = deck.draw_hand(size=1)
    assert isinstance(single, Hand)
    assert len(single) == 1
    assert deck.cards_left == 46


def test_euchre_deck_contents():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert len(set(card_strings(deck.cards))) == 24
    allowed = {"A", "9", "T", "J", "Q", "K"}
    assert all(c.value.char in allowed for c in deck.cards)
