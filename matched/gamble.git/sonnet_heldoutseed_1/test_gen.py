import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck


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


def test_suits_and_values_dict_lookup():
    assert Card.Suits.dict()["H"] is Card.Suits.HEARTS
    assert Card.Values.dict()["T"] is Card.Values.TEN
    assert len(Card.Values.dict()) == 13
    assert len(Card.Suits.dict()) == 4


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_card_get_valid_and_case_insensitive():
    card = Card.get("kh")
    assert card.value is Card.Values.KING
    assert card.suit is Card.Suits.HEARTS
    assert Card.get("TD") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)


@pytest.mark.parametrize("text", ["", "A", "10S", "ASS"])
def test_card_get_wrong_length_raises(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_get_invalid_value_raises():
    with pytest.raises(InvalidCard):
        Card.get("XS")


def test_card_get_invalid_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_card_color_properties():
    black = Card.get("AS")
    red = Card.get("QH")
    assert black.is_black and not black.is_red
    assert red.is_red and not red.is_black
    assert black.color == Card.BLACK
    assert red.color == Card.RED
    assert Card.get("5C").is_black
    assert Card.get("5D").is_red


def test_card_full_name_str_and_repr():
    card = Card.get("AS")
    assert card.full_name == "ace of spades"
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"


def test_card_unicode_skips_knight():
    assert Card.get("AS").unicode == chr(0x1F0A1)
    assert Card.get("JS").unicode == chr(0x1F0AB)
    assert Card.get("QS").unicode == chr(0x1F0AD)
    assert Card.get("KS").unicode == chr(0x1F0AE)


def test_card_comparisons_by_value_only():
    two = Card.get("2H")
    three = Card.get("3S")
    assert two < three
    assert three > two
    assert two <= three
    assert three >= two
    assert Card.get("AS") < two  # ace is low


def test_card_equality_considers_suit_and_value():
    assert Card.get("5H") == Card.get("5H")
    assert Card.get("5H") != Card.get("5S")
    assert Card.get("5H") != Card.get("6H")
    assert Card.get("5H") != "5H"
    # same value different suit: neither < nor >, but <= is False without equality
    assert not (Card.get("5H") < Card.get("5S"))
    assert not (Card.get("5H") <= Card.get("5S"))


def test_deck_default_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len({str(c) for c in deck.cards}) == 52
    assert deck.shuffles == 0
    assert deck.draws == 0


def test_deck_unshuffled_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_default_shuffles_once_and_keeps_cards():
    deck = Deck()
    assert deck.shuffles == 1
    assert deck.cards_left == 52
    assert len({str(c) for c in deck.cards}) == 52


def test_deck_shuffle_multiple_times_counts():
    deck = Deck(shuffle=False)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert deck.cards_left == 52


def test_deck_custom_cards():
    cards = [Card.get("2S"), Card.get("3S"), Card.get("4S")]
    deck = Deck(cards=list(cards), shuffle=False)
    assert deck.cards_left == 3
    assert deck.top == Card.get("4S")
    assert deck.bottom == Card.get("2S")


def test_deck_draw_single_card():
    deck = Deck(shuffle=False)
    top = deck.top
    drawn = deck.draw()
    assert isinstance(drawn, Card)
    assert drawn == top
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_deck_draw_multiple_returns_list():
    deck = Deck(shuffle=False)
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert drawn == [Card.get("AS"), Card.get("2S"), Card.get("3S")]
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_deck_draw_from_empty_raises():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
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


def test_deck_contains_and_str():
    deck = Deck(shuffle=False)
    card = Card.get("AS")
    assert card in deck
    assert "AS" not in deck
    deck.draw()
    assert card not in deck
    assert str(deck) == "<Deck[51]>"
    assert repr(deck) == "<Deck[51]>"
