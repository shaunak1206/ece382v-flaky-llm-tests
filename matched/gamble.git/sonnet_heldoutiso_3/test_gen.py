import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, Hand


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES


def test_suits_all_sorted_and_complete():
    suits = Card.Suits.all()
    assert [s.char for s in suits] == ["S", "C", "D", "H"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_values_all_sorted_and_dict():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    mapping = Card.Values.dict()
    assert mapping["T"] is Card.Values.TEN
    assert mapping["A"] is Card.Values.ACE
    assert set(mapping) == set("A23456789TJQK")


def test_card_get_valid_and_case_insensitive():
    card = Card.get("KH")
    assert card.value == Card.Values.KING
    assert card.suit == Card.Suits.HEARTS
    assert Card.get("td") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)


@pytest.mark.parametrize("text", ["", "A", "10S", "ASX"])
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
    assert Card.get("2C").color == Card.BLACK
    assert Card.get("3D").is_red and not Card.get("3D").is_black
    assert Card.get("4H").color == Card.RED


def test_card_full_name_str_repr():
    card = Card.get("QD")
    assert card.full_name == "queen of diamonds"
    assert str(card) == "Q♦"
    assert repr(card) == "<Card:Q♦>"


def test_card_unicode_skips_knight():
    assert Card.get("AS").unicode == chr(127137)
    assert Card.get("JS").unicode == chr(127136 + 11)
    # queen and king skip the knight code point
    assert Card.get("QS").unicode == chr(127136 + 13)
    assert Card.get("KS").unicode == chr(127136 + 14)


def test_card_comparisons_by_value():
    low = Card.get("2S")
    high = Card.get("KH")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    assert not low > high
    assert not high < low
    assert Card.get("5S") <= Card.get("5S")
    assert Card.get("5S") >= Card.get("5S")


def test_card_equality():
    assert Card.get("AS") == Card.get("as")
    assert Card.get("AS") != Card.get("AH")
    assert Card.get("AS") != Card.get("2S")
    assert Card.get("AS") != "AS"


def test_deck_default_has_52_unique_cards():
    deck = Deck()
    assert deck.cards_left == 52
    assert len({str(c) for c in deck.cards}) == 52
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_unshuffled_order_and_counters():
    deck = Deck(shuffle=False)
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_shuffle_counts_and_preserves_cards():
    deck = Deck(shuffle=False)
    before = sorted(str(c) for c in deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert sorted(str(c) for c in deck.cards) == before
    assert Deck().shuffles == 1


def test_deck_custom_cards_and_empty_list_falls_back_to_default():
    cards = [Card.get("2S"), Card.get("3H")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 2
    assert deck.top == Card.get("3H")
    assert deck.bottom == Card.get("2S")
    assert Deck(cards=[], shuffle=False).cards_left == 52


def test_deck_draw_single_and_multiple():
    deck = Deck(shuffle=False)
    first = deck.draw()
    assert isinstance(first, Card)
    assert first == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1

    group = deck.draw(times=3)
    assert isinstance(group, list)
    assert len(group) == 3
    assert deck.cards_left == 48
    assert deck.draws == 4
    assert group[0] == Card.get("2S")


def test_deck_draw_from_empty_raises():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_draw_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47
    smaller = deck.draw_hand(size=2)
    assert len(smaller) == 2
    assert deck.cards_left == 45


def test_deck_contains():
    deck = Deck(shuffle=False)
    assert Card.get("AS") in deck
    drawn = deck.draw()
    assert drawn not in deck
    assert "KH" not in deck
    assert None not in deck
