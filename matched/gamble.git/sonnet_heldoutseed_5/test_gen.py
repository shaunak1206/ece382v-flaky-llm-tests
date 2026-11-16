import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES


def test_suits_all_sorted_by_value():
    suits = Card.Suits.all()
    assert [s.char for s in suits] == ["S", "C", "D", "H"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_values_all_and_dict():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    mapping = Card.Values.dict()
    assert mapping["A"] is Card.Values.ACE
    assert mapping["T"] is Card.Values.TEN
    assert mapping["K"] is Card.Values.KING
    assert len(mapping) == 13


def test_suits_dict_maps_char_to_suit():
    mapping = Card.Suits.dict()
    assert set(mapping) == {"S", "C", "D", "H"}
    assert mapping["H"] is Card.Suits.HEARTS


def test_card_get_valid_and_case_insensitive():
    card = Card.get("AS")
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES
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


def test_card_color_properties():
    black = Card.get("KC")
    red = Card.get("2H")
    assert black.color == Card.BLACK
    assert black.is_black and not black.is_red
    assert red.color == Card.RED
    assert red.is_red and not red.is_black


def test_card_string_representations():
    card = Card.get("QH")
    assert str(card) == "Q♥"
    assert repr(card) == "<Card:Q♥>"
    assert card.full_name == "queen of hearts"


def test_card_unicode_skips_knight():
    assert Card.get("AS").unicode == chr(0x1F0A1)
    assert Card.get("JS").unicode == chr(0x1F0AB)
    assert Card.get("QS").unicode == chr(0x1F0AD)
    assert Card.get("KS").unicode == chr(0x1F0AE)


def test_card_equality():
    assert Card.get("5D") == Card.get("5D")
    assert Card.get("5D") != Card.get("5H")
    assert Card.get("5D") != Card.get("6D")
    assert Card.get("5D") != "5D"


def test_card_ordering_by_value_only():
    low = Card.get("2S")
    high = Card.get("KH")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    assert not (high < low)
    # same value, same suit
    assert Card.get("9C") <= Card.get("9C")
    assert Card.get("9C") >= Card.get("9C")
    # same value, different suit: neither less nor greater
    assert not (Card.get("9C") < Card.get("9D"))
    assert not (Card.get("9C") > Card.get("9D"))


def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    unique = {str(c) for c in deck.cards}
    assert len(unique) == 52
    assert deck.shuffles == 0
    assert deck.draws == 0


def test_deck_unshuffled_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_shuffle_counts_and_preserves_cards():
    deck = Deck(shuffle=True)
    assert deck.shuffles == 1
    before = sorted(str(c) for c in deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 4
    assert sorted(str(c) for c in deck.cards) == before
    assert deck.cards_left == 52


def test_deck_with_custom_cards():
    cards = [Card.get("2S"), Card.get("3S"), Card.get("4S")]
    deck = Deck(cards=list(cards), shuffle=False)
    assert deck.cards_left == 3
    assert deck.top == Card.get("4S")
    assert deck.bottom == Card.get("2S")


def test_deck_empty_cards_list_falls_back_to_default():
    deck = Deck(cards=[], shuffle=False)
    assert deck.cards_left == 52


def test_deck_draw_single_and_multiple():
    deck = Deck(shuffle=False)
    first = deck.draw()
    assert isinstance(first, Card)
    assert first == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1

    several = deck.draw(times=3)
    assert isinstance(several, list)
    assert len(several) == 3
    assert deck.cards_left == 48
    assert deck.draws == 4
    assert first not in deck
    for card in several:
        assert card not in deck


def test_deck_draw_from_empty_raises():
    deck = Deck(cards=[Card.get("2S")], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_draw_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47

    single = deck.draw_hand(size=1)
    assert isinstance(single, Hand)
    assert len(single) == 1
    assert deck.cards_left == 46


def test_deck_contains_and_string():
    deck = Deck(shuffle=False)
    assert Card.get("QD") in deck
    assert "QD" not in deck
    assert 5 not in deck
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_euchre_deck_contents():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    allowed = {"A", "9", "T", "J", "Q", "K"}
    assert {c.value.char for c in deck.cards} == allowed
    assert len({str(c) for c in deck.cards}) == 24
    assert Card.get("9H") in deck
    assert Card.get("8H") not in deck
    assert Card.get("2S") not in deck
