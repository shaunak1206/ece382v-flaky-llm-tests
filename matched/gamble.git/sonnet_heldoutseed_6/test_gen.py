import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_card_defaults_to_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES


def test_suits_all_sorted_and_complete():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_values_all_sorted_and_dict():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    by_char = Card.Values.dict()
    assert by_char["T"] is Card.Values.TEN
    assert by_char["A"] is Card.Values.ACE
    assert set(by_char) == set("A23456789TJQK")


def test_suits_dict_maps_char_to_suit():
    d = Card.Suits.dict()
    assert set(d) == {"S", "C", "D", "H"}
    assert d["H"] is Card.Suits.HEARTS


def test_card_get_valid_and_case_insensitive():
    card = Card.get("Kh")
    assert card.value == Card.Values.KING
    assert card.suit == Card.Suits.HEARTS
    assert Card.get("KH") == card


@pytest.mark.parametrize("text", ["", "A", "ASD", "1S", "AX", "ZZ"])
def test_card_get_invalid_raises(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_colors():
    assert Card.get("AS").is_black
    assert Card.get("2C").is_black
    assert Card.get("3D").is_red
    assert Card.get("4H").is_red
    assert Card.get("AS").color == Card.BLACK
    assert Card.get("4H").color == Card.RED
    assert not Card.get("4H").is_black
    assert not Card.get("AS").is_red


def test_card_string_representations():
    card = Card.get("TD")
    assert str(card) == "T♦"
    assert repr(card) == "<Card:T♦>"
    assert card.full_name == "ten of diamonds"


def test_card_unicode_skips_knight():
    assert Card.get("AS").unicode == chr(0x1F0A1)
    assert Card.get("JS").unicode == chr(0x1F0AB)
    # queen and king skip the knight card
    assert Card.get("QS").unicode == chr(0x1F0AD)
    assert Card.get("KS").unicode == chr(0x1F0AE)


def test_card_comparisons_by_value():
    low = Card.get("2S")
    high = Card.get("KH")
    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    assert not high < low
    assert not low > high


def test_card_equality_considers_suit_and_value():
    assert Card.get("5C") == Card.get("5C")
    assert Card.get("5C") != Card.get("5D")
    assert Card.get("5C") != Card.get("6C")
    assert Card.get("5C") != "5C"
    assert Card.get("5C") <= Card.get("5C")
    assert Card.get("5C") >= Card.get("5C")


def test_deck_default_has_52_unique_cards():
    deck = Deck()
    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    assert len({str(c) for c in deck.cards}) == 52


def test_deck_unshuffled_order_and_counters():
    deck = Deck(shuffle=False)
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_default_shuffles_once():
    deck = Deck()
    assert deck.shuffles == 1


def test_deck_shuffle_multiple_times_preserves_cards():
    deck = Deck(shuffle=False)
    before = sorted(str(c) for c in deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert sorted(str(c) for c in deck.cards) == before


def test_deck_custom_cards_used():
    cards = [Card.get("2S"), Card.get("3S"), Card.get("4S")]
    deck = Deck(cards=list(cards), shuffle=False)
    assert deck.cards_left == 3
    assert deck.top == Card.get("4S")
    assert deck.bottom == Card.get("2S")


def test_deck_empty_cards_list_falls_back_to_full_deck():
    deck = Deck(cards=[], shuffle=False)
    assert deck.cards_left == 52


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
    single = deck.draw_hand(size=1)
    assert len(single) == 1
    assert deck.cards_left == 46


def test_deck_contains_and_str():
    deck = Deck(shuffle=False)
    assert Card.get("QD") in deck
    assert "QD" not in deck
    assert 5 not in deck
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw(times=2)
    assert str(deck) == "<Deck[50]>"


def test_euchre_deck_contents():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    allowed = {"A", "9", "T", "J", "Q", "K"}
    assert {c.value.char for c in deck.cards} == allowed
    assert len({str(c) for c in deck.cards}) == 24
    assert Card.get("2S") not in deck
    assert Card.get("9H") in deck
