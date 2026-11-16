"""Unit tests for Card and Deck in gamble/models/cards.py."""
import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_default_card_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A\u2660"


def test_card_constructor_stores_value_and_suit():
    card = Card(value=Card.Values.QUEEN, suit=Card.Suits.HEARTS)
    assert card.value is Card.Values.QUEEN
    assert card.suit is Card.Suits.HEARTS


def test_card_get_parses_valid_text():
    card = Card.get("AS")
    assert card == Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)


def test_card_get_is_case_insensitive():
    assert Card.get("th") == Card.get("TH")
    assert Card.get("th").value is Card.Values.TEN
    assert Card.get("th").suit is Card.Suits.HEARTS


@pytest.mark.parametrize("text", ["", "A", "AS1", "10S"])
def test_card_get_rejects_wrong_length(text):
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get(text)


@pytest.mark.parametrize("text", ["1S", "XS", "ZH"])
def test_card_get_rejects_invalid_value(text):
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get(text)


@pytest.mark.parametrize("text", ["AX", "A1", "K?"])
def test_card_get_rejects_invalid_suit(text):
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get(text)


@pytest.mark.parametrize(
    "text, is_black",
    [("AS", True), ("CK", True), ("2D", False), ("TH", False)],
)
def test_color_and_is_black_and_is_red(text, is_black):
    card = Card.get(text)
    assert card.is_black is is_black
    assert card.is_red is (not is_black)
    assert card.color == (Card.BLACK if is_black else Card.RED)


def test_full_name():
    assert Card.get("AS").full_name == "ace of spades"
    assert Card.get("KH").full_name == "king of hearts"
    assert Card.get("TD").full_name == "ten of diamonds"


def test_str_and_repr():
    card = Card.get("TH")
    assert str(card) == "T\u2665"
    assert repr(card) == "<Card:T\u2665>"


def test_unicode_playing_card_characters():
    # Ace of spades is U+1F0A1; the queen skips the knight, so queen of spades is U+1F0AD.
    assert Card.get("AS").unicode == "\U0001F0A1"
    assert Card.get("QS").unicode == "\U0001F0AD"
    assert Card.get("KS").unicode == "\U0001F0AE"


def test_equality_compares_value_and_suit():
    assert Card.get("9C") == Card.get("9C")
    assert Card.get("9C") != Card.get("9D")
    assert Card.get("9C") != Card.get("8C")


def test_equality_with_non_card_is_false():
    assert (Card.get("AS") == "AS") is False
    assert (Card.get("AS") != "AS") is True


def test_ordering_uses_value_only():
    ace = Card.get("AS")
    two = Card.get("2H")
    king = Card.get("KD")
    assert ace < two < king
    assert king > two > ace
    assert ace <= ace
    assert king >= king
    assert ace <= two
    assert king >= two


def test_sorting_cards_orders_by_value():
    cards = [Card.get("KS"), Card.get("AH"), Card.get("7D"), Card.get("TC")]
    assert [str(c.value.char) for c in sorted(cards)] == ["A", "7", "T", "K"]


def test_suits_all_is_sorted_by_value():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]


def test_suits_dict_maps_char_to_suit():
    suits = Card.Suits.dict()
    assert set(suits.keys()) == {"S", "C", "D", "H"}
    assert suits["H"] is Card.Suits.HEARTS


def test_values_all_is_sorted_and_complete():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0] is Card.Values.ACE
    assert values[-1] is Card.Values.KING


def test_values_dict_maps_char_to_value():
    values = Card.Values.dict()
    assert len(values) == 13
    assert values["T"] is Card.Values.TEN
    assert values["A"] is Card.Values.ACE


def test_default_deck_has_52_unique_cards():
    deck = Deck()
    assert deck.cards_left == 52
    assert len({str(card) for card in deck.cards}) == 52


def test_unshuffled_deck_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")
    assert deck.shuffles == 0


def test_deck_with_given_cards_uses_them_and_skips_shuffle():
    cards = [Card.get("2C"), Card.get("3C"), Card.get("4C")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards == cards
    assert deck.cards_left == 3


def test_draw_single_card_returns_top_card():
    deck = Deck(shuffle=False)
    top = deck.top
    drawn = deck.draw()
    assert drawn == top
    assert isinstance(drawn, Card)
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_cards_returns_list():
    deck = Deck(shuffle=False)
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert len(drawn) == 3
    assert all(isinstance(c, Card) for c in drawn)
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_from_empty_deck_raises():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert hand.size == 5
    assert len(hand) == 5
    assert deck.cards_left == 47
    assert deck.draws == 5


def test_shuffle_increments_shuffle_count():
    deck = Deck(shuffle=False)
    deck.shuffle()
    assert deck.shuffles == 1
    deck.shuffle(times=3)
    assert deck.shuffles == 4


def test_shuffle_preserves_cards():
    deck = Deck(shuffle=False)
    before = sorted(str(c) for c in deck.cards)
    deck.shuffle()
    after = sorted(str(c) for c in deck.cards)
    assert before == after


def test_contains_card_and_non_card():
    deck = Deck(shuffle=False)
    assert Card.get("QH") in deck
    assert "QH" not in deck
    drawn = deck.draw()
    assert drawn not in deck


def test_deck_str_and_repr():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_euchre_deck_has_24_cards_without_low_cards():
    deck = EuchreDeck()
    assert isinstance(deck, Deck)
    assert deck.cards_left == 24
    assert len({str(c) for c in deck.cards}) == 24
    assert all(c.value.value >= 9 or c.value.value == 1 for c in deck.cards)
    assert Card.get("2S") not in deck
    assert Card.get("AS") in deck
