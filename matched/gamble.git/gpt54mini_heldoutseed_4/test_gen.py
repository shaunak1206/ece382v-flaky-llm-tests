import random
import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, Hand


def test_suits_all_returns_sorted_suits():
    suits = Card.Suits.all()
    assert [s.value for s in suits] == [0, 1, 2, 3]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]


def test_suits_dict_maps_char_to_suit():
    suits = Card.Suits.dict()
    assert set(suits.keys()) == {"S", "C", "D", "H"}
    assert suits["S"] is Card.Suits.SPADES
    assert suits["H"] is Card.Suits.HEARTS


def test_values_all_returns_sorted_values():
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0] is Card.Values.ACE
    assert values[-1] is Card.Values.KING


def test_values_dict_maps_char_to_value():
    values = Card.Values.dict()
    assert values["A"] is Card.Values.ACE
    assert values["T"] is Card.Values.TEN
    assert values["K"] is Card.Values.KING


def test_card_get_parses_lowercase_input():
    card = Card.get("ah")
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.HEARTS
    assert str(card) == "A♥"
    assert repr(card) == "<Card:A♥>"


def test_card_get_rejects_too_long_text():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("10H")


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("1H")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("ASX"[0:2])  # "AS"


def test_card_color_and_boolean_properties():
    spade = Card.get("AS")
    heart = Card.get("AH")
    assert spade.color == Card.BLACK
    assert spade.is_black is True
    assert spade.is_red is False
    assert heart.color == Card.RED
    assert heart.is_red is True
    assert heart.is_black is False


def test_card_unicode_for_queen_and_king_skips_knight():
    queen = Card.get("QH")
    king = Card.get("KS")
    assert queen.unicode == chr(Card.Suits.HEARTS.unicode + 12 + 1)
    assert king.unicode == chr(Card.Suits.SPADES.unicode + 13 + 1)


def test_card_comparisons_use_only_value():
    two_clubs = Card.get("2C")
    ten_hearts = Card.get("TH")
    ten_spades = Card.get("TS")
    assert two_clubs < ten_hearts
    assert ten_hearts > two_clubs
    assert ten_hearts <= ten_spades
    assert ten_hearts >= ten_spades
    assert ten_hearts == ten_spades
    assert ten_hearts != two_clubs
    assert ten_hearts != "not a card"


def test_default_deck_has_52_cards_and_known_top_bottom():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert isinstance(deck.top, Card)
    assert isinstance(deck.bottom, Card)
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KH")
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_contains_only_card_objects():
    deck = Deck(shuffle=False)
    assert Card.get("AS") in deck
    assert "AS" not in deck
    assert 123 not in deck


def test_deck_draw_single_card_updates_counts_and_removes_top():
    deck = Deck(shuffle=False)
    top_before = deck.top
    drawn = deck.draw()
    assert drawn == top_before
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert drawn not in deck


def test_deck_draw_multiple_cards_returns_list_in_draw_order():
    deck = Deck(shuffle=False)
    top1 = deck.top
    top2 = deck.cards[-2]
    drawn = deck.draw(times=2)
    assert isinstance(drawn, list)
    assert drawn == [top1, top2]
    assert deck.draws == 2
    assert deck.cards_left == 50


def test_deck_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47
    assert deck.draws == 5


def test_deck_shuffle_calls_random_shuffle_requested_number_of_times(monkeypatch):
    deck = Deck(shuffle=False)
    calls = []

    def fake_shuffle(cards):
        calls.append(len(cards))
        cards.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    deck.shuffle(times=3)

    assert deck.shuffles == 3
    assert calls == [52, 52, 52]
    assert deck.top == Card.get("AS")  # reversed three times from initial default ordering


def test_deck_initialized_with_custom_cards_uses_given_list_without_shuffling():
    cards = [Card.get("AS"), Card.get("KH"), Card.get("QC")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards is cards
    assert deck.cards_left == 3
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("QC")
