import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, Hand, EuchreDeck


def test_suits_all_returns_ordered_suits():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]


def test_suits_dict_maps_characters_to_suits():
    suit_map = Card.Suits.dict()
    assert set(suit_map.keys()) == {"S", "C", "D", "H"}
    assert suit_map["S"] is Card.Suits.SPADES
    assert suit_map["H"] is Card.Suits.HEARTS


def test_values_all_returns_ordered_values():
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert [v.char for v in values] == ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


def test_values_dict_maps_characters_to_values():
    value_map = Card.Values.dict()
    assert set(value_map.keys()) == {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}
    assert value_map["A"] is Card.Values.ACE
    assert value_map["K"] is Card.Values.KING


def test_card_get_parses_uppercase_and_lowercase_text():
    card = Card.get("as")
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"


def test_card_get_rejects_invalid_length():
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("10S")


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value for card!"):
        Card.get("XS")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit for card!"):
        Card.get("AX")


def test_card_properties_color_black_red_and_full_name():
    black_card = Card.get("KS")
    red_card = Card.get("QH")

    assert black_card.color == Card.BLACK
    assert black_card.is_black is True
    assert black_card.is_red is False
    assert black_card.full_name == "king of spades"

    assert red_card.color == Card.RED
    assert red_card.is_black is False
    assert red_card.is_red is True
    assert red_card.full_name == "queen of hearts"


def test_card_unicode_skips_knight_for_queen_and_king():
    queen = Card.get("QD")
    king = Card.get("KC")
    assert queen.unicode == chr(Card.Suits.DIAMONDS.unicode + 12 + 1)
    assert king.unicode == chr(Card.Suits.CLUBS.unicode + 13 + 1)


def test_card_comparisons_and_equality():
    two = Card.get("2S")
    three = Card.get("3H")
    another_two_same = Card.get("2S")
    two_different_suit = Card.get("2C")

    assert two < three
    assert three > two
    assert two <= another_two_same
    assert three >= another_two_same
    assert two == another_two_same
    assert two != two_different_suit
    assert two != "2S"


def test_deck_default_construction_has_52_cards_and_known_order():
    deck = Deck(shuffle=False)

    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KH")
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_contains_only_cards():
    deck = Deck(shuffle=False)

    assert Card.get("AS") in deck
    assert "AS" not in deck
    assert 123 not in deck


def test_deck_draw_updates_counts_and_removes_top_card():
    deck = Deck(shuffle=False)
    top_before = deck.top

    drawn = deck.draw()

    assert drawn == top_before
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert drawn not in deck


def test_deck_draw_multiple_returns_list_and_tracks_draws():
    deck = Deck(shuffle=False)
    drawn = deck.draw(times=3)

    assert isinstance(drawn, list)
    assert len(drawn) == 3
    assert deck.draws == 3
    assert deck.cards_left == 49
    assert drawn[0] == Card.get("KH")
    assert drawn[1] == Card.get("QH")
    assert drawn[2] == Card.get("JH")


def test_deck_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)

    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.draws == 5
    assert deck.cards_left == 47
    assert str(hand) == "[9♠, T♠, J♠, Q♠, K♠]"


def test_deck_shuffle_invokes_random_shuffle_requested_number_of_times(monkeypatch):
    deck = Deck(shuffle=False)
    calls = []

    def fake_shuffle(cards):
        calls.append(list(cards))
        cards.reverse()

    monkeypatch.setattr("gamble.models.cards.random.shuffle", fake_shuffle)

    original_top = deck.top
    deck.shuffle(times=3)

    assert deck.shuffles == 3
    assert len(calls) == 3
    assert deck.top == original_top  # reversed 3 times, ends on original top


def test_euchre_deck_contains_only_24_cards_and_expected_values():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert len(deck.cards) == 24

    values = {card.value.value for card in deck.cards}
    assert values == {1, 9, 10, 11, 12, 13}

    assert Card.get("8S") not in deck
    assert Card.get("AS") in deck
    assert Card.get("9H") in deck
