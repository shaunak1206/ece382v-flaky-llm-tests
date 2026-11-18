import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_suits_all_sorted_and_dict_keys():
    suits = Card.Suits.all()
    assert [s.value for s in suits] == [0, 1, 2, 3]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]

    suit_dict = Card.Suits.dict()
    assert set(suit_dict.keys()) == {"S", "C", "D", "H"}
    assert suit_dict["S"] is Card.Suits.SPADES
    assert suit_dict["H"] is Card.Suits.HEARTS


def test_values_all_sorted_and_dict_keys():
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0].char == "A"
    assert values[-1].char == "K"

    value_dict = Card.Values.dict()
    assert set(value_dict.keys()) == {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}
    assert value_dict["A"] is Card.Values.ACE
    assert value_dict["K"] is Card.Values.KING


def test_card_default_constructor_uses_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"
    assert card.full_name == "ace of spades"


def test_card_get_valid_parses_case_insensitive():
    card = Card.get("qh")
    assert card.value is Card.Values.QUEEN
    assert card.suit is Card.Suits.HEARTS
    assert str(card) == "Q♥"
    assert card.full_name == "queen of hearts"


@pytest.mark.parametrize(
    "text, message",
    [
        ("A", "Too many characters for a card!"),
        ("ASX", "Too many characters for a card!"),
    ],
)
def test_card_get_invalid_length_raises_invalid_card(text, message):
    with pytest.raises(InvalidCard, match=message):
        Card.get(text)


@pytest.mark.parametrize(
    "text, message",
    [
        ("1S", "Invalid value for card!"),
        ("AS", None),  # sanity check not used directly
        ("AZ", "Invalid suit for card!"),
    ],
)
def test_card_get_invalid_value_or_suit_raises_invalid_card(text, message):
    if message is None:
        card = Card.get(text)
        assert card.value is Card.Values.ACE
        return

    with pytest.raises(InvalidCard, match=message):
        Card.get(text)


def test_card_color_and_boolean_properties():
    spade = Card.get("AS")
    heart = Card.get("AH")
    diamond = Card.get("AD")

    assert spade.color == Card.BLACK
    assert spade.is_black is True
    assert spade.is_red is False

    assert heart.color == Card.RED
    assert heart.is_red is True
    assert heart.is_black is False

    assert diamond.color == Card.RED


def test_card_unicode_skips_knight_for_queen_and_king():
    queen_spades = Card.get("QS")
    king_spades = Card.get("KS")
    jack_spades = Card.get("JS")

    assert queen_spades.unicode == chr(Card.Suits.SPADES.unicode + 12 + 1)
    assert king_spades.unicode == chr(Card.Suits.SPADES.unicode + 13 + 1)
    assert jack_spades.unicode == chr(Card.Suits.SPADES.unicode + 11)


def test_card_comparisons_and_equality():
    two_spades = Card.get("2S")
    three_hearts = Card.get("3H")
    another_two_spades = Card.get("2S")

    assert two_spades < three_hearts
    assert three_hearts > two_spades
    assert two_spades <= another_two_spades
    assert two_spades >= another_two_spades
    assert two_spades == another_two_spades
    assert two_spades != three_hearts
    assert not (two_spades == object())


def test_deck_default_constructor_builds_52_card_deck_without_shuffle(monkeypatch):
    calls = []

    def fake_shuffle(cards):
        calls.append(len(cards))

    monkeypatch.setattr("gamble.models.cards.random.shuffle", fake_shuffle)

    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert calls == []

    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KH")
    assert Card.get("AS") in deck
    assert "AS" not in deck


def test_deck_shuffle_invokes_random_shuffle_and_updates_counter(monkeypatch):
    calls = []

    def fake_shuffle(cards):
        calls.append(tuple(str(c) for c in cards))

    monkeypatch.setattr("gamble.models.cards.random.shuffle", fake_shuffle)

    deck = Deck(shuffle=False)
    deck.shuffle(times=3)

    assert deck.shuffles == 3
    assert len(calls) == 3
    assert deck.cards_left == 52


def test_deck_draw_single_and_multiple_update_state():
    deck = Deck(cards=[Card.get("AS"), Card.get("2S"), Card.get("3S")], shuffle=False)

    one = deck.draw()
    assert isinstance(one, Card)
    assert one == Card.get("3S")
    assert deck.draws == 1
    assert deck.cards_left == 2

    two = deck.draw(times=2)
    assert [str(c) for c in two] == ["2♠", "A♠"]
    assert deck.draws == 3
    assert deck.cards_left == 0


def test_deck_draw_hand_returns_hand_and_consumes_cards():
    deck = Deck(cards=[Card.get("AS"), Card.get("2S"), Card.get("3S"), Card.get("4S")], shuffle=False)

    hand = deck.draw_hand(size=3)
    assert isinstance(hand, Hand)
    assert len(hand) == 3
    assert [str(c) for c in hand.cards] == ["2♠", "3♠", "4♠"]
    assert deck.cards_left == 1
    assert deck.draws == 3


def test_deck_str_and_repr_show_cards_left():
    deck = Deck(cards=[Card.get("AS"), Card.get("2S")], shuffle=False)
    assert str(deck) == "<Deck[2]>"
    assert repr(deck) == "<Deck[2]>"


def test_euchre_deck_contains_expected_cards_and_count():
    deck = EuchreDeck()
    assert deck.cards_left == 24

    expected_values = {"9", "T", "J", "Q", "K", "A"}
    assert all(card.value.char in expected_values for card in deck.cards)
    assert all(card.suit in Card.Suits.all() for card in deck.cards)

    assert Card.get("9S") in deck
    assert Card.get("8S") not in deck
