import random

import pytest

import gamble.models.cards as cards_module
from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand

SEED = 12345


def test_card_default_is_ace_of_spades():
    random.seed(SEED)
    card = Card()
    assert card.value.name == "ace"
    assert card.suit.name == "spades"


def test_card_str_repr_and_full_name():
    random.seed(SEED)
    card = Card.get("AS")
    assert str(card) == "A\u2660"
    assert repr(card) == "<Card:A\u2660>"
    assert card.full_name == "ace of spades"


def test_card_color_flags():
    random.seed(SEED)
    spade = Card.get("KS")
    heart = Card.get("KH")
    assert spade.color == Card.BLACK
    assert spade.is_black is True
    assert spade.is_red is False
    assert heart.color == Card.RED
    assert heart.is_red is True
    assert heart.is_black is False


def test_get_parses_valid_text():
    random.seed(SEED)
    card = Card.get("TH")
    assert card.value.name == "ten"
    assert card.suit.name == "hearts"
    card = Card.get("2D")
    assert card.value.value == 2
    assert card.suit.name == "diamonds"


def test_get_is_case_insensitive():
    random.seed(SEED)
    assert Card.get("qc") == Card.get("QC")
    assert Card.get("qc").full_name == "queen of clubs"


def test_get_rejects_wrong_length():
    random.seed(SEED)
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("10H")
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("A")
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("")


def test_get_rejects_invalid_value_and_suit():
    random.seed(SEED)
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("1S")
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("AX")


def test_unicode_skips_knight_for_face_cards():
    random.seed(SEED)
    # ace of spades: base 0x1F0A0 + value 1 -> U+1F0A1
    assert Card.get("AS").unicode == chr(0x1F0A1)
    # queen of spades: base + value 12 + knight skip 1 -> U+1F0AD
    assert Card.get("QS").unicode == chr(0x1F0AD)


def test_card_equality():
    random.seed(SEED)
    assert Card.get("AS") == Card()
    assert Card.get("AH") != Card()
    assert (Card() == "AS") is False


def test_card_ordering_uses_value():
    random.seed(SEED)
    assert Card.get("2H") < Card.get("3S")
    assert Card.get("KS") > Card.get("QH")
    assert Card.get("2S") <= Card.get("2S")
    assert Card.get("2S") >= Card.get("2S")
    # same value, different suit: neither is strictly less than the other
    assert not (Card.get("2S") < Card.get("2H"))
    assert not (Card.get("2S") > Card.get("2H"))


def test_suits_all_sorted_by_value():
    random.seed(SEED)
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert set(Card.Suits.dict().keys()) == {"S", "C", "D", "H"}


def test_values_all_sorted_and_complete():
    random.seed(SEED)
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert Card.Values.dict()["T"].name == "ten"
    assert Card.Values.dict()["A"].name == "ace"


def test_default_deck_has_52_unique_cards():
    random.seed(SEED)
    deck = Deck()
    assert len(deck.cards) == 52
    assert len({card.full_name for card in deck.cards}) == 52


def test_unshuffled_deck_order():
    random.seed(SEED)
    deck = Deck(shuffle=False)
    # top is the last element; the default build puts ace of spades on top
    assert deck.top.full_name == "ace of spades"
    assert deck.bottom.full_name == "king of hearts"
    assert deck.shuffles == 0


def test_seeded_shuffle_is_reproducible():
    random.seed(SEED)
    first = Deck(shuffle=False)
    random.seed(SEED)
    first.shuffle()

    random.seed(SEED)
    second = Deck(shuffle=False)
    random.seed(SEED)
    second.shuffle()

    assert [c.full_name for c in first.cards] == [c.full_name for c in second.cards]


def test_shuffle_keeps_cards_and_counts_shuffles():
    random.seed(SEED)
    deck = Deck()  # constructor shuffles once
    before = sorted(c.full_name for c in deck.cards)
    deck.shuffle(times=2)
    after = sorted(c.full_name for c in deck.cards)
    assert before == after
    assert deck.shuffles == 3


def test_draw_single_card_updates_state():
    random.seed(SEED)
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert isinstance(card, Card)
    assert card.full_name == "ace of spades"
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_cards_returns_list():
    random.seed(SEED)
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert [c.full_name for c in cards] == [
        "ace of spades",
        "two of spades",
        "three of spades",
    ]
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_from_empty_deck_raises():
    random.seed(SEED)
    deck = Deck(cards=[Card()], shuffle=False)
    deck.draw()
    assert deck.cards_left == 0
    with pytest.raises(IndexError):
        deck.draw()


def test_draw_hand_returns_hand_of_requested_size():
    random.seed(SEED)
    deck = Deck()
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47


def test_draw_hand_size_one_is_wrapped_in_list():
    random.seed(SEED)
    deck = Deck()
    hand = deck.draw_hand(size=1)
    assert isinstance(hand, Hand)
    assert len(hand) == 1
    assert deck.cards_left == 51


def test_contains_checks_cards_only():
    random.seed(SEED)
    deck = Deck(shuffle=False)
    assert deck.top in deck
    assert Card.get("AS") in deck
    assert "AS" not in deck
    deck.draw()
    assert Card.get("AS") not in deck


def test_str_and_repr_show_cards_left():
    random.seed(SEED)
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw(times=2)
    assert str(deck) == "<Deck[50]>"


def test_shuffle_delegates_to_random_shuffle(monkeypatch):
    random.seed(SEED)
    calls = []

    def fake_shuffle(seq):
        calls.append(len(seq))

    monkeypatch.setattr(cards_module.random, "shuffle", fake_shuffle)
    deck = Deck(shuffle=False)
    deck.shuffle(times=3)
    assert calls == [52, 52, 52]
    assert deck.shuffles == 3


def test_euchre_deck_has_24_cards():
    random.seed(SEED)
    deck = EuchreDeck()
    assert len(deck.cards) == 24
    assert {c.value.char for c in deck.cards} == {"9", "T", "J", "Q", "K", "A"}
    assert "2" not in {c.value.char for c in deck.cards}
