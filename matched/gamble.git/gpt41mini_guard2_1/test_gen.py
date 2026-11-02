import random
import pytest

from gamble.models.cards import Card, Deck
from gamble.errors import InvalidCard


def test_card_suits_all_sorted_and_unique():
    suits = Card.Suits.all()
    values = [s.value for s in suits]
    # suits should be sorted by their value
    assert values == sorted(values)
    # suits should be unique
    assert len(suits) == len(set(values))
    # all suits are instances of Card.Suit
    assert all(hasattr(s, "name") and hasattr(s, "char") and hasattr(s, "symbol") for s in suits)


def test_card_values_all_sorted_and_unique():
    values = Card.Values.all()
    vals = [v.value for v in values]
    # values should be sorted by their value
    assert vals == sorted(vals)
    # values unique
    assert len(vals) == len(set(vals))
    # all values are instances of Card.Value
    assert all(hasattr(v, "name") and hasattr(v, "char") for v in values)


@pytest.mark.parametrize(
    "text,value_char,suit_char",
    [
        ("AS", "A", "S"),
        ("Td", "T", "D"),
        ("7h", "7", "H"),
        ("cJ", "J", "C"),
    ],
)
def test_card_get_valid_text_representations(text, value_char, suit_char):
    card = Card.get(text)
    assert card.value.char == value_char.upper()
    assert card.suit.char == suit_char.upper()
    # str(card) returns the expected format: value char + suit symbol
    assert str(card).startswith(value_char.upper())
    assert str(card)[-1] == card.suit.symbol


@pytest.mark.parametrize(
    "text, exc_match",
    [
        ("", "Too many characters"),
        ("AAA", "Too many characters"),
        ("1S", "Invalid value for card"),
        ("ACX", "Too many characters"),
        ("ASX", "Too many characters"),
        ("AZ", "Invalid suit for card"),
    ],
)
def test_card_get_raises_invalid_card(text, exc_match):
    with pytest.raises(InvalidCard) as excinfo:
        Card.get(text)
    assert exc_match in str(excinfo.value)


def test_card_properties_and_unicode_consistency():
    # test color properties and unicode char
    spades = Card.Suits.SPADES
    diamonds = Card.Suits.DIAMONDS
    ace_spades = Card(value=Card.Values.ACE, suit=spades)
    ten_diamonds = Card(value=Card.Values.TEN, suit=diamonds)

    assert ace_spades.color == Card.BLACK
    assert ace_spades.is_black is True
    assert ace_spades.is_red is False

    assert ten_diamonds.color == Card.RED
    assert ten_diamonds.is_red is True
    assert ten_diamonds.is_black is False

    # full_name combines value.name and suit.name
    assert ace_spades.full_name == "ace of spades"
    assert ten_diamonds.full_name == "ten of diamonds"

    # unicode char should be a single character string
    for value in Card.Values.all():
        for suit in Card.Suits.all():
            card = Card(value=value, suit=suit)
            uni_chr = card.unicode
            assert isinstance(uni_chr, str)
            assert len(uni_chr) == 1


def test_card_str_and_repr_consistency():
    card = Card.get("KS")
    s = str(card)
    r = repr(card)
    # repr includes str(card)
    assert s in r
    # str(card) is value char + suit symbol
    assert s[0] == card.value.char
    assert s[1] == card.suit.symbol


def test_card_comparison_operators():
    ace = Card(value=Card.Values.ACE, suit=Card.Suits.CLUBS)
    king = Card(value=Card.Values.KING, suit=Card.Suits.CLUBS)
    ace2 = Card(value=Card.Values.ACE, suit=Card.Suits.DIAMONDS)
    # equality true when same value and suit
    assert ace == Card(value=ace.value, suit=ace.suit)
    # equality false when different suit or value
    assert ace != king
    assert ace != ace2
    # __lt__ and __gt__ by value only
    assert ace < king
    assert king > ace
    # __le__ and __ge__, including equality
    assert ace <= ace2  # same value, different suit; __eq__ false, but ace.value == ace2.value
    assert ace >= ace2
    assert not (king <= ace)
    assert king >= ace


def test_deck_default_construction_and_contains():
    deck = Deck(shuffle=False)
    # It should have 52 cards
    assert deck.cards_left == 52
    # Cards should be in reverse expected order (because cards.reverse() in constructor)
    first_card = deck.bottom
    last_card = deck.top
    # bottom card should be smallest value suit 0 (spades) value 1 (ace)
    assert first_card.suit.value == 0
    assert first_card.value.value == 1
    # top card should be highest value suit 3 (hearts) value 13 (king)
    assert last_card.suit.value == 3
    assert last_card.value.value == 13
    # deck contains a card we know is in deck
    sample_card = Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    assert sample_card in deck
    # deck does not contain an object that is not a Card
    assert "not a card" not in deck


def test_deck_shuffle_and_draw(monkeypatch):
    # patch random.shuffle so it just reverses the list deterministically
    def fake_shuffle(cards_list):
        cards_list.reverse()

    monkeypatch.setattr("random.shuffle", fake_shuffle)
    deck = Deck(shuffle=True)
    old_cards = deck.cards.copy()
    # After shuffle, cards list reversed (because fake_shuffle)
    deck.shuffle()
    # cards should be reversed compared to shuffled state (so probably back to original order)
    # Since Deck constructor already called shuffle once, we test that shuffle changes cards
    # The first shuffle was a reversed list, now shuffled again will reverse it back
    assert deck.cards == old_cards
    old_cards_left = deck.cards_left
    # Draw one card returns a Card
    card_drawn = deck.draw()
    assert isinstance(card_drawn, Card)
    assert deck.cards_left == old_cards_left - 1
    # Draw multiple cards returns list of Cards
    cards_drawn = deck.draw(times=3)
    assert isinstance(cards_drawn, list)
    assert len(cards_drawn) == 3
    assert all(isinstance(c, Card) for c in cards_drawn)
    assert deck.cards_left == old_cards_left - 4


def test_deck_draw_hand_produces_correct_hand_and_length():
    random.seed(42)  # seed to make shuffle reproducible if any
    deck = Deck(shuffle=True)
    hand = deck.draw_hand(size=5)
    assert len(hand) == 5
    # Hand cards should come from deck, so cards_left decreases by 5
    assert deck.cards_left == 52 - 5
    # The str of hand includes all card strings
    for card in hand.cards:
        assert str(card) in str(hand)


def test_deck_str_and_repr_format():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert str(deck.cards_left) in s
    assert s == r
    assert s.startswith("<Deck[") and s.endswith(">")


def test_card_unicode_hack_effect_on_queen_and_king():
    # Check that for cards Queen and King, unicode is incremented by 1 due to hack
    for suit in Card.Suits.all():
        queen = Card(value=Card.Values.QUEEN, suit=suit)
        king = Card(value=Card.Values.KING, suit=suit)
        queen_code = suit.unicode + queen.value.value + 1  # hack active
        king_code = suit.unicode + king.value.value + 1
        assert ord(queen.unicode) == queen_code
        assert ord(king.unicode) == king_code
    # For cards below queen, hack not active
    for suit in Card.Suits.all():
        jack = Card(value=Card.Values.JACK, suit=suit)
        jack_code = suit.unicode + jack.value.value
        assert ord(jack.unicode) == jack_code


def test_card_get_method_accepts_lowercase_and_uppercase():
    # lowercase inputs convert correctly to uppercase chars internally
    card_lower = Card.get("ah")
    card_upper = Card.get("AH")
    assert card_lower == card_upper
    assert card_lower.value == Card.Values.ACE
    assert card_lower.suit == Card.Suits.HEARTS


def test_card_eq_returns_false_for_non_card_objects():
    card = Card.get("AS")
    assert (card == "AS") is False
    assert (card == 42) is False
    assert (card == None) is False


def test_card_comparisons_only_consider_value_not_suit_order():
    # Two cards with same value but different suits considered equal in value comparison
    c1 = Card(value=Card.Values.TEN, suit=Card.Suits.HEARTS)
    c2 = Card(value=Card.Values.TEN, suit=Card.Suits.SPADES)
    assert not (c1 < c2)
    assert not (c1 > c2)
    # Equality depends on suit too
    assert c1 != c2
    # For different values comparison works
    c3 = Card(value=Card.Values.NINE, suit=Card.Suits.HEARTS)
    assert c3 < c1
    assert c1 > c3


def test_deck_draw_raises_index_error_if_empty():
    deck = Deck(cards=[], shuffle=False)
    with pytest.raises(IndexError):
        deck.draw()


def test_deck_draw_multiple_more_than_cards():
    deck = Deck(cards=[Card.get("AS"), Card.get("2S")], shuffle=False)
    # Try to draw 3 cards from 2 card deck: should raise IndexError
    with pytest.raises(IndexError):
        deck.draw(times=3)


def test_deck_shuffles_counter_increments_correctly(monkeypatch):
    monkeypatch.setattr("random.shuffle", lambda x: None)
    deck = Deck(shuffle=False)
    assert deck.shuffles == 0
    deck.shuffle(times=3)
    assert deck.shuffles == 3


def test_deck_draws_counter_increments_correctly():
    deck = Deck(shuffle=False)
    initial_draws = deck.draws
    deck.draw(times=1)
    assert deck.draws == initial_draws + 1
    deck.draw(times=5)
    assert deck.draws == initial_draws + 6


def test_deck_custom_cards_initialization():
    cards = [Card.get("AS"), Card.get("KD")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 2
    for c in cards:
        assert c in deck
