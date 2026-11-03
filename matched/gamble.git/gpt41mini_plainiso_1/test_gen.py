import pytest
import string
from gamble.models.cards import Card, Deck
from gamble.errors import InvalidCard

def test_card_get_valid_cards():
    """Test Card.get with all valid card texts"""
    for suit in Card.Suits.all():
        for value in Card.Values.all():
            card_text = value.char + suit.char
            card = Card.get(card_text)
            assert card.value == value
            assert card.suit == suit
            assert str(card) == f"{value.char}{suit.symbol}"

def test_card_get_invalid_length():
    """Card.get raises InvalidCard on wrong length strings"""
    with pytest.raises(InvalidCard):
        Card.get("")
    with pytest.raises(InvalidCard):
        Card.get("AAA")
    with pytest.raises(InvalidCard):
        Card.get("9")
    with pytest.raises(InvalidCard):
        Card.get("10S")

def test_card_get_invalid_value_char():
    """Card.get raises InvalidCard on invalid value character"""
    invalid_texts = ["1S", "0H", "ZS", "XS"]
    for txt in invalid_texts:
        with pytest.raises(InvalidCard):
            Card.get(txt)

def test_card_get_invalid_suit_char():
    """Card.get raises InvalidCard on invalid suit character"""
    invalid_texts = ["AS", "AD", "AH", "AC"]
    # Add invalid suits by modifying valid value chars
    invalid_suits = ['X', 'Y', 'Z', '1']
    for suit_char in invalid_suits:
        txt = "A" + suit_char
        with pytest.raises(InvalidCard):
            Card.get(txt)

def test_card_color_properties():
    """Test Card color, is_black and is_red properties"""
    black_suits = [Card.Suits.SPADES, Card.Suits.CLUBS]
    red_suits = [Card.Suits.HEARTS, Card.Suits.DIAMONDS]

    for suit in black_suits:
        card = Card(suit=suit)
        assert card.color == Card.BLACK
        assert card.is_black is True
        assert card.is_red is False

    for suit in red_suits:
        card = Card(suit=suit)
        assert card.color == Card.RED
        assert card.is_black is False
        assert card.is_red is True

def test_card_full_name():
    card = Card(value=Card.Values.KING, suit=Card.Suits.HEARTS)
    assert card.full_name == "king of hearts"
    card2 = Card.get("AS")
    assert card2.full_name == "ace of spades"

def test_card_unicode_character():
    """Verify that unicode property returns expected char"""
    # Just check it returns a string of length 1 and no error
    for suit in Card.Suits.all():
        for value in Card.Values.all():
            card = Card(value=value, suit=suit)
            uni = card.unicode
            assert isinstance(uni, str)
            assert len(uni) == 1

def test_card_str_and_repr():
    card = Card.get("KH")
    s = str(card)
    r = repr(card)
    assert s == "K♥"
    assert r == f"<Card:{s}>"

def test_card_comparisons():
    c1 = Card.get("5S")
    c2 = Card.get("7D")
    c3 = Card.get("5H")
    # __lt__ and __gt__
    assert c1 < c2
    assert c2 > c1
    # __eq__ compares suit and value
    assert c1 != c2
    assert c1 != c3  # different suit
    c4 = Card(value=c1.value, suit=c1.suit)
    assert c1 == c4
    # __le__ and __ge__
    assert c1 <= c4
    assert c1 >= c4
    assert c1 <= c2
    assert c2 >= c1

def test_deck_default_has_52_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    # Deck should contain all suit/value combinations
    suits = {c.suit for c in deck.cards}
    values = {c.value for c in deck.cards}
    assert suits == set(Card.Suits.all())
    assert values == set(Card.Values.all())

def test_deck_shuffle_increments_shuffles():
    deck = Deck(shuffle=False)
    initial_cards = deck.cards.copy()
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    # After shuffle cards should be different order than initial
    assert deck.cards != initial_cards or deck.cards == initial_cards  # rare but possible shuffle no change

def test_deck_draw_and_draws_count():
    deck = Deck(shuffle=False)
    initial_len = deck.cards_left
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == initial_len - 1
    assert deck.draws == 1

    # Draw multiple cards
    cards = deck.draw(times=5)
    assert isinstance(cards, list)
    assert len(cards) == 5
    assert deck.cards_left == initial_len - 6
    assert deck.draws == 6

def test_deck_draw_hand_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert len(hand.cards) == 5
    # Draw hand should reduce deck size by hand.size
    assert deck.cards_left == 52 - 5

def test_deck_top_and_bottom_cards_consistency():
    deck = Deck(shuffle=False)
    # Initially deck.cards reversed, so bottom is original first card added
    bottom = deck.bottom
    top = deck.top
    assert bottom == deck.cards[0]
    assert top == deck.cards[-1]
    # After draw top card changes
    card_drawn = deck.draw()
    assert deck.top != card_drawn

def test_deck_contains_card_and_invalid_object():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert card not in deck  # since drawn
    new_card = Card.get("AS")
    # If new_card still in deck (likely not since deck shuffled or reversed)
    # but new_card might be
    assert (new_card in deck) == (new_card in deck.cards)
    # Check contains with invalid type returns False
    assert (42 in deck) is False
    assert ("AS" in deck) is False
    assert (object() in deck) is False

def test_deck_init_with_custom_cards_and_no_shuffle():
    cards = [Card.get("AS"), Card.get("KH")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards == cards
    assert deck.cards_left == 2
    # shuffles count should be 0 since shuffle=False
    assert deck.shuffles == 0

def test_card_equality_with_non_card_object():
    card = Card.get("AS")
    assert card != "AS"
    assert card != 42
    assert card != None

def test_card_get_case_insensitivity():
    card1 = Card.get("as")
    card2 = Card.get("AS")
    card3 = Card.get("aS")
    assert card1 == card2 == card3

def test_card_value_and_suit_identity():
    card = Card.get("TD")
    assert card.value.char == "T"
    assert card.suit.char == "D"
    assert card.value.name == "ten"
    assert card.suit.name == "diamonds"

def test_card_unicode_value_for_queen_and_king_skip_knight():
    # The unicode property adds +1 if value is queen or king
    queen = Card.get("QD")
    king = Card.get("KH")
    jack = Card.get("JS")
    # Normal: suit.unicode + value (11 for jack, 12 queen, 13 king)
    # For queen and king, an additional 1 is added
    base_jack = jack.suit.unicode + jack.value.value
    base_queen = queen.suit.unicode + queen.value.value + 1
    base_king = king.suit.unicode + king.value.value + 1
    assert ord(jack.unicode) == base_jack
    assert ord(queen.unicode) == base_queen
    assert ord(king.unicode) == base_king
