import pytest
import random
from gamble.models.cards import Card, Deck, Hand, EuchreDeck
from gamble.errors import InvalidCard


@pytest.mark.parametrize("text,value_char,suit_char,value_name,suit_name", [
    ("AS", "A", "S", "ace", "spades"),
    ("2C", "2", "C", "two", "clubs"),
    ("TD", "T", "D", "ten", "diamonds"),
    ("KH", "K", "H", "king", "hearts"),
])
def test_card_get_valid(text, value_char, suit_char, value_name, suit_name):
    card = Card.get(text)
    assert card.value.char == value_char
    assert card.suit.char == suit_char
    assert card.value.name == value_name
    assert card.suit.name == suit_name


@pytest.mark.parametrize("text", [
    "", "ABC", "1", "ZK", "A1", "AB", "QQQ"
])
def test_card_get_invalid_text_raises(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_color_properties():
    spade_card = Card(Card.Values.ACE, Card.Suits.SPADES)
    heart_card = Card(Card.Values.ACE, Card.Suits.HEARTS)
    assert spade_card.color == Card.BLACK
    assert spade_card.is_black is True
    assert spade_card.is_red is False
    assert heart_card.color == Card.RED
    assert heart_card.is_red is True
    assert heart_card.is_black is False


@pytest.mark.parametrize("value,suit,expected_unicode", [
    (Card.Values.ACE, Card.Suits.SPADES, chr(127136 + 1)),  # Ace spades
    (Card.Values.QUEEN, Card.Suits.CLUBS, chr(127184 + 12 + 1)),  # Queen clubs, skip knight (+1)
    (Card.Values.KING, Card.Suits.DIAMONDS, chr(127168 + 13 + 1)),  # King diamonds
    (Card.Values.TEN, Card.Suits.HEARTS, chr(127152 + 10)),  # Ten hearts no hack
])
def test_card_unicode(value, suit, expected_unicode):
    card = Card(value=value, suit=suit)
    assert card.unicode == expected_unicode


def test_card_str_and_repr():
    card = Card(Card.Values.TWO, Card.Suits.DIAMONDS)
    s = str(card)
    r = repr(card)
    assert s == "2♦"
    assert r == f"<Card:{s}>"


def test_card_comparisons():
    card_two = Card(Card.Values.TWO, Card.Suits.SPADES)
    card_three = Card(Card.Values.THREE, Card.Suits.HEARTS)
    card_two_copy = Card(Card.Values.TWO, Card.Suits.SPADES)
    assert card_two < card_three
    assert card_three > card_two
    assert card_two <= card_two_copy
    assert card_two >= card_two_copy
    assert card_two == card_two_copy
    assert (card_two == "not a card") is False


def test_deck_default_construction_and_properties():
    deck = Deck(shuffle=False)
    # Deck should have 52 cards initially
    assert deck.cards_left == 52
    # Bottom card should be Ace of Spades (smallest value, first suit reversed)
    bottom = deck.bottom
    assert bottom.value == Card.Values.ACE
    assert bottom.suit == Card.Suits.SPADES
    # Top card should be King of Hearts (largest value, last suit reversed)
    top = deck.top
    assert top.value == Card.Values.KING
    assert top.suit == Card.Suits.HEARTS
    # shuffles should be zero since shuffle=False
    assert deck.shuffles == 0
    # draws should be zero initially
    assert deck.draws == 0


def test_deck_contains_method():
    deck = Deck(shuffle=False)
    card = Card(Card.Values.ACE, Card.Suits.SPADES)
    # card in deck
    assert card in deck
    # non-Card object returns False
    assert 42 not in deck


def test_deck_draw_single_card_updates_counts_correctly():
    deck = Deck(shuffle=False)
    initial_cards_left = deck.cards_left
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == initial_cards_left - 1
    assert deck.draws == 1
    # card is no longer in deck
    assert card not in deck


def test_deck_draw_multiple_cards_returns_list_and_updates_counts():
    deck = Deck(shuffle=False)
    initial_cards_left = deck.cards_left
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert deck.cards_left == initial_cards_left - 3
    assert deck.draws == 3
    for card in cards:
        assert card not in deck


def test_deck_draw_hand_returns_hand_and_reduces_cards():
    deck = Deck(shuffle=False)
    initial_cards_left = deck.cards_left
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == initial_cards_left - 5


def test_deck_shuffle_multiple_times_increases_counter_and_shuffles_cards(monkeypatch):
    deck = Deck(shuffle=False)
    # Save the original list for comparison
    original_cards = deck.cards[:]

    # Patch random.shuffle to a no-op so deck remains in same order
    def fake_shuffle(lst):
        # We'll fake the shuffle to leave list identical
        pass

    monkeypatch.setattr("random.shuffle", fake_shuffle)
    deck.shuffle(times=4)

    # The cards order is unchanged due to our fake shuffle
    assert deck.cards == original_cards
    # The shuffles counter incremented properly
    assert deck.shuffles == 4


def test_hand_properties_and_rank_identification():
    # Straight flush: 6S,7S,8S,9S,TS
    hand = Hand.get("6S,7S,8S,9S,TS")
    assert hand.is_flush
    assert hand.is_straight
    assert hand.is_straight_flush
    assert hand.rank.name == "straight flush"
    # Four of a kind: 9S,9H,9D,9C,2S
    hand = Hand.get("9S,9H,9D,9C,2S")
    assert hand.is_four_of_a_kind
    assert hand.rank.name == "four of a kind"
    # Full house: TS,TD,TH,2S,2D
    hand = Hand.get("TS,TD,TH,2S,2D")
    assert hand.is_full_house
    assert hand.rank.name == "full house"
    # Flush: 2H,5H,9H,QH,KH
    hand = Hand.get("2H,5H,9H,QH,KH")
    assert hand.is_flush
    assert not hand.is_straight
    assert not hand.is_full_house
    assert hand.rank.name == "flush"
    # Straight: 3C,4D,5H,6S,7S
    hand = Hand.get("3C,4D,5H,6S,7S")
    assert hand.is_straight
    assert not hand.is_flush
    assert hand.rank.name == "straight"
    # Three of a kind: 4H,4D,4S,8C,9D
    hand = Hand.get("4H,4D,4S,8C,9D")
    assert hand.is_three_of_a_kind
    assert hand.rank.name == "three of a kind"
    # Two pair: AH,AD,7S,7D,3C
    hand = Hand.get("AH,AD,7S,7D,3C")
    assert hand.is_two_pair
    assert hand.rank.name == "two pair"
    # One pair: 5H,5D,3S,8C,9D
    hand = Hand.get("5H,5D,3S,8C,9D")
    assert hand.is_one_pair
    assert hand.rank.name == "pair"
    # High Card: 2H,5D,8S,JD,KC
    hand = Hand.get("2H,5D,8S,JD,KC")
    assert hand.rank.name == "high card"


def test_hand_comparison_operators():
    # Hand with pair (5s) vs hand with three of a kind (4s)
    pair_hand = Hand.get("5H,5D,3S,8C,9D")  # pair
    three_kind_hand = Hand.get("4H,4D,4S,8C,9D")  # three of a kind
    assert pair_hand < three_kind_hand
    assert three_kind_hand > pair_hand
    # Same rank comparison returns False for < and >
    pair_hand2 = Hand.get("6H,6D,3S,8C,9D")
    assert (pair_hand < pair_hand2) is False
    assert (pair_hand > pair_hand2) is False


def test_hand_len_and_str_and_repr():
    hand = Hand.get("2H,3D,4S,5C,6D")
    assert len(hand) == 5
    s = str(hand)
    r = repr(hand)
    for card_str in ["2♥", "3♦", "4♠", "5♣", "6♦"]:
        # Could be in sorted order; suit symbol must be present
        assert any(c.value.char in s and c.suit.symbol in s for c in hand.cards)
    assert r.startswith("<Hand[5]")


def test_euchre_deck_has_24_cards_and_correct_values():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    all_values = set(card.value.value for card in deck.cards)
    # Euchre deck only values: 9,10(=T),11=J,12=Q,13=K,1=A
    assert all(v in {1, 9, 10, 11, 12, 13} for v in all_values)
    suits = set(card.suit for card in deck.cards)
    assert suits == set(Card.Suits.all())


def test_deck_shuffle_changes_order(monkeypatch):
    deck1 = Deck(shuffle=False)
    deck2 = Deck(shuffle=False)
    # Seed RNG to make shuffle deterministic
    random.seed(12345)
    deck1.shuffle()
    random.seed(12345)
    deck2.shuffle()
    # Because seed is same, order should be same
    assert [str(card) for card in deck1.cards] == [str(card) for card in deck2.cards]
    # But different from original order
    deck3 = Deck(shuffle=False)
    assert [str(card) for card in deck1.cards] != [str(card) for card in deck3.cards]


def test_card_full_name_property():
    c = Card(Card.Values.KING, Card.Suits.DIAMONDS)
    assert c.full_name == "king of diamonds"


def test_card_get_invalid_value_or_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("1S")  # '1' is invalid value char
    with pytest.raises(InvalidCard):
        Card.get("ASX")  # too many characters
    with pytest.raises(InvalidCard):
        Card.get("AZ")  # 'Z' invalid suit char


def test_comparison_with_non_card_returns_false():
    card = Card()
    assert (card == object()) is False
    assert (card == "AS") is False


def test_hand_get_parsing_spaces_and_case():
    hand = Hand.get("   as, 2h, 3C , 4d, 5s  ")
    assert len(hand) == 5
    cards_str = [str(c).upper() for c in hand.cards]
    expected = {"AS", "2H", "3C", "4D", "5S"}
    assert set(cards_str) == expected
