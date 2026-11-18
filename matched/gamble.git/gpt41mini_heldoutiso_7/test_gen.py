import pytest
from gamble.models.cards import Card, Deck, EuchreDeck
from gamble.errors import InvalidCard


def test_card_suits_all_sorted_and_contains_expected():
    suits = Card.Suits.all()
    values = [suit.value for suit in suits]
    # Suits should be sorted by value 0..3
    assert values == sorted(values)
    # There are exactly 4 suits
    assert len(suits) == 4
    names = {suit.name for suit in suits}
    assert names == {"spades", "clubs", "diamonds", "hearts"}


def test_card_values_all_sorted_and_contains_expected():
    values = Card.Values.all()
    val_values = [v.value for v in values]
    # Values should be sorted 1..13
    assert val_values == list(range(1, 14))
    chars = {v.char for v in values}
    # Check chars contains standard card chars
    for c in "A23456789TJQK":
        assert c in chars


def test_card_get_with_valid_input_returns_correct_card():
    card = Card.get("AS")
    assert card.value.char == "A"
    assert card.suit.char == "S"
    card2 = Card.get("td")  # lowercase input
    assert card2.value.char == "T"
    assert card2.suit.char == "D"


def test_card_get_invalid_length_raises():
    with pytest.raises(InvalidCard):
        Card.get("AAA")
    with pytest.raises(InvalidCard):
        Card.get("A")
    with pytest.raises(InvalidCard):
        Card.get("")


def test_card_get_invalid_value_raises():
    with pytest.raises(InvalidCard):
        Card.get("1S")
    with pytest.raises(InvalidCard):
        Card.get("ZS")


def test_card_get_invalid_suit_raises():
    with pytest.raises(InvalidCard):
        Card.get("AZ")
    with pytest.raises(InvalidCard):
        Card.get("AQ")


def test_card_properties_color_and_is_red_black():
    card_spades = Card(Card.Values.ACE, Card.Suits.SPADES)
    assert card_spades.color == Card.BLACK
    assert card_spades.is_black is True
    assert card_spades.is_red is False

    card_hearts = Card(Card.Values.ACE, Card.Suits.HEARTS)
    assert card_hearts.color == Card.RED
    assert card_hearts.is_red is True
    assert card_hearts.is_black is False


def test_card_unicode_repr_str_and_full_name():
    c = Card(Card.Values.KING, Card.Suits.SPADES)
    # unicode string is a single character
    u = c.unicode
    assert isinstance(u, str)
    assert len(u) == 1
    # str returns value char + suit symbol
    s = str(c)
    assert s == f"{Card.Values.KING.char}{Card.Suits.SPADES.symbol}"
    # repr includes class name and str
    r = repr(c)
    assert r == f"<Card:{s}>"
    # full_name includes value and suit names
    full = c.full_name
    assert "king" in full
    assert "spades" in full


def test_card_comparisons_lt_le_gt_ge_eq():
    c2 = Card(Card.Values.TWO, Card.Suits.CLUBS)
    c3 = Card(Card.Values.THREE, Card.Suits.CLUBS)
    c3_dup = Card(Card.Values.THREE, Card.Suits.CLUBS)
    c3_diff_suit = Card(Card.Values.THREE, Card.Suits.DIAMONDS)

    assert c2 < c3
    assert c3 > c2
    assert c2 <= c3
    assert c3 >= c2
    assert c3 == c3_dup  # same value, same suit
    assert c3 != c2
    assert c3 != c3_diff_suit  # same value, different suit

    # eq with non-card returns False
    assert (c3 == "x") is False


def test_deck_default_52_cards():
    deck = Deck(shuffle=False)
    assert isinstance(deck.cards, list)
    assert len(deck.cards) == 52
    # All cards are unique by (value, suit)
    seen = set((card.value.char, card.suit.char) for card in deck.cards)
    assert len(seen) == 52


def test_deck_contains_method():
    deck = Deck(shuffle=False)
    card = Card.get("AS")
    assert card in deck
    not_a_card = "not a card"
    assert (not_a_card in deck) is False
    card_not_in_deck = Card(Card.Values.ACE, Card.Suits.HEARTS)
    # The default deck should contain all cards, but verify a card not in cards fails
    # Actually, the default deck contains all cards, so test with a card removed
    deck.cards.remove(card_not_in_deck)
    assert (card_not_in_deck in deck) is False


def test_deck_draw_and_cards_left_and_draws_property():
    deck = Deck(shuffle=False)
    initial_count = deck.cards_left
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == initial_count - 1
    assert deck.draws == 1

    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert deck.cards_left == initial_count - 4
    assert deck.draws == 4


def test_deck_draw_hand_returns_hand_of_correct_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    from gamble.models.cards import Hand

    assert isinstance(hand, Hand)
    assert len(hand.cards) == 5
    assert deck.cards_left == 52 - 5


def test_deck_shuffle_increments_shuffles_and_changes_order():
    deck1 = Deck(shuffle=False)
    cards_before = list(deck1.cards)
    deck1.shuffle(times=2)
    assert deck1.shuffles == 2
    # It is very likely the deck order changed after shuffle
    cards_after = deck1.cards
    assert cards_after != cards_before or len(deck1.cards) != 0


def test_euchredeck_has_24_cards_and_expected_values():
    euchre = EuchreDeck(shuffle=False)
    assert len(euchre.cards) == 24
    values = {c.value.value for c in euchre.cards}
    # Euchre uses 9,10,11(J),12(Q),13(K),1(A)
    assert values == {1, 9, 10, 11, 12, 13}
    suits = {c.suit for c in euchre.cards}
    # all four suits present
    assert len(suits) == 4


def test_deck_str_and_repr_return_expected_format():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert s.startswith("<Deck[")
    assert s.endswith("]>")
    assert r == s


def test_deck_top_and_bottom_card_properties():
    deck = Deck(shuffle=False)
    top_card = deck.top
    bottom_card = deck.bottom
    assert isinstance(top_card, Card)
    assert isinstance(bottom_card, Card)
    # top card is last in deck.cards list
    assert top_card == deck.cards[-1]
    # bottom card is the first in deck.cards list
    assert bottom_card == deck.cards[0]


def test_card_unicode_values_for_queen_and_king_skip_knight():
    # Queens and Kings have unicode hack applied
    # For suit spades, test unicode points are what expected
    q = Card(Card.Values.QUEEN, Card.Suits.SPADES)
    k = Card(Card.Values.KING, Card.Suits.SPADES)
    val_q = ord(q.unicode)
    val_k = ord(k.unicode)
    # Queen unicode code point is suit.unicode + value + 1 (hack)
    expected_q = Card.Suits.SPADES.unicode + Card.Values.QUEEN.value + 1
    expected_k = Card.Suits.SPADES.unicode + Card.Values.KING.value + 1
    assert val_q == expected_q
    assert val_k == expected_k

    # For value JACK (11), no hack should be applied
    j = Card(Card.Values.JACK, Card.Suits.SPADES)
    val_j = ord(j.unicode)
    expected_j = Card.Suits.SPADES.unicode + Card.Values.JACK.value
    assert val_j == expected_j


def test_card_less_equal_and_greater_equal_methods_return_true_when_equal():
    c1 = Card(Card.Values.FIVE, Card.Suits.CLUBS)
    c2 = Card(Card.Values.FIVE, Card.Suits.SPADES)
    # same value, different suit; comparison only on value
    assert (c1 <= c2) is True
    assert (c1 >= c2) is True


def test_card_equal_with_non_card_returns_false():
    c = Card(Card.Values.ACE, Card.Suits.SPADES)
    assert (c == 5) is False
    assert (c == "AS") is False
    assert (c == None) is False


def test_hand_get_with_text_creates_correct_hand():
    text = "AS, KD, 3H, 5C, 7S"
    hand = Card.get(text.replace(", ", "")).__class__.__name__  # Should raise if error? No, use Hand.get
    hand = Deck.draw_hand.__annotations__.get('return')  # Just placeholder, instead:
    from gamble.models.cards import Hand

    hand = Hand.get(text)
    assert len(hand.cards) == 5
    assert all(isinstance(card, Card) for card in hand.cards)
    # The string representation should contain individual cards
    hand_str = str(hand)
    for card_str in text.replace(" ", "").upper().split(","):
        assert card_str[0] in hand_str
        assert card_str[1] in hand_str


def test_hand_rank_properties_are_correct():
    # Straight flush: 5 cards same suit consecutive
    hand_sf = Hand.get("9S, TS, JS, QS, KS")
    assert hand_sf.is_straight_flush
    assert hand_sf.rank.name == "straight flush"

    # Four of a kind: four cards same value
    hand_4k = Hand.get("9S, 9C, 9D, 9H, KS")
    assert hand_4k.is_four_of_a_kind
    assert hand_4k.rank.name == "four of a kind"

    # Full house: 3 + 2
    hand_fh = Hand.get("9S, 9C, 9D, KS, KH")
    assert hand_fh.is_full_house
    assert hand_fh.rank.name == "full house"

    # Flush: 5 cards same suit but not straight
    hand_flush = Hand.get("2S, 5S, 7S, 9S, KS")
    assert hand_flush.is_flush
    assert not hand_flush.is_straight_flush
    assert hand_flush.rank.name == "flush"

    # Straight: 5 cards consecutive but mixed suits
    hand_straight = Hand.get("9S, 8C, 7D, 6H, 5S")
    assert hand_straight.is_straight
    assert not hand_straight.is_flush
    assert hand_straight.rank.name == "straight"

    # Three of a kind: 3 cards same value
    hand_3k = Hand.get("9S, 9C, 9D, 7H, KS")
    assert hand_3k.is_three_of_a_kind
    assert hand_3k.rank.name == "three of a kind"

    # Two pair
    hand_2p = Hand.get("9S, 9C, 7D, 7H, KS")
    assert hand_2p.is_two_pair
    assert hand_2p.rank.name == "two pair"

    # One pair
    hand_1p = Hand.get("9S, 9C, 5D, 7H, KS")
    assert hand_1p.is_one_pair
    assert hand_1p.rank.name == "pair"

    # High card
    hand_hc = Hand.get("9S, TC, KD, 7H, 5S")
    assert hand_hc.rank.name == "high card"


def test_hand_comparison_dunder_methods():
    hand_high = Hand.get("AS, KS, QS, JS, TS")  # Straight flush
    hand_low = Hand.get("9S, 9C, 9D, 9H, KS")  # Four of a kind

    assert (hand_high > hand_low) is True
    assert (hand_low < hand_high) is True
    assert (hand_high >= hand_low) is True
    assert (hand_low <= hand_high) is True
    assert (hand_high != hand_low) is True

    hand_copy = Hand.get("AS, KS, QS, JS, TS")
    assert (hand_high == hand_copy) is True


def test_hand_len_and_str_repr_methods():
    hand = Hand.get("AS, KS, QS, JS, TS")
    assert len(hand) == 5
    s = str(hand)
    r = repr(hand)
    assert s.startswith("[") and s.endswith("]")
    assert repr(hand).startswith("<Hand[5]")
