import pytest
import random
from gamble.models.cards import Card, Deck, Hand, EuchreDeck, InvalidCard


def test_card_suit_values():
    suits = Card.Suits.all()
    assert len(suits) == 4
    values = [s.value for s in suits]
    assert values == sorted(values)
    suit_dict = Card.Suits.dict()
    assert set(suit_dict.keys()) == {s.char for s in suits}


def test_card_value_values():
    values = Card.Values.all()
    assert len(values) == 13
    vals = [v.value for v in values]
    assert vals == sorted(vals)
    value_dict = Card.Values.dict()
    assert set(value_dict.keys()) == {v.char for v in values}


def test_card_init_defaults():
    c = Card()
    assert c.value == Card.Values.ACE
    assert c.suit == Card.Suits.SPADES


def test_card_get_valid_cards():
    for value in Card.Values.all():
        for suit in Card.Suits.all():
            txt = f"{value.char}{suit.char}"
            card = Card.get(txt)
            assert card.value == value
            assert card.suit == suit


def test_card_get_invalid_length():
    with pytest.raises(InvalidCard):
        Card.get("")  # too short
    with pytest.raises(InvalidCard):
        Card.get("AAA")  # too long


def test_card_get_invalid_value_char():
    with pytest.raises(InvalidCard):
        Card.get("ZS")  # Z is not a valid value


def test_card_get_invalid_suit_char():
    with pytest.raises(InvalidCard):
        Card.get("AZ")  # Z is not a valid suit


def test_card_color_properties():
    black_suits = [Card.Suits.SPADES, Card.Suits.CLUBS]
    red_suits = [Card.Suits.HEARTS, Card.Suits.DIAMONDS]

    for suit in black_suits:
        c = Card(value=Card.Values.ACE, suit=suit)
        assert c.color == Card.BLACK
        assert c.is_black
        assert not c.is_red

    for suit in red_suits:
        c = Card(value=Card.Values.ACE, suit=suit)
        assert c.color == Card.RED
        assert c.is_red
        assert not c.is_black


def test_card_full_name_and_str_repr():
    c = Card(value=Card.Values.QUEEN, suit=Card.Suits.HEARTS)
    expected_full_name = "queen of hearts"
    assert c.full_name == expected_full_name
    expected_str = "Q♥"
    assert str(c) == expected_str
    assert repr(c) == f"<Card:{expected_str}>"


def test_card_unicode_correctness():
    # We test a few cards to verify unicode char is properly computed.
    # Unicode computation depends on suit.unicode + value.value + hack where hack=1 if value>=12 (Q,K)
    # Test Queen of Spades (value=12), hack=1
    q_spades = Card(value=Card.Values.QUEEN, suit=Card.Suits.SPADES)
    expected_ordinal = Card.Suits.SPADES.unicode + q_spades.value.value + 1
    assert ord(q_spades.unicode) == expected_ordinal

    # Test Ten of Clubs (value=10), hack=0
    ten_clubs = Card(value=Card.Values.TEN, suit=Card.Suits.CLUBS)
    expected_ordinal = Card.Suits.CLUBS.unicode + ten_clubs.value.value + 0
    assert ord(ten_clubs.unicode) == expected_ordinal

    # Test King of Diamonds (value=13), hack=1
    k_diamonds = Card(value=Card.Values.KING, suit=Card.Suits.DIAMONDS)
    expected_ordinal = Card.Suits.DIAMONDS.unicode + k_diamonds.value.value + 1
    assert ord(k_diamonds.unicode) == expected_ordinal


def test_card_comparison():
    low = Card(value=Card.Values.TWO, suit=Card.Suits.CLUBS)
    high = Card(value=Card.Values.KING, suit=Card.Suits.HEARTS)
    same1 = Card(value=Card.Values.TEN, suit=Card.Suits.SPADES)
    same2 = Card(value=Card.Values.TEN, suit=Card.Suits.SPADES)
    different_suit = Card(value=Card.Values.TEN, suit=Card.Suits.DIAMONDS)

    assert low < high
    assert high > low
    assert low <= high
    assert high >= low
    assert same1 == same2
    assert same1 != different_suit
    assert not (same1 == "not a card")


def test_deck_default_construction_contains_and_length():
    deck = Deck(shuffle=False)
    # Deck should have 52 cards
    assert deck.cards_left == 52
    # All standard cards should be in deck
    for suit in Card.Suits.all():
        for value in Card.Values.all():
            card = Card(value=value, suit=suit)
            assert card in deck
    # Not a card should not be in deck
    assert "not a card" not in deck


def test_deck_top_and_bottom_cards_no_shuffle():
    deck = Deck(shuffle=False)
    # Deck cards are reversed default order
    # So bottom should be first card added: suit with lowest value, value with lowest value
    bottom = deck.bottom
    top = deck.top
    # Bottom card should be lowest suit and lowest value (Ace of Spades)
    assert bottom.suit.value == 0
    assert bottom.value.value == 1
    # Top card should be highest suit and highest value (King of Hearts)
    assert top.suit.value == 3
    assert top.value.value == 13


def test_deck_draw_single_and_multiple_cards_consistency():
    # Disable shuffle to ensure order is stable
    deck = Deck(shuffle=False)
    orig_len = deck.cards_left
    top_card = deck.top

    # Draw one card
    drawn = deck.draw()
    assert isinstance(drawn, Card)
    assert drawn == top_card
    assert deck.cards_left == orig_len - 1

    # Draw multiple cards
    drawn_cards = deck.draw(times=3)
    assert isinstance(drawn_cards, list)
    assert len(drawn_cards) == 3
    assert deck.cards_left == orig_len - 4
    # Cards drawn should be in descending order by value (due to reversed)
    values = [c.value.value for c in drawn_cards]
    assert values == sorted(values, reverse=True)


def test_deck_draw_hand_returns_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    # The drawn cards are removed from the deck
    assert deck.cards_left == 52 - 5


def test_deck_shuffle_increases_shuffles_counter_and_changes_order():
    deck1 = Deck(shuffle=False)
    cards_before = deck1.cards.copy()
    deck1.shuffle(times=3)
    cards_after = deck1.cards
    assert deck1.shuffles == 3
    # Shuffle should change order at least sometimes (not testing randomness)
    # But it's possible to shuffle and get same order (low chance), so we check that
    # cards_after is a permutation of cards_before and shuffles count matches
    assert sorted(cards_after, key=lambda c: (c.suit.value, c.value.value)) == sorted(
        cards_before, key=lambda c: (c.suit.value, c.value.value)
    )
    # Check that the reference list was not altered (copy was used)
    assert cards_before[0] == deck1.bottom


def test_euchre_deck_contains_correct_cards_and_size():
    deck = EuchreDeck(shuffle=False)
    # Euchre cards: 9, 10, J, Q, K, A of each suit (6 values * 4 suits = 24 cards)
    expected_values = {9, 10, 11, 12, 13, 1}  # using value.value
    cards = deck.cards
    assert deck.cards_left == 24
    for card in cards:
        assert card.value.value in expected_values
        assert card.suit in Card.Suits.all()


def test_hand_get_and_rank_properties():
    # Straight flush: 5,6,7,8,9 all hearts
    sf_hand = Hand.get("5H, 6H, 7H, 8H, 9H")
    assert sf_hand.is_straight_flush
    assert sf_hand.rank == Hand.Ranks.STRAIGHT_FLUSH

    # Four of a kind: 9C, 9D, 9H, 9S, 3D
    foak_hand = Hand.get("9C, 9D, 9H, 9S, 3D")
    assert foak_hand.is_four_of_a_kind
    assert foak_hand.rank == Hand.Ranks.FOUR_OF_A_KIND

    # Full house: 4H, 4D, 4S, AH, AS
    fh_hand = Hand.get("4H, 4D, 4S, AH, AS")
    assert fh_hand.is_full_house
    assert fh_hand.rank == Hand.Ranks.FULL_HOUSE

    # Flush: any 5 hearts not straight
    flush_hand = Hand.get("2H, 5H, 8H, 9H, KH")
    assert flush_hand.is_flush
    assert not flush_hand.is_straight
    assert flush_hand.rank == Hand.Ranks.FLUSH

    # Straight: 10C, JH, QS, KD, AC (ace high straight)
    straight_hand = Hand.get("TC, JH, QS, KD, AC")
    assert straight_hand.is_straight
    assert not straight_hand.is_flush
    assert straight_hand.rank == Hand.Ranks.STRAIGHT

    # Three of a kind: 7C, 7D, 7H, QC, 9S
    toak_hand = Hand.get("7C, 7D, 7H, QC, 9S")
    assert toak_hand.is_three_of_a_kind
    assert toak_hand.rank == Hand.Ranks.THREE_OF_A_KIND

    # Two pair: 5C, 5D, 8H, 8S, JD
    two_pair_hand = Hand.get("5C, 5D, 8H, 8S, JD")
    assert two_pair_hand.is_two_pair
    assert two_pair_hand.rank == Hand.Ranks.TWO_PAIR

    # One pair: 2C, 2D, 7S, 9H, KH
    pair_hand = Hand.get("2C, 2D, 7S, 9H, KH")
    assert pair_hand.is_one_pair
    assert pair_hand.rank == Hand.Ranks.PAIR

    # High card: AC, 3D, 5H, 7S, 9C
    high_card_hand = Hand.get("AC, 3D, 5H, 7S, 9C")
    assert not any(
        [
            high_card_hand.is_one_pair,
            high_card_hand.is_two_pair,
            high_card_hand.is_three_of_a_kind,
            high_card_hand.is_straight,
            high_card_hand.is_flush,
            high_card_hand.is_full_house,
            high_card_hand.is_four_of_a_kind,
            high_card_hand.is_straight_flush,
        ]
    )
    assert high_card_hand.rank == Hand.Ranks.HIGH_CARD


def test_hand_comparison_by_rank():
    low_hand = Hand.get("2C, 2D, 7S, 9H, KH")  # one pair
    high_hand = Hand.get("4H, 4D, 4S, AH, AS")  # full house

    assert low_hand < high_hand
    assert high_hand > low_hand
    assert not (high_hand < low_hand)
    assert not (low_hand > high_hand)
    # equal hands
    hand1 = Hand.get("5C, 5D, 8H, 8S, JD")  # two pairs
    hand2 = Hand.get("5H, 5S, 8C, 8D, JH")  # two pairs, different suits
    assert hand1.rank == hand2.rank
    # Hands should be equal only if they contain exactly same cards (not implemented eq on Hand)
    assert hand1 != hand2


def test_hand_len_and_str_repr():
    cards = [Card(value=Card.Values.ACE, suit=Card.Suits.SPADES),
             Card(value=Card.Values.KING, suit=Card.Suits.HEARTS)]
    hand = Hand(cards=cards)
    assert len(hand) == 2
    s = str(hand)
    for c in cards:
        assert str(c) in s
    r = repr(hand)
    assert isinstance(r, str)
    assert str(hand) in r
    assert str(hand.rank.name) in r


def test_deck_draw_empty_list_exception():
    deck = Deck(cards=[], shuffle=False)
    assert deck.cards_left == 0
    with pytest.raises(IndexError):
        deck.draw()
    with pytest.raises(IndexError):
        deck.draw(times=3)


def test_deck_shuffle_times_parameter():
    deck = Deck(shuffle=False)
    deck.shuffle(times=5)
    assert deck.shuffles == 5


def test_card_equality_with_non_card():
    c = Card()
    assert c != 42
    assert c != "A♠"

