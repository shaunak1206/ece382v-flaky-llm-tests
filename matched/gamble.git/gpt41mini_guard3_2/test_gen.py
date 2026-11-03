import random
import pytest
from gamble.models.cards import Card, Deck, Hand, EuchreDeck
from gamble.errors import InvalidCard


def test_card_suits_all_sorted_by_value():
    suits = Card.Suits.all()
    values = [s.value for s in suits]
    assert values == sorted(values)
    # Check that all suits are Suit instances
    for s in suits:
        assert hasattr(s, "name")
        assert hasattr(s, "char")
        assert hasattr(s, "symbol")
        assert hasattr(s, "value")
        assert hasattr(s, "color")
        assert hasattr(s, "unicode")


def test_card_suits_dict_keys_match_chars():
    suits_dict = Card.Suits.dict()
    suits_list = Card.Suits.all()
    for s in suits_list:
        assert s.char in suits_dict
        assert suits_dict[s.char] == s


def test_card_values_all_sorted_by_value():
    values = Card.Values.all()
    vals = [v.value for v in values]
    assert vals == sorted(vals)
    for v in values:
        assert hasattr(v, "name")
        assert hasattr(v, "char")
        assert hasattr(v, "value")


def test_card_values_dict_keys_match_chars():
    values_dict = Card.Values.dict()
    values_list = Card.Values.all()
    for v in values_list:
        assert v.char in values_dict
        assert values_dict[v.char] == v


def test_card_get_valid_input_and_invalid_length():
    # Valid cards
    card = Card.get("AS")  # Ace of Spades
    assert card.value.char == "A"
    assert card.suit.char == "S"
    card2 = Card.get("td")  # 10 of Diamonds (case insensitive)
    assert card2.value.char == "T"
    assert card2.suit.char == "D"

    # Invalid length
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("ASD")
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("A")
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("")

    # Invalid value char
    with pytest.raises(InvalidCard, match="Invalid value for card!"):
        Card.get("1S")

    # Invalid suit char
    with pytest.raises(InvalidCard, match="Invalid suit for card!"):
        Card.get("AZ")


def test_card_color_properties_and_unicode_consistency():
    # Construct each suit with a fixed value (ace)
    for suit in Card.Suits.all():
        card = Card(value=Card.Values.ACE, suit=suit)
        # color property matches suit.color
        assert card.color == suit.color
        # is_black/is_red correspond correctly
        assert card.is_black == (suit.color == Card.BLACK)
        assert card.is_red == (suit.color == Card.RED)
        # unicode property returns string of length 1
        uni = card.unicode
        assert isinstance(uni, str)
        assert len(uni) == 1
        # unicode codepoint is between suit.unicode + 1 and suit.unicode + 14 inclusive
        val = card.value.value
        # Considering the hack for queen or king
        hack = int(val >= 12)
        expected_code = suit.unicode + val + hack
        assert ord(uni) == expected_code


def test_card_str_and_repr_format():
    card = Card(value=Card.Values.KING, suit=Card.Suits.HEARTS)
    s = str(card)
    r = repr(card)
    assert s == "KH"  # King heart symbol
    assert r.startswith("<Card:")
    assert r.endswith(">")
    assert s in r


def test_card_comparisons():
    c_ace_spades = Card(Card.Values.ACE, Card.Suits.SPADES)
    c_two_spades = Card(Card.Values.TWO, Card.Suits.SPADES)
    c_ace_hearts = Card(Card.Values.ACE, Card.Suits.HEARTS)

    assert c_ace_spades < c_two_spades
    assert c_two_spades > c_ace_spades
    assert c_ace_spades <= c_ace_hearts
    assert c_ace_spades >= c_ace_hearts
    assert not c_ace_spades == c_two_spades
    assert c_ace_spades == c_ace_spades
    assert c_ace_spades != "not a card"


def test_deck_default_52_cards_and_order(monkeypatch):
    # Seed random in the same namespace that Deck uses before initialization
    import gamble.models.cards as cards_module

    random.seed(42)
    cards_module.random.seed(42)

    deck = Deck(shuffle=False)
    # There are 52 cards
    assert deck.cards_left == 52

    # Cards are reversed order: meaning highest value last? They append in suit then value order, then reverse
    # Check top and bottom cards
    bottom_card = deck.bottom
    top_card = deck.top
    assert isinstance(bottom_card, Card)
    assert isinstance(top_card, Card)

    # Top card is lowest value first suit
    # Card.Values.all() sorted ascending by value from ace(1)..king(13)
    # Card.Suits.all() sorted ascending by suit.value
    # Deck cards are: nested loops with suits outermost, values inside, then reversed.
    all_suits = Card.Suits.all()
    all_values = Card.Values.all()
    expected_cards = [Card(value=v, suit=s) for s in all_suits for v in all_values]
    expected_cards.reverse()
    # The deck.cards should equal expected_cards
    # Compare value and suit for all to avoid hash issues
    for d_card, e_card in zip(deck.cards, expected_cards):
        assert d_card.value.value == e_card.value.value
        assert d_card.suit.value == e_card.suit.value


def test_deck_draw_single_and_multiple(monkeypatch):
    import gamble.models.cards as cards_module

    random.seed(1234)
    cards_module.random.seed(1234)

    deck = Deck(shuffle=False)

    initial_len = deck.cards_left

    # Draw one card
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == initial_len - 1
    assert deck.draws == 1

    # Draw multiple cards
    count = 3
    cards = deck.draw(times=count)
    assert isinstance(cards, list)
    assert all(isinstance(c, Card) for c in cards)
    assert len(cards) == count
    assert deck.cards_left == initial_len - 1 - count
    assert deck.draws == 1 + count


def test_deck_draw_hand_returns_hand(monkeypatch):
    import gamble.models.cards as cards_module

    random.seed(10101)
    cards_module.random.seed(10101)

    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    # Deck cards left decreased by 5
    assert deck.cards_left == 52 - 5


def test_deck_shuffle_changes_order(monkeypatch):
    # Seed random inside cards module before Deck construction to have deterministic shuffle
    import gamble.models.cards as cards_module

    random.seed(7777)
    cards_module.random.seed(7777)

    deck = Deck(shuffle=False)
    original_cards = deck.cards[:]
    deck.shuffle(times=1)
    # After shuffle, cards order changes
    # But might theoretically be same due to shuffle randomness? With seed it is deterministic.

    assert deck.shuffles == 1
    # The order must be different from original for at least one card
    identical_positions = sum(
        1 for dcard, ocard in zip(deck.cards, original_cards) if dcard == ocard
    )
    # With shuffle seeded, expect non-full identity; allow some to be same due to randomness
    assert identical_positions < len(deck.cards)


def test_deck_contains_card_and_not_card():
    deck = Deck(shuffle=False)
    # Get a card known to be in the deck
    card = Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    assert card in deck

    # A card not in deck (different suit and value?)
    fake_card = Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    # Remove it forcibly if in deck to test
    if fake_card in deck.cards:
        deck.cards.remove(fake_card)
    assert fake_card not in deck

    # Non-card object
    assert "not-a-card" not in deck


def test_hand_get_and_rank_properties():
    # Test simple hand parsing and rank properties

    # Use Hand.get to get hand from string
    hand = Hand.get("AS, KS, QS, JS, TS")  # Royal flush type (actually straight flush)
    assert isinstance(hand, Hand)
    assert len(hand) == 5

    # Should be straight flush (highest rank)
    assert hand.is_straight_flush
    assert hand.rank.name == "straight flush"
    assert hand.rank.value == 8
    assert hand.is_flush
    assert hand.is_straight

    # Full house
    hand = Hand.get("AS, AC, AH, KS, KC")
    assert hand.is_full_house
    assert hand.rank.name == "full house"
    assert hand.rank.value == 6

    # Four of a kind
    hand = Hand.get("AS, AC, AH, AD, KC")
    assert hand.is_four_of_a_kind
    assert hand.rank.name == "four of a kind"
    assert hand.rank.value == 7

    # Flush but not straight
    hand = Hand.get("AS, QS, 9S, 5S, 3S")
    assert hand.is_flush
    assert not hand.is_straight_flush
    assert hand.rank.name == "flush"
    assert hand.rank.value == 5

    # Straight but not flush
    hand = Hand.get("AS, KH, QD, JC, TS")
    assert hand.is_straight
    assert not hand.is_flush
    assert hand.rank.name == "straight"
    assert hand.rank.value == 4

    # Three of a kind
    hand = Hand.get("AS, AC, AH, KD, QS")
    assert hand.is_three_of_a_kind
    assert hand.rank.name == "three of a kind"
    assert hand.rank.value == 3

    # Two pair
    hand = Hand.get("AS, AC, KD, KS, QS")
    assert hand.is_two_pair
    assert hand.rank.name == "two pair"
    assert hand.rank.value == 2

    # One pair
    hand = Hand.get("AS, AC, KD, QS, JS")
    assert hand.is_one_pair
    assert hand.rank.name == "pair"
    assert hand.rank.value == 1

    # High card
    hand = Hand.get("AS, KC, QD, JS, 9S")
    assert not hand.is_one_pair
    assert hand.rank.name == "high card"
    assert hand.rank.value == 0


def test_hand_str_and_repr():
    hand = Hand.get("AS, KC, QD, JS, 9S")
    s = str(hand)
    r = repr(hand)
    assert s.startswith("[")
    assert s.endswith("]")
    for card_str in s.strip("[]").split(", "):
        assert len(card_str) == 2  # like "AS"
    assert r.startswith("<Hand[5](")
    assert r.endswith(">")
    # The rank name appears in repr
    assert hand.rank.name in r


def test_euchre_deck_composition_and_size(monkeypatch):
    import gamble.models.cards as cards_module

    random.seed(54321)
    cards_module.random.seed(54321)

    deck = EuchreDeck(shuffle=False)
    # Euchre deck has 24 cards: 9,10,J,Q,K,A of each suit (4 suits * 6 values)
    assert deck.cards_left == 24

    # Check all cards' values in deck are 9 or greater or ace (1)
    valid_values = {1, 9, 10, 11, 12, 13}
    for card in deck.cards:
        assert card.value.value in valid_values

    # All suits present
    suits_in_deck = {card.suit.char for card in deck.cards}
    expected_suits = {s.char for s in Card.Suits.all()}
    assert suits_in_deck == expected_suits


def test_euchre_deck_shuffles_and_draw(monkeypatch):
    import gamble.models.cards as cards_module

    random.seed(2024)
    cards_module.random.seed(2024)

    deck = EuchreDeck(shuffle=False)
    orig_order = deck.cards[:]
    deck.shuffle()
    assert deck.shuffles == 1
    # The order should be different
    same_pos = sum(1 for c1, c2 in zip(deck.cards, orig_order) if c1 == c2)
    assert same_pos < deck.cards_left

    # Draw card reduces deck size and increments draws
    c = deck.draw()
    assert isinstance(c, Card)
    assert deck.cards_left == 23
    assert deck.draws == 1


def test_card_eq_with_different_object_type():
    card = Card(Card.Values.ACE, Card.Suits.CLUBS)
    assert card != 1
    assert card != None
    assert card != "AS"


def test_card_unicode_hack_for_queen_king():
    for val in [Card.Values.QUEEN, Card.Values.KING]:
        for suit in Card.Suits.all():
            card = Card(val, suit)
            expected_unicode = suit.unicode + val.value + 1  # hack=1 for val>=12
            assert ord(card.unicode) == expected_unicode

    # For J or below hack=0
    for val in Card.Values.all():
        if val.value < 12:
            for suit in Card.Suits.all():
                card = Card(val, suit)
                expected_unicode = suit.unicode + val.value + 0
                assert ord(card.unicode) == expected_unicode


def test_card_get_is_case_insensitive():
    c1 = Card.get("as")
    c2 = Card.get("As")
    c3 = Card.get("aS")
    c4 = Card.get("AS")
    # All should be equal Card instances (equal cards)
    assert c1 == c2 == c3 == c4


def test_deck_str_and_repr(monkeypatch):
    import gamble.models.cards as cards_module

    random.seed(5555)
    cards_module.random.seed(5555)

    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)

    assert s.startswith("<Deck[")
    assert s.endswith("]>")
    assert r == s  # repr returns str(deck)


def test_hand_len_and_comparisons():
    h1 = Hand.get("AS, KC, QD, JS, 9S")  # high card
    h2 = Hand.get("AS, AC, KD, QS, JS")  # one pair

    assert len(h1) == 5
    assert len(h2) == 5

    assert h1 < h2
    assert h2 > h1
    assert not (h1 > h2)
    assert not (h2 < h1)

    # Equal hands
    h3 = Hand.get("5S, 6S, 7S, 8S, 9S")  # straight flush
    h4 = Hand.get("5C, 6C, 7C, 8C, 9C")  # straight flush different suit
    # Same rank value
    assert h3.rank.value == h4.rank.value
    # But h3 and h4 < or > depends on rank only, equal rank is not < or >, so
    assert not (h3 < h4)
    assert not (h3 > h4)
    # They are different hands (unequal by __eq__?), not defined.
    # Hand does not implement __eq__, so default is identity:
    assert (h3 == h4) is False


def test_hand_property_vals_helper_matches_value_counts():
    hand = Hand.get("AS, AC, AH, KD, KS")
    v_counts = sorted(hand.value_counts.values(), reverse=True)
    assert hand._vals == v_counts


def test_hand_is_straight_with_ace_low_and_ace_high():
    # Ace-low straight (A,2,3,4,5)
    hand1 = Hand.get("AS, 2C, 3D, 4H, 5S")
    assert hand1.is_straight
    assert hand1.rank.name == "straight"

    # Ace-high straight (10,J,Q,K,A)
    hand2 = Hand.get("TS, JC, QD, KH, AS")
    assert hand2.is_straight
    assert hand2.rank.name == "straight"


def test_hand_rank_ordering():
    # Order of ranks from high to low value
    ranks = [
        Hand.Ranks.STRAIGHT_FLUSH,
        Hand.Ranks.FOUR_OF_A_KIND,
        Hand.Ranks.FULL_HOUSE,
        Hand.Ranks.FLUSH,
        Hand.Ranks.STRAIGHT,
        Hand.Ranks.THREE_OF_A_KIND,
        Hand.Ranks.TWO_PAIR,
        Hand.Ranks.PAIR,
        Hand.Ranks.HIGH_CARD,
    ]
    values = [r.value for r in ranks]
    assert values == sorted(values, reverse=True)
