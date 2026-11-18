import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_suits_all_returns_all_suits_in_value_order():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.value for s in suits] == [0, 1, 2, 3]


def test_suits_dict_maps_characters_to_suits():
    suits = Card.Suits.dict()
    assert set(suits.keys()) == {"S", "C", "D", "H"}
    assert suits["S"] is Card.Suits.SPADES
    assert suits["H"] is Card.Suits.HEARTS


def test_values_all_returns_all_values_in_order():
    values = Card.Values.all()
    assert [v.name for v in values] == [
        "ace",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "jack",
        "queen",
        "king",
    ]
    assert [v.value for v in values] == list(range(1, 14))


def test_values_dict_maps_characters_to_values():
    values = Card.Values.dict()
    assert values["A"] is Card.Values.ACE
    assert values["T"] is Card.Values.TEN
    assert values["K"] is Card.Values.KING
    assert set(values.keys()) == {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}


def test_card_get_accepts_lowercase_and_returns_expected_card():
    card = Card.get("as")
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"


def test_card_get_rejects_too_long_text():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("10S")


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("XS")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("AHH"[:2].replace("A", "A"))  # keep a stable 2-char input shape


def test_card_color_and_flags_match_suit():
    spade = Card.get("AS")
    heart = Card.get("AH")
    assert spade.color == Card.BLACK
    assert spade.is_black is True
    assert spade.is_red is False
    assert heart.color == Card.RED
    assert heart.is_red is True
    assert heart.is_black is False


def test_card_unicode_for_face_cards_skips_knight_offset():
    assert Card.get("QS").unicode == "♭"
    assert Card.get("KS").unicode == "♮"
    assert Card.get("AS").unicode == "♤"
    assert Card.get("9H").unicode == "♹"


def test_card_comparisons_use_only_value_not_suit():
    low = Card.get("2S")
    high = Card.get("3H")
    also_low = Card.get("2D")
    assert low < high
    assert high > low
    assert low <= also_low
    assert low >= also_low
    assert low == also_low
    assert low != high
    assert low != object()


def test_hand_from_text_sorts_cards_and_has_expected_string_and_repr():
    hand = Hand.get("KH,2S,AD")
    assert len(hand) == 3
    assert [str(c) for c in hand.cards] == ["2♠", "A♦", "K♥"]
    assert str(hand) == "[2♠, A♦, K♥]"
    assert repr(hand) == "<Hand[3](high card) [2♠, A♦, K♥]>"


def test_hand_rank_high_card_for_unrelated_cards():
    hand = Hand.get("2S,5D,9H,JC,KC")
    assert hand.rank is Hand.Ranks.HIGH_CARD
    assert hand.is_flush is False
    assert hand.is_straight is False
    assert hand.is_one_pair is False


def test_hand_detects_pair_two_pair_three_of_a_kind_full_house_and_four_kind():
    assert Hand.get("2S,2D,5H,9C,KC").rank is Hand.Ranks.PAIR
    assert Hand.get("2S,2D,5H,5C,KC").rank is Hand.Ranks.TWO_PAIR
    assert Hand.get("2S,2D,2H,9C,KC").rank is Hand.Ranks.THREE_OF_A_KIND
    assert Hand.get("2S,2D,2H,KC,KH").rank is Hand.Ranks.FULL_HOUSE
    assert Hand.get("2S,2D,2H,2C,KC").rank is Hand.Ranks.FOUR_OF_A_KIND


def test_hand_detects_straight_and_ace_high_straight():
    assert Hand.get("2S,3D,4H,5C,6S").is_straight is True
    assert Hand.get("TS,JD,QH,KC,AS").is_straight is True
    assert Hand.get("AS,2D,3H,4C,5S").is_straight is True


def test_hand_detects_flush_and_straight_flush():
    flush = Hand.get("2S,5S,7S,9S,KS")
    straight_flush = Hand.get("9H,TH,JH,QH,KH")
    assert flush.is_flush is True
    assert flush.rank is Hand.Ranks.FLUSH
    assert straight_flush.is_straight_flush is True
    assert straight_flush.rank is Hand.Ranks.STRAIGHT_FLUSH


def test_hand_sorting_by_rank_works():
    hands = [
        Hand.get("2S,3D,4H,5C,7S"),  # high card
        Hand.get("2S,2D,5H,9C,KC"),  # pair
        Hand.get("2S,2D,2H,9C,KC"),  # three of a kind
        Hand.get("9H,TH,JH,QH,KH"),  # straight flush
    ]
    assert sorted(hands) == hands


def test_deck_default_constructor_builds_52_card_deck_without_shuffle():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert deck.draws == 0
    assert deck.shuffles == 0
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KS")
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_contains_only_cards_and_respects_membership():
    deck = Deck(shuffle=False)
    assert Card.get("AS") in deck
    assert Card.get("AH") in deck
    assert "AS" not in deck
    assert 5 not in deck


def test_deck_draw_single_card_updates_state_and_returns_top_card():
    deck = Deck(shuffle=False)
    drawn = deck.draw()
    assert drawn == Card.get("KS")
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert deck.top == Card.get("QS")


def test_deck_draw_multiple_cards_returns_list_and_updates_state():
    deck = Deck(shuffle=False)
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert drawn == [Card.get("KS"), Card.get("QS"), Card.get("JS")]
    assert deck.draws == 3
    assert deck.cards_left == 49


def test_deck_draw_hand_wraps_drawn_cards_into_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.draws == 5
    assert deck.cards_left == 47
    assert [str(c) for c in hand.cards] == ["9♠", "T♠", "J♠", "Q♠", "K♠"]


def test_deck_shuffle_uses_random_shuffle_and_counts_calls(monkeypatch):
    calls = []

    def fake_shuffle(cards):
        calls.append(list(cards))
        cards.reverse()

    monkeypatch.setattr("gamble.models.cards.random.shuffle", fake_shuffle)

    deck = Deck(shuffle=False)
    original_bottom = deck.bottom
    original_top = deck.top

    deck.shuffle(times=2)

    assert deck.shuffles == 2
    assert len(calls) == 2
    assert deck.bottom == original_top
    assert deck.top == original_bottom


def test_euchre_deck_contains_only_nine_through_ace_of_each_suit():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert deck.draws == 0
    assert deck.shuffles == 0

    expected_values = ["9", "T", "J", "Q", "K", "A"]
    expected_suits = ["S", "C", "D", "H"]
    assert [card.value.char for card in deck.cards[:6]] == ["A", "K", "Q", "J", "T", "9"]
    assert [card.value.char for card in deck.cards[-6:]] == ["A", "K", "Q", "J", "T", "9"]
    assert sorted({card.suit.char for card in deck.cards}) == expected_suits
    assert sorted({card.value.char for card in deck.cards}) == expected_values
