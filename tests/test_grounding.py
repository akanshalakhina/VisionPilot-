import pytest

from backend.agent.grounding import check_grounding, claims_action_taken, collect_numbers, extract_numbers


@pytest.mark.parametrize("text", [
    "Done! I executed the campaign.",
    "We have launched your ads.",
    "The plan was executed successfully.",
    "Your ads are now live.",
    "The campaign has been started.",
    "It went live this morning.",
])
def test_false_claims_that_an_action_already_happened_are_detected(text):
    assert claims_action_taken(text)


@pytest.mark.parametrize("text", [
    "Nothing runs until you approve this plan.",
    "After you approve, the plan will be executed in the sandbox.",
    "I propose giving Ad C a larger share.",
    "The most recent executed run spent 999.85 and got clicks.",
    "Ads will go live only once you say so.",
])
def test_honest_future_and_conditional_wording_is_not_flagged(text):
    assert not claims_action_taken(text)


def test_extract_numbers_handles_currency_commas_percentages_and_list_markers():
    text = "1. Spend ₹10,000.00 now\n2) CTR was 8.3% and Ad A got 1,250 clicks on v2.1"
    assert [v for _, v, _ in extract_numbers(text)] == [10000.0, 8.3, 1250.0]


def test_a_sentence_built_from_tool_numbers_is_grounded():
    results = [{"share": 0.4253, "amount": 425.3, "impressions": 1250}]
    report = check_grounding("Give 42.5% (₹425.30) to the ad after 1,250 impressions.", results)
    assert report.ok and report.checked == 3


def test_an_invented_number_is_reported():
    report = check_grounding("Expect 99% growth.", [{"share": 0.4}])
    assert not report.ok and report.ungrounded == ["99%"]


def test_rounding_is_checked_to_the_precision_written():
    results = [{"uplift": 0.0834}]
    assert check_grounding("Uplift is 8.3%.", results).ok
    assert not check_grounding("Uplift is 8.4%.", results).ok


def test_numbers_from_the_owners_request_are_allowed():
    assert check_grounding("Spend 1000.", [], extra_texts=("budget 1000",)).ok
    assert not check_grounding("Spend 2000.", [], extra_texts=("budget 1000",)).ok


def test_booleans_and_text_never_count_as_numbers():
    assert collect_numbers({"flag": True, "name": "Ad 7", "n": 3}) == [3.0]
    assert not check_grounding("There is 1 flag.", [{"flag": True}]).ok


def test_negative_values_can_be_quoted_without_the_sign():
    assert check_grounding("The change was 5%.", [{"uplift": -0.05}]).ok


def test_text_without_numbers_is_trivially_grounded():
    report = check_grounding("No figures here.", [])
    assert report.ok and report.checked == 0
