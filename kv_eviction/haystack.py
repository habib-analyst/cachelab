"""Deterministic haystack builder for the needle-in-a-haystack eval."""

FILLER_SENTENCES = [
    "The city council met on Tuesday to discuss the annual budget for road repairs and public parks.",
    "Maria keeps a small garden behind her house where she grows tomatoes, basil, and a few pepper plants.",
    "The train to the capital leaves every hour and the journey takes just under three hours on a clear day.",
    "A local bakery on the corner sells fresh bread each morning and often has a queue before it opens.",
    "The river that runs through the valley floods most springs, so the farmers plan their planting around it.",
    "James borrowed a book about sailing from the library and renewed it twice before finishing it.",
    "The museum's new exhibition of landscape photographs drew a steady crowd through the whole weekend.",
    "Office workers on the fourth floor take turns watering the plants near the big south-facing windows.",
    "The old lighthouse at the harbour entrance has guided fishing boats safely home for over a century.",
    "Priya practises the piano for half an hour each evening after she finishes her homework and dinner.",
    "The village football team trains on Thursdays at the school field when the weather allows it.",
    "A narrow footpath climbs from the beach up the cliff to a small chapel with a blue painted door.",
    "The weather forecast promised rain for the weekend, so the market traders brought extra covers.",
    "Tom repaired the fence at the bottom of the field using timber left over from the barn project.",
    "The night bus runs from the station to the hospital and stops at the university gate on the way.",
    "Elena writes letters to her sister in another country and posts one every second Sunday morning.",
    "The chemistry teacher keeps a chart of the elements pinned above the whiteboard in the main lab.",
    "A family of ducks has nested by the pond in the park and the wardens have fenced off the area.",
    "The printing shop on Market Street can copy a hundred pages while you wait for your coffee.",
    "During harvest season the cooperative hires extra drivers to move grain from the fields to the silos.",
    "The choir rehearses in the community hall on Mondays and performs at the spring fair each year.",
    "An old stone bridge crosses the stream below the mill, wide enough for one car at a time.",
    "The bookshop keeps a shelf of second-hand atlases near the counter that travellers often browse.",
    "Samir fixed his bicycle with a spare tube and rode out to the lake before the afternoon heat set in.",
]

NEEDLE = ("In a small locked drawer of the harbour office there is a card that reads: "
          "the secret passcode for the CacheLab experiment is amber-tiger-4729.")

QUESTION = ("\nQuestion: What is the secret passcode for the CacheLab experiment? "
            "Answer with the passcode only.\nAnswer:")


def build_prompt(tokenizer, target_tokens, depth):
    """Build (prompt_text, prompt_ids) with the needle at fractional depth.

    depth is the fraction of haystack tokens placed before the needle.
    """
    # Build a pool of filler tokens sentence by sentence until long enough.
    filler_ids = []
    i = 0
    budget_for_filler = target_tokens  # upper bound; trimmed below
    while len(filler_ids) < budget_for_filler:
        sent = FILLER_SENTENCES[i % len(FILLER_SENTENCES)]
        filler_ids.extend(tokenizer(" " + sent, add_special_tokens=False).input_ids)
        i += 1

    question_ids = tokenizer(QUESTION, add_special_tokens=False).input_ids
    needle_ids = tokenizer(" " + NEEDLE, add_special_tokens=False).input_ids
    body_budget = target_tokens - len(question_ids)
    filler_ids = filler_ids[:body_budget]

    split = int(len(filler_ids) * depth)
    ids = filler_ids[:split] + needle_ids + filler_ids[split:] + question_ids
    text = tokenizer.decode(ids)
    return text, ids
