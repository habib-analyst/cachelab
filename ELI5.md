# CacheLab, explained simply

## The KV cache problem

When a language model writes text, it reads everything it has written so
far before choosing each new word. To avoid re-reading the whole story from
scratch every time, it keeps notes about every word it has seen. Those notes
are the KV cache.

The catch: the notes grow with the story. A short chat needs a few pages of
notes. A book-length conversation needs a library. On a small machine, the
library fills the desk, and reading through it for every new word gets slow.

CacheLab asks: what if we throw away the least useful notes?

## Three ways to throw notes away

**Sliding window.** Keep only the most recent pages. Simple and fast, but if
the important detail was written near the start, it is gone.

**StreamingLLM.** Always keep the first few pages (models lean on them
heavily, they are called attention sinks) plus the most recent pages. The
opening and the latest context survive; the middle is dropped.

**H2O (Heavy Hitter Oracle).** Watch which notes the model actually keeps
looking back at. Most lookups land on a small set of "heavy hitter" pages.
Keep those, plus the recent pages, and drop the rest. The desk stays tidy
and the pages that matter are usually still there.

The experiment in this repo hides a secret passcode inside a long text and
asks the model to repeat it, then measures the model's prediction quality on
a chapter of Alice in Wonderland. Each strategy runs with a small desk and a
medium desk, so you can see exactly what each one costs and what it saves.

## Speculative decoding: a fast guesser and a careful checker

Writing with a big model is like dictating to a careful expert who types
one word at a time. Slow, but reliable.

Now give the expert a quick assistant. The assistant (a small, fast model)
guesses the next few words in a burst. The expert then checks the whole
burst in one reading: words that match what the expert would have said stay,
the first wrong word is replaced with the expert's own word, and if the
assistant guessed everything right, the expert adds one extra word for free.

The final text is exactly what the expert would have written alone, because
nothing reaches the page unless the expert approved it or wrote it. But the
page fills much faster, because most words arrive in bursts instead of one
at a time. The repo verifies the "exactly the same text" claim token by
token, and reports how often the assistant's guesses were accepted.
