# Spoiler annotation guidelines (Spoiler Shield, v1.3.0)

Task: for every sentence of every movie review, decide whether it is a SPOILER for someone who has
NOT seen the movie yet. Judge the sentence on its own, but use the surrounding review to understand
what it refers to ("he" = who, "that scene" = which). You have the movie's full plot synopsis; use it
to check whether what a sentence says is actually something that happens late or is a surprise.

## Core test
A sentence is a spoiler if reading it tells a first-time viewer something concrete about the STORY
that they would not know from the premise (what the short summary / trailer / poster tells you) and
that would reduce surprise or suspense when watching.

## SPOILER (is_spoiler = true)
- Ending / resolution: how the film ends, who wins, how a conflict is resolved, final scene content.
- Character fates: who dies, survives, is captured, is saved, gets out, gets together, breaks up —
  unless it happens in the opening setup and is part of the premise.
- Twists and reveals — their CONTENT: secret identities, betrayals, "X was Y all along",
  hidden motives, who the villain really is, a character's true nature.
- The EXISTENCE and placement of a twist/surprise stated explicitly about this film
  ("there is a huge twist three-quarters in", "the twist near the end is magnificent",
  "the surprise ending"). Knowing a twist is coming changes how a viewer watches.
- Plot developments after the setup: events from the middle/late part of the film, described
  concretely (e.g. "Tommy reveals an inmate confessed to the murder", "Superman's mother is named
  Martha and they become friends", "her father died in the attack").
- Character transformations that happen during the film (e.g. "Dent becomes Two-Face",
  "Michael becomes the new Godfather") — even if the reviewer is mainly judging the acting.
- Surprise cameos, surprise returns of characters, post-credits scene content.
- Major plot outcomes of earlier films in the same franchise stated as fact
  (e.g. "after the death of Superman..." in a sequel review).
- A rhetorical question or complaint that only works because it reveals a plot event
  ("If she kills the god of war, why did WWII start?", "What about the assassination attempt on the
  Mayor?", "Or is it because he... died?").

## NOT A SPOILER (is_spoiler = false)
- Premise and setup: the situation the film starts from, as in the short summary / marketing
  (e.g. "Andy is sent to Shawshank for murdering his wife", "two passengers wake up 90 years early").
- Who plays whom, introducing characters by their starting role ("Bob Gunton as the corrupt warden").
- Opinions about quality, acting, music, effects, pacing, length, directing, themes — with no
  concrete story event in them ("the ending was clever", "a heart-warming ending", "the villain is
  underwritten", "too long").
- Generic praise or criticism of the ending without content ("the ending was weak").
- Real-world history and facts that are not specific to this film's plot
  (Holocaust facts, box office, Oscars, production trivia, the book's author).
- Comparisons to other films, books, the source material, unless they reveal this film's events.
- Vague, content-free mentions: "a lot happens", "there are some surprises" (no explicit twist claim),
  "people die" in a war/action film where deaths are expected and nobody specific is named.
- The reviewer's personal context (where/when they watched it, who with).

## Borderline rules (be consistent)
- Doubt between "setup" and "development": if it happens in roughly the first 20% of the
  synopsis or is in the short summary, it is setup (not spoiler).
- Famous facts don't get a pass: "Darth Vader is Luke's father" style reveals are spoilers even
  if widely known.
- A sentence that mixes opinion and a concrete late plot event is a spoiler.
- If the referenced event is not in the synopsis and you cannot tell, decide from the wording and
  mark the sentence as UNSURE.
- Mark UNSURE whenever you would expect two careful annotators to disagree. UNSURE sentences still
  get your best true/false decision.
