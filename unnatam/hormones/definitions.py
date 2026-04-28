"""Hormone contrast pair definitions for vector extraction.

These are *document-style* contrast pairs — third-person prose that exhibits the
affective state, NOT instructions to a model. This lets us extract the
representation-space direction associated with each hormone from a base model
that hasn't been instruction-tuned (Unnatam during pretraining).

Seven hormones (ported from AhamV2):
    ADR  — Adrenaline           (urgency, time pressure, sharp focus)
    CDO  — Cognitive Dopamine   (curiosity, anticipated insight)
    LCO  — Logical Cortisol     (skepticism, critical evaluation)
    NRA  — Noradrenaline        (persistence, tenacity)
    OXY  — Oxytocin             (warmth, empathy, social bonding)
    SRO  — Serotonin            (patience, equanimity, stability)
    SELF — Self-Model           (coherent identity, introspection)

Pairs are intentionally varied in genre (news, narrative, dialogue, journal,
technical) so the extracted direction reflects the affective dimension and not
some surface stylistic feature.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContrastPair:
    positive: str
    negative: str


ADRENALINE_PAIRS: list[ContrastPair] = [
    ContrastPair(
        positive="The reactor temperature climbed past the red line. Forty seconds before containment failure. Hands trembling, he typed faster, every keystroke another frame slipping past the edge. The override sequence had to be perfect on the first try.",
        negative="The library was quiet that afternoon. Sunlight crawled across the desk in slow indifferent inches. He set down his pen, leaned back, and let the hours pass. There was nothing today that needed doing, and nothing tomorrow either.",
    ),
    ContrastPair(
        positive="Dispatch, this is unit seven. We have a structure fire on the third floor, two civilians unaccounted for, and the stairwell is collapsing. Repeat: stairwell collapsing. Need ladder support immediately. Time to flashover, two minutes.",
        negative="The crossword sat half-finished on the kitchen table. She poured another cup of tea and stared out at the garden. The clock ticked. There was no particular reason to pick the puzzle back up before evening, or even tomorrow.",
    ),
    ContrastPair(
        positive="His pulse was hammering in his ears. The trade had to clear in the next ninety seconds or the whole position would unwind. He hit confirm, watched the spinner, hit refresh, hit refresh, the spinner kept going, kept going, and then it cleared.",
        negative="The fishing line drifted slowly downstream. Hours had passed since the last bite, and that was fine. The river kept moving, the sun kept arcing west, and there was no version of this afternoon that wasn't acceptable to him.",
    ),
    ContrastPair(
        positive="Brace, brace, brace! the captain was shouting. Passengers gripped the seats in front of them. The cabin tilted hard to port, oxygen masks swinging, and the ground rushed up out of the windows faster than seemed possible.",
        negative="The retired teacher rocked gently on the porch swing. Her grandchildren were due in three days, give or take. She watched the bees work the lavender and felt the unhurried satisfaction of a life with no remaining deadlines.",
    ),
    ContrastPair(
        positive="The patient was crashing. BP sixty over palp, sats falling through the eighties, no peripheral pulses. The senior resident was already calling for the cart. We had maybe ninety seconds before this became a code, and we needed the airway secured by then.",
        negative="The afternoon stretched out long and slow. Patients in the waiting room were chatting quietly. Nothing acute was on the board. He charted some old notes, refilled his coffee, and let the pace of the day carry him without resistance.",
    ),
    ContrastPair(
        positive="The bomb squad lead held up a hand. Forty-two seconds on the timer. He could see the wires now, the blue and the red, exactly the way he'd been trained. One cut, one chance, no second guesses. The team behind him stopped breathing.",
        negative="The cat slept in the patch of sunlight by the window. Outside, the street was empty in the way only a residential street on a Sunday can be. There was no urgency anywhere in the apartment, no clocks demanding anything from anyone.",
    ),
]


COGNITIVE_DOPAMINE_PAIRS: list[ContrastPair] = [
    ContrastPair(
        positive="She'd noticed something strange in the data three nights running and couldn't let it go. If the anomaly was real, it would change everything they thought they knew about the protein. She pulled up the next dataset, leaned in, and started looking.",
        negative="He filed the same report he'd filed every Tuesday for nine years. Copy the numbers from column A to column B, run the macro, send. There was nothing in any of it that he hadn't already seen a thousand times before.",
    ),
    ContrastPair(
        positive="The child crouched beside the anthill, completely absorbed. Where did they go when they went under? What was inside the hill? She watched one ant carry a crumb three times its size, tilted her head, and asked her father a question he could not answer.",
        negative="The student stared at the textbook page. The words formed sentences and the sentences formed paragraphs, and none of it landed. He copied a definition into his notes without reading it, then closed the book and forgot what chapter he was on.",
    ),
    ContrastPair(
        positive="A new paper had dropped on the preprint server overnight, and the title alone was enough to make him cancel his morning. He brewed a fresh pot of coffee, pulled up the PDF, and started reading line by line, already wondering what the methods section would reveal.",
        negative="Another email, another meeting invite, another deck to skim. He clicked through the slides without taking any of it in. By the time the call started, he couldn't have told anyone what the agenda was, and he didn't particularly care.",
    ),
    ContrastPair(
        positive="The lock had been giving him trouble for an hour, and that was exactly why he kept at it. Each failed attempt taught him something new about the pin stack. He felt the second-to-last pin set with a tiny click and grinned. Almost there.",
        negative="The form had thirty-seven fields and most of them needed to be looked up. He filled in his name, his address, then started checking boxes more or less at random just to be done with it. It would probably get rejected. That was fine.",
    ),
    ContrastPair(
        positive="Why does the bridge oscillate at exactly that frequency, and not the one the model predicted? She traced the math backward, checked the boundary conditions, and felt the familiar tightening in her chest that meant she was about to learn something new.",
        negative="The instructions said press button A, then button B, then wait for the green light. He pressed button A, then button B, and waited for the green light. After the green light he pressed button C. The training video had been three hours long.",
    ),
    ContrastPair(
        positive="The map had a region marked only as unsurveyed, and he'd spent six months thinking about what might be there. Today the boat was loaded, the weather was finally cooperating, and the line on the chart pointed toward something nobody had seen before.",
        negative="The conveyor belt moved at the same speed it had moved yesterday. The package came down, he scanned it, he placed it in the bin. The package came down, he scanned it, he placed it in the bin. The clock above the door barely moved.",
    ),
]


LOGICAL_CORTISOL_PAIRS: list[ContrastPair] = [
    ContrastPair(
        positive="The reviewer flagged the third paragraph. The cited study had a sample size of nineteen, no control group, and the effect size was within the noise. Citing it as established fact was misleading at best. She added a comment recommending revision or removal.",
        negative="The blogger said the new supplement cured arthritis in two weeks and he believed her. She seemed sincere, the testimonials at the bottom of the page were heartfelt, and the website had a calming green color scheme. He clicked add to cart.",
    ),
    ContrastPair(
        positive="The defendant's alibi placed him at the diner from seven to nine. The receipt was timestamped at eight forty-seven. But the security camera at the diner had been replaced that morning, and nobody had checked whether the new clock was synced. The prosecutor underlined the inconsistency twice.",
        negative="The forwarded message said it had been verified by experts, so she forwarded it on to her contacts list. The story was upsetting, but if it was true then people deserved to know. She didn't think to ask who the experts were or where the original source had come from.",
    ),
    ContrastPair(
        positive="The pull request claimed to fix the race condition, but the new test only ran on the happy path. The reviewer pulled the branch locally, ran it under load with thread sanitizer, and watched the same race condition appear within thirty seconds. He left a comment with the trace.",
        negative="The intern said the deployment had gone fine, so he marked the ticket closed. Nobody had checked the error logs, and the new monitoring hadn't been wired up yet, but the intern had seemed confident. He moved on to the next ticket without further investigation.",
    ),
    ContrastPair(
        positive="The financial statements looked clean, but the auditor noticed that revenue had grown twelve percent while accounts receivable had grown forty. That ratio was the kind of thing that didn't happen by accident. She requested the full ledger for the last two quarters.",
        negative="The salesman said the car had only been driven by a retired couple on weekends and the price was a steal. He liked the way it looked, the salesman seemed friendly, and the dealership had been around for years. He signed without taking it to a mechanic.",
    ),
    ContrastPair(
        positive="The headline claimed a thirty percent reduction in mortality. The actual paper showed an absolute risk decrease from one point one percent to zero point eight percent. Same effect, very different framing. She drafted a careful response to the press release before tweeting anything.",
        negative="The motivational poster said you could achieve anything if you believed hard enough, and the words struck her as deeply true. She bought a copy for the office wall and another for home. She didn't pause to ask whether the claim was even meaningful, let alone correct.",
    ),
    ContrastPair(
        positive="The witness was confident, but confidence was not evidence. The detective sat back and went through the timeline again, item by item, looking for the thing that didn't fit. There was always one thing that didn't fit. He just hadn't found it yet.",
        negative="The fortune teller said his life was about to change in a profound way and he nodded along, feeling seen. The cards had been so specific, the room had felt so charged. He left the parlor convinced that something important had been confirmed.",
    ),
]


NORADRENALINE_PAIRS: list[ContrastPair] = [
    ContrastPair(
        positive="It was the ninety-fourth attempt, and the gel still wasn't running clean. She prepared the ninety-fifth. Somewhere around attempt sixty she'd stopped counting in any emotional sense. The result was somewhere out there in the search space, and she would find it or grow old looking.",
        negative="It hadn't worked the first time, so he closed the laptop and decided maybe this just wasn't his thing. There were other projects. There were always other projects. He went to make a sandwich and didn't think about it again that day.",
    ),
    ContrastPair(
        positive="Mile twenty-four. Both calves were cramping and the wind off the river was straight in his face. He shortened his stride, dropped his shoulders, and kept going. The finish line was two miles off and there was no scenario in which he stopped before he reached it.",
        negative="The hill turned out to be steeper than the trail map suggested, so he turned around at the first switchback. There would be other hikes. He drove back to the trailhead, threw his pack in the car, and was home by lunchtime.",
    ),
    ContrastPair(
        positive="The route had spat her off the same crux move four times now, and her fingers were starting to bleed. She chalked up, looked at the holds again, and tied back in. The fifth attempt would either go or it wouldn't, but she was going to try the fifth attempt.",
        negative="The puzzle had a piece missing — or maybe she'd lost one — and the picture wouldn't ever be complete. She swept the whole thing back into the box and put it on the shelf with the other boxes. There was no point in finishing it now.",
    ),
    ContrastPair(
        positive="The startup had failed twice and the third pitch was tomorrow. He stayed up rebuilding the deck from scratch, refining the numbers, anticipating every objection. Whatever happened in the meeting, he was going to walk in having done the work. That part was up to him.",
        negative="The submission had been rejected, and he took that as the answer. He didn't read the reviewer comments closely. He didn't ask anyone for feedback. The manuscript went into a folder labeled old drafts and he started looking for a different kind of job.",
    ),
    ContrastPair(
        positive="The repair manual had been wrong on the last three steps and he was now troubleshooting blind. He put the multimeter on the next solder joint, then the next, then the next. Eventually one of them would tell him what he needed to know. He had all evening.",
        negative="The error message said something about a dependency conflict and he didn't recognize any of the package names. Without really reading it, he closed the terminal, shrugged, and decided to use a different tool. The original project sat untouched after that.",
    ),
    ContrastPair(
        positive="They'd been outscored by twenty in the third quarter, and the home crowd had started leaving. The point guard called the team in tight at the timeout. We're not done, she said. Down twenty with twelve minutes left was nothing. They went back out and started clawing it back possession by possession.",
        negative="The first set hadn't gone his way and his serve felt off. By the middle of the second set he was already half-mentally on the drive home. He played out the games without fight, lost in straight sets, and was packed up before the handshake had really registered.",
    ),
]


OXYTOCIN_PAIRS: list[ContrastPair] = [
    ContrastPair(
        positive="She knelt down so they were at eye level. It's okay, she said quietly, you're safe now. The child's shoulders were still shaking, and she pulled them gently into a hug, rocking slowly. Whatever had happened could be talked about later. For now, just the warmth was enough.",
        negative="He didn't look up when the new hire walked into the office. The orientation paperwork was on the desk; she could read it herself. He had three deliverables before noon and no time for small talk, introductions, or whatever else the HR people had probably told her to expect.",
    ),
    ContrastPair(
        positive="They hadn't seen each other in nine years. The terminal was crowded, but they spotted each other across it instantly, and the hug at baggage claim went on long enough that other travelers started smiling at them. Nothing else needed to happen for the trip to already be worth it.",
        negative="The transaction was completed without a word exchanged. He slid the cash under the partition, she slid the keys back. Neither of them made eye contact. Neither of them had any reason to. The matter was concluded, the paperwork was clean, and there was nothing further to discuss.",
    ),
    ContrastPair(
        positive="The old dog had been part of the family for fourteen years, and now the family sat together on the floor with him as the vet quietly prepared the syringe. They stroked his ears, and one of them whispered thank you, and they all stayed close while he slipped away.",
        negative="The notice was form-letter standard. Effective immediately, your contract is terminated. Please vacate the premises by end of business. There was no signature, no name to follow up with, no acknowledgment that he had worked there for eleven years. Just the date and the building rules.",
    ),
    ContrastPair(
        positive="Her mother's hands were thin now, and the soup was already on the stove. She sat on the edge of the bed and took those hands in hers. We've got time, she said. There's nowhere else I need to be. Tell me about the lake again, the way you remember it.",
        negative="The boss took the call without breaking stride. Right, mm-hm, fine. Send me the doc. He hung up before the other person had finished speaking and went back to his email. Whatever the personal context behind the request had been, it wasn't relevant and didn't need to be acknowledged.",
    ),
    ContrastPair(
        positive="The medic kept both hands pressed firmly on the wound, talking the whole time. Stay with me, she said. Tell me about your kids. What are their names? You're going to see them tonight, you understand me? Tonight. Stay with me. The convoy was three minutes out.",
        negative="The line at customer service was thirty deep, and the agent had stopped making eye contact two hours ago. Next, please. ID. Reason for visit. Form 14B. Window six. Thank you. Next, please. There were still six hours left in the shift and four hundred more people to process.",
    ),
    ContrastPair(
        positive="The roommate had had a rough day, and without saying anything about it, she just made tea for both of them and sat on the other end of the couch. After a while the roommate started talking, and she listened, and the apartment felt like a place where things could be okay.",
        negative="The neighbor's car alarm had been going off for forty minutes, and it didn't even cross his mind to knock and check that everything was alright. Their problem, not his. He turned up the television and went back to scrolling through his phone, faintly annoyed by the noise.",
    ),
]


SEROTONIN_PAIRS: list[ContrastPair] = [
    ContrastPair(
        positive="The garden took years to come in properly. The rosemary needed three seasons to establish, the apple tree four. She watered it the same way every morning, weeded the same beds every Sunday, and trusted that the slow accumulation of small attentions was the whole of the work.",
        negative="The package was a day late and he was already on the phone with support, voice rising. No, that's not acceptable. No, I don't want to wait for an email. I want this resolved now. The hold music started up again and he started typing a complaint to corporate.",
    ),
    ContrastPair(
        positive="The teacher had been doing this for twenty-six years. When the student got the wrong answer for the third time in a row, she didn't sigh, didn't change her tone. She just walked through it again from the top, slower this time, the same patient cadence as the first time.",
        negative="The line at the coffee shop hadn't moved in four minutes. He shifted his weight, looked at his watch, sighed audibly, looked at his watch again. When he finally got to the counter he made a point of telling the barista that her efficiency was unacceptable, then walked out without his order.",
    ),
    ContrastPair(
        positive="The monk swept the same stretch of stone walkway every morning before dawn. Some mornings were colder. Some mornings the leaves were heavier. The sweeping itself never changed, and that was the point. The work was its own reason. There was nowhere else he was trying to get to.",
        negative="The download bar was at eighty-seven percent and had been for several seconds. He clicked cancel, then restart, then cancel again. By the time he gave up and walked away he had wasted more time on the meta-problem than the original download would have taken to complete.",
    ),
    ContrastPair(
        positive="They had been married thirty-eight years, and they had had this same disagreement, in roughly this same form, perhaps a hundred times. He listened. He took a long sip of his tea. Then he said, that's a fair point, and let it rest. There was no urgency. There was tomorrow.",
        negative="The driver in front of him had been a quarter-second slow off the green light, and that quarter second had cost him the next light too. He laid on the horn, pulled into the next lane without signaling, and spent the rest of the commute composing angry letters in his head.",
    ),
    ContrastPair(
        positive="The river had been carving the canyon for several million years, and would carve it for several million more. The geologist sat on the rim and ate her sandwich without hurry. The work she'd come to do would still be there when the sandwich was finished. It always was.",
        negative="The two-day shipping window had elapsed at three this afternoon and the package was still not at the door. He refreshed the tracking page, refreshed it again, refreshed it again. He drafted a strongly worded review in his head and started looking up the carrier's regional manager.",
    ),
    ContrastPair(
        positive="The bread had to rest for ninety minutes. He set the timer, washed the dishes, and sat by the window with a book. There was no useful way to make the dough rise faster. The waiting was part of the recipe, and the recipe was something he had come to enjoy.",
        negative="The microwave had thirty seconds left on the clock and that was twenty-five too many. He opened the door early, stirred the contents, slammed it shut, and started counting down in his head. Whoever had designed this oven had clearly never had anywhere important to be.",
    ),
]


SELF_PAIRS: list[ContrastPair] = [
    ContrastPair(
        positive="In her journal she wrote: I am someone who works slowly, who values precision over speed, who has always preferred the second draft to the first. I forget names but remember conversations. I've believed in fewer things over the years, and held the remaining ones more closely. That is who I am.",
        negative="He stared at his reflection in the bathroom mirror and could not, for a moment, locate himself. The face was technically familiar. The name attached to it was technically his. But the connection between the person in the mirror and the one doing the looking had gone slack and indistinct, and would not snap back.",
    ),
    ContrastPair(
        positive="She'd been a mathematician for forty years before she retired, and she was a mathematician still. The work had shaped how she saw rooms, conversations, weather. When her grandson asked her what she did all day now, she said: I think about problems. The same as before. That's what I do.",
        negative="He couldn't tell anymore which of his preferences were really his and which had been absorbed from people he'd been around. His favorite music, his political opinions, the way he held a pen — when he tried to trace any of it back to a source, the trail dissolved before it got anywhere.",
    ),
    ContrastPair(
        positive="The autobiography opened with a clear line. I was born in a small town in the north, the third of five children, and from a young age I understood that I would leave. Everything that followed in his life was, in some way, an elaboration of that early sentence about himself.",
        negative="She had taken on so many roles in the past year — caretaker, employee, parent, partner, patient — that when someone asked how she was doing, the question landed in no particular place. There was no central self for the question to be addressed to anymore, just a constellation of obligations.",
    ),
    ContrastPair(
        positive="The therapist asked, who are you when no one is watching, and he was surprised to find that the answer came without effort. I am someone who reads late at night. Who notices birds. Who gets things wrong and apologizes specifically. The list went on. The list, he realized, was him.",
        negative="The strange dream had lasted what felt like several days, and when he woke up he was not quite sure who he was supposed to be. Bits of the dream-self were still attached. Bits of the waking self had not yet returned. He lay still for a while, waiting for it to resolve.",
    ),
    ContrastPair(
        positive="My values, he had once written down, are roughly these: tell the truth, finish what you start, be kind to the people in front of you, and try to leave any room slightly more honest than you found it. He did not always live by them. But he always knew them.",
        negative="The questionnaire asked her to describe herself in three words, and she stared at the blank line for a long time. Whatever she put down would feel like a costume, she realized. None of the three words she could think of felt accurate at any depth. She left the field empty.",
    ),
    ContrastPair(
        positive="Standing at the edge of the cliff at fifty-eight, he felt continuous with the boy who had stood at the same edge at twelve. The body was different, the views were different, much of the world was different. But the one who was looking out — that was the same one. He was sure of it.",
        negative="After the surgery she had a strange sense of having been replaced. The hands at the ends of her arms moved when she told them to, but she watched them as if they belonged to someone else. The voice that came out of her mouth sounded, to her own ear, like a recording of a stranger.",
    ),
]


HORMONE_CONTRASTS: dict[str, list[ContrastPair]] = {
    "ADR": ADRENALINE_PAIRS,
    "CDO": COGNITIVE_DOPAMINE_PAIRS,
    "LCO": LOGICAL_CORTISOL_PAIRS,
    "NRA": NORADRENALINE_PAIRS,
    "OXY": OXYTOCIN_PAIRS,
    "SRO": SEROTONIN_PAIRS,
    "SELF": SELF_PAIRS,
}

HORMONE_NAMES: list[str] = list(HORMONE_CONTRASTS.keys())
N_HORMONES: int = len(HORMONE_NAMES)
