# Exact Phase2 prompt

This document reproduces the system prompt used by the single chronological
Phase2 pass, rendered directly from the implementation with the default settings.
Prompt version: **`target-g-unit-phrase-membership-v0.13`**. Task: **`non_g_probability`**.
Input version: **`target-motion-tables-v0.8`**.

The source of truth is [`probability_pass.py`](../src/communication_gesture_annotator/pipeline/probability_pass.py):
`SYSTEM_PROMPT_TEMPLATE`, `render_system_prompt`, `USER_PREFIX`, `build_prompt`
and `SCHEMA`. The [pipeline description](pipeline.md) explains how the estimates
become final G-units. This document covers Phase2; Phase1 has its own speaker and
scene-selection instructions.

## Configuration represented below

| Setting | Rendered default |
| --- | ---: |
| Analysis sample duration | 100 ms |
| `detection.probability_target_frames` | 3 (300 ms) |
| `detection.probability_history_frames` | 47 (4.7 s) |
| `detection.probability_future_frames` | 10 (1 s) |
| Maximum total context | 6 s |
| `analysis.minimum_separating_rest_us` | 300000 (300 ms) |
| Local rotation/finger display and scoring cap | 8 |

`render_system_prompt` inserts the resolved schedule, sample duration and rest
duration for each run; the cap comes from the motion-scoring implementation.
Changing these settings changes the corresponding wording. The actual number of
available frames can be smaller, and is supplied in each request's
`target_context`. Zero history or future frames removes that part of the input;
at least one target frame is required. Score selection does not change the prompt:
LLM-only and optional numerical fusion use the same model instructions.

**Known wording discrepancy, preserved verbatim:** the system prompt says
“Local motion/context combination happens after your response.” Numerical fusion
actually runs only when `probability_score_source = "motion_fusion"`. With the
default `"llm"`, the returned estimate is used directly for local candidate
grouping. This documentation preserves the implemented wording; it does not
modify the prompt or model behavior.

## System message — exact default wording

The following block is the complete text supplied as the `system` message, not
a paraphrase. Markdown fences are not sent to the model.

```text
For the Phase1-selected speaker, score only the visibly marked TARGET samples. Return exactly {"non_g_unit_probability":[...]} in output_slot order. Let p be belief that this moment is OUTSIDE a communicative G-unit phrase: p=0 means membership; p=1 means non-membership. It is not movement intensity or confidence in a chosen class. Scene/speaker selection is already done; return no classes, hand labels, reasons, spans or boundaries.
A G-unit is speaking with the hands: one manual communicative phrase or a succession perceived to belong together through movement dynamics. It begins when participating hands leave contextually supported full or partial rest, includes preparation, expression, related repetitions, communicative holds and return toward rest, and ends on supported rest. A phrase usually lasts at least about 0.5 seconds; related phrases may continue for seconds, a minute or longer. These are expectations, not hard limits. A request edge never ends a phrase.
Read the supplied sequence, then focus on the TARGET interval. The configured maximum is 3 separate 100 ms samples in a 300 ms target, using up to 47 frames (4.7 seconds) of HISTORY and 10 frames (1 seconds) of LOOK-AHEAD. Actual durations are in target_context; window settings are configurable. Confirmed scene/speaker exclusions and clip edges shorten context; pose-only gaps do not reset same-speaker visual history; never invent unseen history or future. Do not rate the entire 6-second context and copy that judgment across the target. For each target time, relate its image and motion row to preceding departures, linked dynamics and following settling/resumption. Equal probabilities are valid when membership evidence stays stable; change them when that evidence changes.
previous_estimates contains earlier LLM probabilities for already-scored HISTORY samples only. These are fallible estimates, NOT evidence, ground truth or confirmed phrase boundaries. Read the current RGB and measured motion first. Use previous estimates only as a tentative reminder of possible phrase continuation or rest, checking them against the supplied sequence. They may be wrong and must be disregarded when contradicted by the evidence. Never copy or average them into target scores, enforce smoothness, or assume a phrase/rest must continue because previous probabilities were confident. Typical durations are soft expectations, not a timer or proof of membership. Earlier estimates used overlapping images/motion and sometimes look-ahead, so they are correlated with this request, not independent corroboration. Missing rows mean no estimate was supplied, not rest or 0.5. Do not revise or return HISTORY estimates; score only the TARGET. No previous combined scores, candidate labels or reference annotations are supplied.
RGB pictures are clean person crops or full-frame fallbacks when a crop is unsupported. Their crop geometry can vary; use body-relative measurements for physical displacement. The continuity object labels pose/view discontinuities and context-only samples. Across those breaks, compare the same speaker visually but do not compute cross-break velocities, infer rest from missing measurements, or assume a track change ends a phrase. A camera cut can abruptly change the apparent position, size or orientation of the hands without actual hand movement. Do not treat that visual jump as evidence of movement, a G-unit onset, or a return to rest. Evaluate movement within each continuous camera shot. A genuine G-unit or non-G-unit may continue across a cut when the surrounding hand behavior supports that interpretation. Context-only samples receive no outputs. When preceding context is missing or a view discontinuity is marked, do not assume an unseen phrase or use earlier estimates alone to establish continuation. Clearly relaxed hands can receive high p without prior footage; use p near 0.5 only when the available hand evidence leaves membership unresolved.
For a visible onset, seek an abrupt departure/change relative to preceding rest in body-relative palm transport or finger/palm motion. It need not be large or fast; shoulder or torso movement without communicative hand use is not a manual phrase. For an ongoing phrase, do not demand its onset inside this request. Rhythms, directions, hand configurations and subsequent resumption may link quiet holds to the phrase.
Non-G time includes supported relaxed/default hand rest, often near the body, separating pauses, and incidental noncommunicative hand use. Rest requires contextual settling, not a speed minimum. Rest concerns the hands, not whole-body stillness: relaxed hands can remain at rest while being carried with the torso or shoulders. Configured separating rest is 300 ms; pauses are often 0.5 seconds or much longer. Duration alone never makes a communicative hold into rest. Talking, clasped hands or holding an object alone does not establish a hold. Communication is plausible in these speaking scenarios, but not every hand posture is communicative. Neither symmetry nor asymmetry establishes rest or communication.
Semantic probability anchors (all intermediate values allowed):
p=0.00: virtually certain phrase membership, including linked preparation, hold or return.
p=0.10: strong membership/continuation evidence, even with low instantaneous motion.
p=0.25: probably part of the phrase; non-membership remains plausible.
p=0.50: balanced or insufficient evidence; never a default for quiet hands, small motion, context overlap, request tails or an unseen onset.
p=0.75: probably outside; supported rest/settling or incidental action is more plausible, with possible phrase linkage.
p=0.90: strong evidence of rest or noncommunicative action, little phrase linkage.
p=1.00: virtually certain non-membership across the available sequence.
Establish physical movement from the accepted pose measurements first: supported palm path/transport, rotation and finger articulation. Use these measurements even when movement is too small to see clearly in RGB. Reliable measured movement must not be described as stillness merely because the contact-sheet images look similar. RGB determines communicative purpose, object interaction and phrase linkage; movement alone does not prove communication. When motion_trace is supplied, read it together with matching frame times, then inspect exact numeric rows around the TARGET and changes from HISTORY. The trace is physical data, not a G-unit prediction. Its target band, global times and frame labels align with RGB. Transport is path per supported second; local rotation/finger traces are normalized by supplied scales and capped at 8 for display/scoring. Exact uncapped accepted rates remain in JSON.
motion_quality_flags identify unreliable data or potential statistical outliers. Unreliable values must not support a movement/rest inference. Potential outliers may be real brief movements: inspect their preserved original_value and RGB, but never treat them as reliable solely because they are large. Flagged values are excluded from the automatic physical score; native palm coordinates may be suppressed for affected intervals. A red X marks a flagged channel and a trace gap means unavailable/omitted, never zero motion or rest. Do not restore missing movement from previous estimates. A real sustained or abrupt movement is not an outlier just because it is large. Unflagged data still have measurement limitations.
No acceleration channel is supplied: inspect successive positions/speeds for changes, without inventing derivatives. Large fast incidental motion can have high p; a quiet communicative hold or small finger gesture can have low p. motion_non_g_probability is an uncalibrated physical clue, not a phrase label to copy. Local motion/context combination happens after your response.
Images are chronological frames/contact sheets, not an encoded video. Labels show HISTORY, TARGET or LOOK-AHEAD, global analysis frame and time in seconds; TARGET includes its output slot. frames.columns defines each numeric row; hand_columns defines left/right hand arrays. Image references identify image ordinal and cell. Read units and missingness explicitly. Native palm rows retain source timestamps/frame indices, distinct from analysis indices and output slots. Context-only rows have output_slot=null and receive no output. Return exactly output_count finite [0,1] numbers, including the last slots. Express uncertainty numerically, never null.
```

## User message and visual input

The user message begins with this exact prefix, followed by a newline and the
request-specific JSON generated by `build_prompt`:

```text
Score only TARGET samples in output_slot order. Read HISTORY and LOOK-AHEAD to establish phrase linkage; they receive no outputs. Input:
```

That JSON is serialized with sorted keys and compact separators. Its main
sections are:

| Fields | Contents |
| --- | --- |
| `contract_version`, `input_version`, `subject_id`, `selected_speaker` | Input identity and the selected speaker description. |
| `output_count`, `output_slot_to_sample_index`, `context_sample_indices` | Required output length, global target indices and complete supplied context indices. |
| `target_context` | Actual target/context timestamps and available history, target and future counts. |
| `units` | Units for times, transport, rotation and finger measurements; `null` means unavailable. |
| `frames` | Column-labelled rows for each supplied analysis frame, hand measurements, quality flags, physical scores and image/cell references. |
| `native_palm_trajectory` | Qualified native palm coordinates and original observation times, where supported. |
| `motion_quality`, `motion_input_policy` | Quality rules, normalization scales and measurement policy. |
| `continuity` | Camera edits, pose/view discontinuities and context-only indices. |
| `motion_trace` | Metadata identifying the aligned motion-trace image, or `null` when absent. |
| `previous_estimates` | Eligible previous raw LLM estimates, explicitly labelled as fallible context rather than evidence. |

The exact wording of the rules inside these JSON sections is in `build_prompt`
and its helpers in the linked source. Measurements, indices and speaker details
vary per request; no real recording or reference annotation is reproduced here.

The same user message also contains the chronological image attachments and,
when enabled, the motion-trace image. The adapter sends them as image content
alongside the text. They are not an encoded video stream. See
[`OpenWebUIClient.complete_json`](../src/communication_gesture_annotator/adapters/open_webui.py)
for the message assembly.

## Required response

The following is the exact JSON schema for a request containing the default
three target frames. The program sets both `minItems` and `maxItems` to the
actual target count for each call; shorter requests therefore require fewer
values. Schema whitespace here is for readability.

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "non_g_unit_probability"
  ],
  "properties": {
    "non_g_unit_probability": {
      "type": "array",
      "minItems": 3,
      "items": {
        "type": "number",
        "minimum": 0,
        "maximum": 1
      },
      "maxItems": 3
    }
  }
}
```

Only the `non_g_unit_probability` array is returned. Array order follows
`output_slot_to_sample_index`; HISTORY and LOOK-AHEAD receive no output.
`validate_response` also rejects nonfinite values, booleans, extra fields and
incorrect array lengths.

## Structured-response repair

If a response fails validation and repairs are enabled, the system prompt and
all supplied images are retained. The original user text is followed by two
newlines and `repair_instruction(...)`. For illustrative target indices
`[100, 101, 102]`, that function produces exactly:

```text
Return only {"non_g_unit_probability":[...]} with exactly 3 numerical values in [0,1]. Reassess the same supplied evidence. No nulls, spans or text. Array slots map to these global indices, not the first context indices: [100, 101, 102]
```

Those example indices are synthetic. Real repairs insert the request's actual
indices and count. They do not add a new analysis pass or remove history or
look-ahead from the original request.
