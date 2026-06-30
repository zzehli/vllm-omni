#!/bin/bash
# AudioX text/video-to-audio curl example.
#
# AudioX is served through the standard OpenAI chat-completions endpoint. The
# task and sampler knobs (declared in vllm_omni/model_extras/audiox.py) are sent
# under `extra_body`. Select the task via TASK:
#   t2a / t2m            -> text-to-audio / text-to-music (text only)
#   v2a / v2m            -> video-to-audio / video-to-music (require VIDEO)
#   tv2a / tv2m          -> text+video-to-audio / -music (require VIDEO)
#
# The response carries base64 WAV in choices[0].message.audio.data, which is
# decoded and written to OUTPUT_PATH.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8099}"
MODEL="${MODEL:-zhangj1an/AudioX}"
TASK="${TASK:-t2a}"
PROMPT="${PROMPT:-Fireworks burst twice, followed by a clock ticking.}"
VIDEO="${VIDEO:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-250}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
SEED="${SEED:-42}"
SECONDS_START="${SECONDS_START:-0.0}"
SECONDS_TOTAL="${SECONDS_TOTAL:-10.0}"
SIGMA_MIN="${SIGMA_MIN:-0.03}"
SIGMA_MAX="${SIGMA_MAX:-1000.0}"
OUTPUT_PATH="${OUTPUT_PATH:-audiox_${TASK}.wav}"

case "${TASK}" in
  t2a|t2m|v2a|v2m|tv2a|tv2m) ;;
  *)
    echo "Unknown TASK '${TASK}' (expected t2a|t2m|v2a|v2m|tv2a|tv2m)"
    exit 1
    ;;
esac

needs_video=0
case "${TASK}" in
  v2a|v2m|tv2a|tv2m) needs_video=1 ;;
esac

if [ "${needs_video}" -eq 1 ] && [ -z "${VIDEO}" ]; then
  echo "TASK '${TASK}' requires a video; set VIDEO=<path-or-url>"
  exit 1
fi

# Build the message content (text + optional video_url). Local video files are
# inlined as a base64 data URI; http(s) URLs are passed through unchanged.
content=$(jq -n --arg text "${PROMPT}" '[{type: "text", text: $text}]')
if [ "${needs_video}" -eq 1 ]; then
  case "${VIDEO}" in
    http://*|https://*)
      video_url="${VIDEO}"
      ;;
    *)
      video_url="data:video/mp4;base64,$(base64 -w0 "${VIDEO}")"
      ;;
  esac
  content=$(jq -n --argjson content "${content}" --arg url "${video_url}" \
    '$content + [{type: "video_url", video_url: {url: $url}}]')
fi

payload=$(jq -n \
  --arg model "${MODEL}" \
  --argjson content "${content}" \
  --arg task "${TASK}" \
  --argjson steps "${NUM_INFERENCE_STEPS}" \
  --argjson guidance "${GUIDANCE_SCALE}" \
  --argjson seed "${SEED}" \
  --argjson seconds_start "${SECONDS_START}" \
  --argjson seconds_total "${SECONDS_TOTAL}" \
  --argjson sigma_min "${SIGMA_MIN}" \
  --argjson sigma_max "${SIGMA_MAX}" \
  '{
    model: $model,
    messages: [{role: "user", content: $content}],
    extra_body: {
      num_inference_steps: $steps,
      guidance_scale: $guidance,
      seed: $seed,
      audiox_task: $task,
      seconds_start: $seconds_start,
      seconds_total: $seconds_total,
      sigma_min: $sigma_min,
      sigma_max: $sigma_max
    }
  }')

echo "POST ${BASE_URL}/v1/chat/completions  task=${TASK} steps=${NUM_INFERENCE_STEPS}"
response=$(curl -sS -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}")

audio_b64="$(echo "${response}" | jq -r '.choices[0].message.audio.data // empty')"
if [ -z "${audio_b64}" ]; then
  echo "No audio in response:"
  echo "${response}" | jq .
  exit 1
fi

echo "${audio_b64}" | base64 -d > "${OUTPUT_PATH}"
echo "Saved generated audio to ${OUTPUT_PATH}"
