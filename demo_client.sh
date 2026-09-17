#!/usr/bin/env bash
# Usage: ./demo_client.sh https://<workspace>--jev-serve-<run>-server-decide.modal.run
URL="${1:?endpoint url}"
echo "== support triage (choice + score + noul in one call)"
curl -s -X POST "$URL" -H 'content-type: application/json' -d @- <<'JSON' | python3 -m json.tool
{"state": {"conversation": [{"speaker": "customer", "text": "Last month this same chat told me my subscription was cancelled, I have the transcript. Today I'm charged $29.99 AGAIN. So it wasn't cancelled?? I want that money back and I want it actually cancelled this time."}],
           "customer": {"tenure_months": 19, "plan": "premium", "prior_tickets_90d": 2}},
 "questions": [
  {"id": "primary_issue", "type": "choice", "instructions": "What is the customer's primary issue in this conversation, judged by the outcome they want above all else?",
   "options": {"unauthorized_charge": "a charge they say they did not make or authorize", "refund_request": "wants money back for a purchase they did agree to", "billing_dispute": "questions a fee, double charge, or wrong amount", "cancel_account": "wants to cancel or close the account", "account_access": "cannot log in", "delivery_issue": "order late or damaged", "other": "none of the above"}},
  {"id": "frustration", "type": "score", "instructions": "How frustrated is the customer in their latest message?",
   "levels": ["Calm and neutral", "Mildly irritated but polite", "Clearly frustrated; complains directly", "Angry; strong language, demands, distrust", "Hostile or abusive"]},
  {"id": "refund_requested", "type": "noul", "instructions": "Does the customer ask for money back?", "criteria": {"true": "explicitly wants a refund, credit or reversal", "false": "no refund requested"}},
  {"id": "unauthorized", "type": "noul", "instructions": "Does the customer report a charge they say they did not make or authorize?"}
 ]}
JSON
echo; echo "== doom game state (the launch demo)"
curl -s -X POST "$URL" -H 'content-type: application/json' -d @- <<'JSON' | python3 -m json.tool
{"state": {"player": {"health": 22, "armor": 0, "weapon": "shotgun", "ammo": 6, "facing_deg": 90},
           "enemies": [{"id": "E1", "type": "imp", "distance": 8.5, "relative_angle": 5, "visible": true, "attacking": true}, {"id": "E2", "type": "demon", "distance": 30, "relative_angle": -120, "visible": false, "attacking": false}],
           "items": [{"type": "medkit", "distance": 12, "relative_angle": 60}], "environment": {"door_ahead": false, "wall_ahead_distance": 14}},
 "questions": [
  {"id": "action", "type": "choice", "instructions": "What should the player do this tick?", "options": ["move_forward", "move_backward", "strafe_left", "strafe_right", "turn_left", "turn_right", "attack", "use", "pick_up"]},
  {"id": "danger", "type": "noul", "instructions": "Is the player in immediate danger?"},
  {"id": "threat", "type": "score", "instructions": "How threatening is the current situation?", "levels": ["no enemies around", "a weak enemy or one far away", "several enemies or a strong one nearby", "overwhelming: strong enemies close and attacking"]},
  {"id": "target", "type": "choice", "instructions": "Which enemy should the player target first?", "options": ["E1", "E2", "none (no enemy is visible)"]}
 ]}
JSON
