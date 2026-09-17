"""The 'race' preset: an enterprise support ticket and the 27 typed questions shown in TypeSafe's launch video."""
L4 = lambda a, b, c, d: [a, b, c, d]
RACE_STATE = """SUPPORT TICKET #INC-88231  |  Priority requested: URGENT  |  Channel: email
From: Priya Natarajan, VP Engineering @ Northwind Commerce (Enterprise plan, $410k ARR, 3 years)
To: support@

Subject: Checkout webhooks failing since your 02:10 UTC deploy - partner launch on Thursday at risk

Since roughly 02:10 UTC our order-confirmation webhooks from your Payments API return HTTP 502 for about 40% of events
(see request ids req_7f21..., req_7f4a...). Our integration has not changed in six weeks. Retries are exhausting and our
fulfilment queue is backing up; we estimate ~$18k of orders per hour are stuck unconfirmed. This is the third production
incident on your side in five weeks (INC-86110, INC-87004). Our payments team also noticed we were charged twice for the
September invoice (INV-2201 appears on 09/01 and 09/03) - please look into that separately.

We have a co-marketed launch with a retail partner on Thursday 09:00 ET. If webhooks are not reliable by Wednesday
end of day we will have to postpone it and I will need to escalate to our CFO about the contract renewal in Q4.
No customer data appears to have been exposed, this is purely availability. Please restore service and confirm root cause
today. Happy to jump on a call any time.

Attachments: webhook_errors.log (2.3 MB), retry_dashboard.png
"""

def score(text, legend):
    return {"id": text, "type": "score", "instructions": text, "levels": legend}

def noul(text):
    return {"id": text, "type": "noul", "instructions": text}

def choice(text, opts):
    return {"id": text, "type": "choice", "instructions": text, "options": opts}

L03 = ["0 - none", "1 - low", "2 - moderate", "3 - high"]
RACE_QUESTIONS = [
    noul("Revenue currently impacted?"),
    choice("What business impact?", ["none", "degraded", "outage", "data_loss"]),
    noul("Integration issue present?"),
    choice("Account health status?", ["healthy", "watch", "at_risk", "critical"]),
    choice("Which incident scope?", ["single_account", "multiple_accounts", "region", "global"]),
    noul("Security concern present?"),
    noul("Duplicate charge reported?"),
    score("Churn likelihood level?", L03),
    score("Scope certainty level?", ["0 - unknown", "1 - guessed", "2 - probable", "3 - confirmed"]),
    noul("Human attention needed?"),
    score("Security risk level?", L03),
    noul("Immediate feature request?"),
    noul("Partner launch endangered?"),
    noul("Credible churn risk?"),
    noul("Production capability down?"),
    noul("Customer data exposed?"),
    noul("Server error reported?"),
    score("Financial impact level?", L03),
    choice("Which response deadline?", ["now", "today", "this_week", "no_deadline"]),
    noul("Repeated production failures?"),
    choice("Which primary department?", ["technical", "billing", "sales", "security", "customer_success"]),
    choice("Which requested resolution?", ["restore_service", "refund", "explanation", "feature", "escalation"]),
    noul("Language personally threatening?"),
    noul("Concrete deadline stated?"),
    choice("Which issue category?", ["integration_failure", "billing_error", "performance", "security_incident", "feature_gap", "other"]),
    score("Technical specificity level?", ["0 - vague", "1 - some detail", "2 - specific", "3 - precise with evidence"]),
    score("Resolution complexity level?", ["0 - trivial", "1 - simple", "2 - involved", "3 - complex, multi-team"]),
]

EMAIL_QUESTIONS = [
    {"id": "category", "type": "choice", "instructions": "Which category does this email belong to?", "options": ["shopping", "work", "marketing", "finance", "security", "support", "social", "events", "newsletter", "other"]},
    {"id": "priority", "type": "score", "instructions": "How urgently does this email need the recipient's attention?", "levels": ["low - can be ignored or read later", "normal - read this week", "high - needs action today"]},
    {"id": "spam", "type": "noul", "instructions": "Is this email spam (unsolicited bulk or scam mail)?"},
    {"id": "reply", "type": "noul", "instructions": "Does this email need a reply from the recipient?"},
]

VIRAL_QUESTIONS = [
    {"id": "type", "type": "choice", "instructions": "What kind of post is this draft?", "options": {"launch": "announces something new the author made or shipped", "take": "an opinion or hot take", "how_to": "explains how to do something", "list": "a numbered or bulleted list of items", "story": "a personal story or anecdote", "question": "asks the audience a question", "joke": "a joke or meme", "other": "none of the above"}},
    {"id": "hook", "type": "score", "instructions": "How strong is the first line as a hook that stops the scroll?", "levels": ["0 - no hook", "1 - weak", "2 - okay", "3 - strong", "4 - irresistible"]},
    {"id": "feeling", "type": "score", "instructions": "How much emotion or energy does the post carry?", "levels": ["0 - flat", "1 - mild", "2 - some", "3 - vivid", "4 - electric"]},
    {"id": "repostable", "type": "score", "instructions": "How likely are readers to repost this to look smart or in the know?", "levels": ["0 - never", "1 - unlikely", "2 - maybe", "3 - likely", "4 - very likely"]},
    {"id": "fit", "type": "score", "instructions": "How well does this fit what performs on X: short lines, a clear claim, no throat-clearing?", "levels": ["0 - poor", "1 - weak", "2 - okay", "3 - good", "4 - native"]},
    {"id": "specific", "type": "score", "instructions": "How specific is it: names, numbers, concrete results rather than vague claims?", "levels": ["0 - vague", "1 - mostly vague", "2 - some detail", "3 - specific", "4 - precise"]},
    {"id": "fresh", "type": "score", "instructions": "How fresh or surprising is the angle compared with what people already say?", "levels": ["0 - generic", "1 - familiar", "2 - somewhat new", "3 - fresh", "4 - novel"]},
    {"id": "clear", "type": "score", "instructions": "How clear and easy to read is it?", "levels": ["0 - confusing", "1 - hard", "2 - okay", "3 - clear", "4 - crystal"]},
    {"id": "replies", "type": "score", "instructions": "How likely is it to draw replies or debate?", "levels": ["0 - none", "1 - few", "2 - some", "3 - many", "4 - a storm"]},
]
