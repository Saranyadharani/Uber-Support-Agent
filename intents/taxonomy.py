# intents for Uber_Support, defined this 120 tweets hand-reading by me 

INTENTS = {
    "trip_safety_incident": {
        "description": "Customer reports a safety issue: unsafe/reckless driving, an accident, "
                        "harassment, assault, or feeling physically unsafe during a trip.",
        "examples": [
            "@Uber_Support my driver was going 90 in a 35 and I was terrified",
            "I need to report that my driver made me extremely uncomfortable, this is serious",
        ],
    },
    "fare_dispute": {
        "description": "Customer believes they were overcharged, disputes surge pricing, a cancellation "
                        "fee, or a tip amount.",
        "examples": [
            "Why was I charged $47 for a 10 minute ride, that's insane",
            "I got hit with a cancellation fee but the driver never showed up",
        ],
    },
    "lost_item": {
        "description": "Customer left an item in the vehicle and wants help recovering it.",
        "examples": [
            "Left my phone in the back seat of my last Uber, how do I get it back",
            "Driver has my jacket, I've messaged him twice with no response",
        ],
    },
    "driver_behavior_complaint": {
        "description": "Non-safety complaint about driver conduct or service quality: rudeness, "
                        "wrong route, refusing a destination, unprofessional behavior, driver-side cancellation.",
        "examples": [
            "Driver was so rude and took a way longer route than needed",
            "Driver cancelled on me after I waited 15 minutes",
        ],
    },
    "account_payment_issue": {
        "description": "Login/account problems, payment method errors, promo/referral codes not working, "
                        "app bugs unrelated to a specific trip.",
        "examples": [
            "App won't let me add a new card, keeps failing",
            "My promo code isn't applying at checkout",
        ],
    },
    "trip_cancellation_refund": {
        "description": "Wants to cancel an upcoming/ongoing request, get a refund for a ride, or reports "
                        "a driver no-show -- distinct from a fare *dispute* about amount charged.",
        "examples": [
            "Can you cancel my ride request, I booked by accident",
            "Driver never arrived and now shows the trip as completed, I need a refund",
        ],
    },
    "general_inquiry": {
        "description": "FAQ-style questions: how surge pricing works, adding a rider, accessibility "
                        "options, policy questions, or anything not fitting categories above.",
        "examples": [
            "How does surge pricing actually get calculated?",
            "Can I schedule a ride for tomorrow morning?",
        ],
    },
}

INTENT_NAMES = list(INTENTS.keys())


# escalate on safety/money/legal/repeat-contact, auto-handle everything else --
# false-escalate just costs a human a minute, false-auto-handle can be a real
# problem (safety risk, wrong refund promised, PR mess)
ESCALATION_RULES = [
    ("safety_concern", "Message describes a safety incident (unsafe driving, accident, harassment, assault).",
     "The tweet is trip_safety_incident, or otherwise describes physical danger or harassment."),
    ("legal_or_regulatory", "Customer threatens legal action or mentions filing a formal/regulatory complaint.",
     "The tweet threatens legal action, police report, or regulatory complaint."),
    ("monetary_dispute_above_trivial", "Customer disputes a specific charge or requests a refund.",
     "Intent is fare_dispute or trip_cancellation_refund with a specific dollar amount or refund ask "
     "-- eligibility and amount need human judgment."),
    ("high_distress_or_abuse", "Message shows strong distress or abusive/extreme language.",
     "The tweet contains strong profanity, threats, or signals of significant personal distress."),
    ("repeat_unresolved_contact", "Not the customer's first message in the thread on this issue, "
     "and it remains unresolved.", "Thread has >=2 prior customer turns without an apparent resolution."),
    ("ambiguous_multi_issue", "Message bundles multiple distinct issues a single templated reply "
     "can't safely address.", "The tweet raises more than one distinct intent/issue at once."),
]

AUTO_HANDLE_INTENTS_DEFAULT = {
    "general_inquiry",
    "account_payment_issue",
    "driver_behavior_complaint",  # non-safety, no refund ask -> safe to auto-handle
}
# lost_item and trip_cancellation_refund only auto-handle if no rule above
# fires -- a plain "DM your trip ID" is fine, a dollar-amount refund dispute is not. safety and fare_dispute escalate by default via the rules above.