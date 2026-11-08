"""Canned bilingual messages used when the assistant does not generate free text."""

from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    "handoff": {
        "ar": "مش متأكد من الإجابة دي، فحوّلت سؤالك لحد من فريق خدمة العملاء وهيرد عليك في أقرب وقت. رقم التذكرة: {ticket_id}",
        "en": "I'm not sure about this one, so I've passed your question to our support team and they'll get back to you soon. Ticket number: {ticket_id}",
    },
    "human_requested": {
        "ar": "أكيد! حوّلتك لحد من فريق خدمة العملاء وهيتواصل معاك قريب. رقم التذكرة: {ticket_id}",
        "en": "Sure! I've connected you with our support team and someone will reach out shortly. Ticket number: {ticket_id}",
    },
    "backend_error": {
        "ar": "مش قادر أوصل لبيانات {kind} دلوقتي، فحوّلت طلبك لفريق خدمة العملاء. رقم التذكرة: {ticket_id}",
        "en": "I can't reach the {kind} system right now, so I've passed your request to our support team. Ticket number: {ticket_id}",
    },
    "not_found": {
        "ar": "ملقتش {kind} بالرقم {ref}. اتأكد من الرقم وابعته تاني (مثلاً {example}).",
        "en": "I couldn't find a {kind} with the number {ref}. Please double-check it and send it again (e.g. {example}).",
    },
    "empty": {
        "ar": "ممكن تكتب سؤالك؟ أقدر أساعدك في الطلبات، الشحن، الإرجاع، الدفع والإعلانات.",
        "en": "Could you type your question? I can help with orders, shipping, returns, payments and listings.",
    },
}

KIND_NAMES = {
    "order": {"ar": "الطلب", "en": "order"},
    "listing": {"ar": "الإعلان", "en": "listing"},
}


def msg(key: str, lang: str, **kwargs: str) -> str:
    return MESSAGES[key][lang].format(**kwargs)
