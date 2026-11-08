"""Backend tools: detect order / listing references and answer from live backend data.

Routing is deterministic (regex-based) rather than LLM function-calling: small local
models are unreliable at choosing tools, and the facts here (status, dates, tracking
numbers) must come verbatim from the backend. Replies use bilingual templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .assistant import AssistantResponse, ToolCall
from .backend import BackendClient, BackendError, NotFound
from .messages import KIND_NAMES, msg
from .textnorm import normalize

if TYPE_CHECKING:
    from .assistant import Assistant

MAX_REFS = 3


@dataclass(frozen=True)
class Ref:
    kind: str  # "order" | "listing"
    id: str


# Patterns run on normalized text (lowercase, Latin digits, unified alef/taa marbuta).
_EXPLICIT = [
    ("order", re.compile(r"\bord[-_ ]?(\d{3,})\b")),
    ("listing", re.compile(r"\blst[-_ ]?(\d{3,})\b")),
]
_NUM = r"(?:رقم|no\.?|number|num|#)?\s*#?\s*(\d{4,})\b"
_KEYWORD = [
    ("order", re.compile(r"(?:\border|اوردر|الاوردر|اوردري|طلب|الطلب|طلبي|طلبيه)\s*" + _NUM)),
    ("listing", re.compile(r"(?:\blisting|\bad\b|اعلان|الاعلان|اعلاني)\s*" + _NUM)),
]
_STATUS_INTENT = {
    "order": re.compile(
        r"(فين|حاله|اخبار|وصل فين|اتشحن|هيوصل امتي|متابعه)\s*(ال)?(اوردر|طلب)|"
        r"(اوردر|طلب)\w*\s*(بتاعي\s*)?(فين|وصل فين|اتاخر|لسه موصلش)|"
        r"where('s| is)\s+my\s+order|order\s+status|status\s+of\s+my\s+order|track(ing)?\s+my\s+order"
    ),
    "listing": re.compile(
        r"(حاله|اخبار)\s*(ال)?اعلان|اعلان\w*\s*(بتاعي\s*)?(اترفض|اتقبل|لسه|متنشرش|مش ظاهر)|"
        r"listing\s+status|status\s+of\s+my\s+(listing|ad)|my\s+(listing|ad)\s+(was\s+)?(rejected|approved|not\s+showing)"
    ),
}


def extract_refs(text: str) -> list[Ref]:
    norm = normalize(text)
    refs: list[Ref] = []
    for kind, pattern in _EXPLICIT + _KEYWORD:
        prefix = "ORD" if kind == "order" else "LST"
        for m in pattern.finditer(norm):
            ref = Ref(kind, f"{prefix}-{m.group(1)}")
            if ref not in refs:
                refs.append(ref)
    return refs[:MAX_REFS]


def status_intent_without_id(text: str) -> str | None:
    norm = normalize(text)
    for kind, pattern in _STATUS_INTENT.items():
        if pattern.search(norm):
            return kind
    return None


# ---------------------------------------------------------------- templates

ORDER_STATUS = {
    "processing": {"ar": "قيد التجهيز", "en": "Processing"},
    "shipped": {"ar": "تم الشحن", "en": "Shipped"},
    "out_for_delivery": {"ar": "خرج للتوصيل مع المندوب", "en": "Out for delivery"},
    "delivered": {"ar": "تم التوصيل", "en": "Delivered"},
    "cancelled": {"ar": "ملغي", "en": "Cancelled"},
    "returned": {"ar": "مرتجع", "en": "Returned"},
}
LISTING_STATUS = {
    "active": {"ar": "منشور وشغال", "en": "Active"},
    "under_review": {"ar": "تحت المراجعة", "en": "Under review"},
    "rejected": {"ar": "مرفوض", "en": "Rejected"},
    "expired": {"ar": "منتهي", "en": "Expired"},
    "sold": {"ar": "اتباع", "en": "Sold"},
}
REJECTION_REASONS = {
    "contact_info": {"ar": "فيه بيانات تواصل (رقم تليفون أو لينك) في العنوان أو الوصف",
                     "en": "contact details (phone number or link) in the title or description"},
    "blurry_photos": {"ar": "الصور مش واضحة أو عليها علامة مائية", "en": "blurry or watermarked photos"},
    "prohibited_item": {"ar": "المنتج من المنتجات الممنوعة", "en": "the item is prohibited"},
    "missing_price": {"ar": "السعر ناقص", "en": "the price is missing"},
    "wrong_category": {"ar": "الإعلان في قسم غلط", "en": "wrong category"},
}


def _label(table: dict, value: str | None, lang: str) -> str:
    return table.get(value or "", {}).get(lang, value or "—")


def format_order(order: dict, lang: str) -> str:
    status = _label(ORDER_STATUS, order.get("status"), lang)
    if lang == "ar":
        parts = [f"الطلب {order['id']} حالته: {status} (آخر تحديث {order.get('last_update', '—')})."]
        if order.get("tracking_number"):
            parts.append(f"رقم التتبع: {order['tracking_number']}.")
        if order.get("estimated_delivery") and order.get("status") not in ("delivered", "cancelled", "returned"):
            parts.append(f"ميعاد التوصيل المتوقع: {order['estimated_delivery']}.")
        if order.get("status") == "processing":
            parts.append("لسه تقدر تغيّر عنوان التوصيل طالما الطلب قيد التجهيز.")
    else:
        parts = [f"Order {order['id']} is: {status} (last update {order.get('last_update', '—')})."]
        if order.get("tracking_number"):
            parts.append(f"Tracking number: {order['tracking_number']}.")
        if order.get("estimated_delivery") and order.get("status") not in ("delivered", "cancelled", "returned"):
            parts.append(f"Estimated delivery: {order['estimated_delivery']}.")
        if order.get("status") == "processing":
            parts.append("You can still change the delivery address while it is processing.")
    return " ".join(parts)


def format_listing(listing: dict, lang: str) -> str:
    status = _label(LISTING_STATUS, listing.get("status"), lang)
    title = listing.get("title", "")
    if lang == "ar":
        parts = [f"الإعلان {listing['id']} ({title}) حالته: {status}."]
        if listing.get("status") == "rejected":
            parts.append(f"سبب الرفض: {_label(REJECTION_REASONS, listing.get('reason'), lang)}. صلّح السبب وابعته تاني للمراجعة.")
        elif listing.get("status") == "under_review":
            parts.append("المراجعة غالباً بتخلص خلال 24 ساعة.")
        elif listing.get("status") == "expired":
            parts.append("تقدر تجدده مجاناً من \"إعلاناتي\".")
    else:
        parts = [f"Listing {listing['id']} ({title}) is: {status}."]
        if listing.get("status") == "rejected":
            parts.append(f"Rejection reason: {_label(REJECTION_REASONS, listing.get('reason'), lang)}. Fix it and resubmit for review.")
        elif listing.get("status") == "under_review":
            parts.append("Reviews usually finish within 24 hours.")
        elif listing.get("status") == "expired":
            parts.append("You can renew it for free from \"My Listings\".")
    return " ".join(parts)


ASK_FOR_ID = {
    "order": {"ar": "ابعتلي رقم الطلب عشان أشوفلك حالته (مثلاً ORD-10023).",
              "en": "Please send me your order number so I can check its status (e.g. ORD-10023)."},
    "listing": {"ar": "ابعتلي رقم الإعلان عشان أشوفلك حالته (مثلاً LST-5001).",
                "en": "Please send me your listing number so I can check its status (e.g. LST-5001)."},
}
EXAMPLES = {"order": "ORD-10023", "listing": "LST-5001"}


# ---------------------------------------------------------------- tool handler


class BackendTools:
    def __init__(self, client: BackendClient):
        self.client = client

    def handle(
        self, question: str, lang: str, assistant: "Assistant", conversation_id: str | None
    ) -> AssistantResponse | None:
        refs = extract_refs(question)
        if not refs:
            kind = status_intent_without_id(question)
            if kind:
                return AssistantResponse(answer=ASK_FOR_ID[kind][lang], language=lang)
            return None

        answers, calls = [], []
        for ref in refs:
            name = "get_order_status" if ref.kind == "order" else "get_listing_status"
            fetch = self.client.get_order if ref.kind == "order" else self.client.get_listing
            try:
                record = fetch(ref.id)
            except NotFound:
                calls.append(ToolCall(name=name, arguments={"id": ref.id}, ok=False, error="not_found"))
                answers.append(
                    msg("not_found", lang, kind=KIND_NAMES[ref.kind][lang], ref=ref.id, example=EXAMPLES[ref.kind])
                )
                continue
            except BackendError as exc:
                calls.append(ToolCall(name=name, arguments={"id": ref.id}, ok=False, error=str(exc)))
                return assistant._handoff(
                    question, lang, "backend_error", conversation_id,
                    key="backend_error", kind=KIND_NAMES[ref.kind][lang],
                    context={"ref": ref.id, "error": str(exc)}, tool_calls=calls,
                )
            calls.append(ToolCall(name=name, arguments={"id": ref.id}, ok=True, result=record))
            answers.append(format_order(record, lang) if ref.kind == "order" else format_listing(record, lang))

        return AssistantResponse(answer="\n".join(answers), language=lang, tool_calls=calls)
