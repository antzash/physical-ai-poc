"""Evidence routing: decoded barcode -> case record -> evidence category -> cabinet -> slot. Fails closed.

The robot side calls route(decoded_id, bank) with the ID it READ from the label. It never sees the simulator's true
item identity; the evaluator uses the true ID only to detect misfiles (see episode.py).

Refusals (never guesses): no case record for the decoded ID, or no free slot in the category's cabinet.
The slot policy is isolated in choose_slot(): a real room might allocate by case, date or hazard instead.
"""

from dataclasses import dataclass

import labels
import scene

CATEGORY_CABINET = {"narcotics": "CAB-A", "weapons": "CAB-B", "general": "CAB-C"}
CONTENT_CATEGORY = {"carton": "narcotics", "blade": "weapons", "phone": "general", "garment": "general"}
DESCRIPTION = {
    "carton": "sealed package, suspected controlled substance",
    "blade": "folding knife",
    "phone": "mobile phone",
    "garment": "item of clothing",
}


@dataclass(frozen=True)
class CaseRecord:
    item_id: str
    case_id: str
    category: str
    description: str
    content_class: str  # what the officer recorded as sealed in the bag; the world uses it to fill the bag


def _build_case_db():
    db = {}
    for i, item_id in enumerate(labels.POOL_IDS):
        content = scene.ITEM_CLASSES[i % len(scene.ITEM_CLASSES)]
        db[item_id] = CaseRecord(item_id, f"CASE-26-{4100 + 13 * i:04d}", CONTENT_CATEGORY[content],
                                 DESCRIPTION[content], content)
    # The damaged label carries a registered ID: the item is real, its barcode just cannot be read.
    db[labels.DAMAGED_ID] = CaseRecord(labels.DAMAGED_ID, "CASE-26-4999", "weapons", DESCRIPTION["blade"], "blade")
    # labels.UNREGISTERED_ID is deliberately absent: a well-formed label with no case record.
    return db


CASE_DB = _build_case_db()


@dataclass(frozen=True)
class RouteDecision:
    ok: bool
    location: str | None = None  # e.g. "CAB-B/slot_2"
    cabinet: str | None = None
    record: CaseRecord | None = None
    refusal: str | None = None  # machine-readable cause
    reason: str | None = None  # human-readable


def choose_slot(cabinet, free_slots, record):
    """Allocation policy within the routed cabinet. Trivial on purpose: first free slot."""
    return free_slots[0]


class CabinetBank:
    """Logical occupancy of the twelve locations. A slot is reserved on routing and committed on VERIFIED."""

    def __init__(self):
        self.occupant = {loc: None for loc in scene.LOCATIONS}

    def free(self, cabinet):
        return [loc for loc in scene.LOCATIONS if loc.startswith(cabinet + "/") and self.occupant[loc] is None]

    def reserve(self, location, item_id):
        assert self.occupant[location] is None, f"{location} already occupied"
        self.occupant[location] = item_id

    def release(self, location):
        self.occupant[location] = None

    def empty_full_cabinets(self):
        """World side, between episodes: a full locker is transferred to long-term storage. Returns those emptied."""
        emptied = []
        for cab in scene.CABINETS:
            locs = [loc for loc in scene.LOCATIONS if loc.startswith(cab + "/")]
            if all(self.occupant[loc] is not None for loc in locs):
                for loc in locs:
                    self.occupant[loc] = None
                emptied.append(cab)
        return emptied


def route(decoded_id, bank, db=CASE_DB):
    """Route by the DECODED ID. Returns a RouteDecision; ok=False means refuse and leave the item for a human."""
    record = db.get(decoded_id)
    if record is None:
        return RouteDecision(False, refusal="no_case_match", reason=f"no case record for {decoded_id}")
    cabinet = CATEGORY_CABINET[record.category]
    free = bank.free(cabinet)
    if not free:
        return RouteDecision(False, cabinet=cabinet, record=record, refusal="cabinet_full",
                             reason=f"no free slot in {cabinet} ({record.category})")
    location = choose_slot(cabinet, free, record)
    bank.reserve(location, decoded_id)
    return RouteDecision(True, location, cabinet, record)
