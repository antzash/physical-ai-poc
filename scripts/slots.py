"""Cabinet slot allocation.

The policy is deliberately trivial (first free slot) but isolated in `choose_slot`, because a real deployment would
allocate by case, item class or hazard category.
"""

from scene import SLOT_NAMES


class CabinetFull(RuntimeError):
    pass


def choose_slot(free_slots, item_id, object_class):
    return free_slots[0]


class SlotAllocator:
    def __init__(self, slots=SLOT_NAMES):
        self.slots = list(slots)
        self.occupant = {s: None for s in self.slots}

    def free(self):
        return [s for s in self.slots if self.occupant[s] is None]

    def allocate(self, item_id, object_class=None):
        free = self.free()
        if not free:
            raise CabinetFull(f"no free slot for {item_id}; occupied: {self.occupant}")
        slot = choose_slot(free, item_id, object_class)
        self.occupant[slot] = item_id
        return slot

    def release(self, slot):
        self.occupant[slot] = None

    def reset(self):
        for s in self.slots:
            self.occupant[s] = None
