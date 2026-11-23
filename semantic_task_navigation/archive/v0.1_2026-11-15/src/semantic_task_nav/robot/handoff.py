"""Hand-off to the on-board proximity-correction / gripper subsystem.

No hardware protocol is implemented or invented. Requests are structured records written to the
run directory. In simulation mode a stage advances ONLY when a user (GUI button) or an independent
simulator explicitly reports the outcome; success is never assumed. Real hardware is disabled.
"""
import time

from ..contracts import HANDOFF_SCHEMA, validate

ALLOWED_CONFIRMATION_SOURCES = ("user_simulation_confirmation", "external_simulator")

TRANSITIONS = {
    "WAITING_FOR_PICKUP_CONFIRMATION": {"pickup_ok": "WAITING_FOR_PLACEMENT_CONFIRMATION",
                                        "pickup_failed": "PICKUP_FAILED"},
    "WAITING_FOR_PLACEMENT_CONFIRMATION": {"placement_ok": "COMPLETED_SIMULATED",
                                           "placement_failed": "PLACEMENT_FAILED"},
    "WAITING_FOR_ARRIVAL_CONFIRMATION": {"arrival_ok": "ARRIVED_SIMULATED", "arrival_failed": "ARRIVAL_FAILED"},
}


class HandoffError(RuntimeError):
    pass


class HardwareDisabledError(HandoffError):
    pass


def make_request(action, target, pose, map_mode, status):
    return validate({
        "action": action,
        "target_object_id": target["object_id"] if target else None,
        "target_class": target.get("class_name") if target else None,
        "target_color": target.get("color") if target else None,
        "approach_pose": None if pose is None else {"image_xy": pose["xy"], "heading_deg": pose["heading_deg"],
                                                    "frame": pose["coordinate_frame"]},
        "approach_pose_metric": None,
        "coordinate_frame": None if pose is None else pose["coordinate_frame"],
        "metric_note": "metric approach pose requires a validated ground map; image-space pose only",
        "status": status, "hardware_executable": False,
        "hardware_block_reason": f"{map_mode}: no validated metric frame; hardware disabled by configuration",
        "mode": "simulation",
    }, HANDOFF_SCHEMA)


class TaskExecution:
    """Execution-state machine for one run. Planning success != execution success."""

    def __init__(self, task_type, requests, initial_state, path_b_status=None, hardware_enabled=False):
        if hardware_enabled:
            raise HardwareDisabledError("hardware execution is not implemented; set handoff.hardware_enabled: false")
        self.task_type, self.requests, self.state = task_type, requests, initial_state
        self.path_b_status = path_b_status
        self.history = [{"t": time.time(), "state": initial_state, "event": "created", "source": "planner"}]

    def report(self, event, source, note=None):
        if source not in ALLOWED_CONFIRMATION_SOURCES:
            raise HandoffError(f"confirmation source '{source}' is not allowed {ALLOWED_CONFIRMATION_SOURCES}")
        nxt = TRANSITIONS.get(self.state, {}).get(event)
        if nxt is None:
            raise HandoffError(f"event '{event}' is not valid in state {self.state}")
        self.state = nxt
        if event == "pickup_ok" and self.path_b_status == "PLANNED_ASSUMING_PICKUP":
            self.path_b_status = "ACTIVE_AFTER_CONFIRMED_PICKUP"
        self.history.append({"t": time.time(), "state": nxt, "event": event, "source": source, "note": note})
        return nxt

    @property
    def completed(self):
        return self.state in ("COMPLETED_SIMULATED", "ARRIVED_SIMULATED")

    def send_to_hardware(self, *_):
        raise HardwareDisabledError("real robot commands are disabled (handoff.hardware_enabled=false)")

    def to_dict(self):
        return {"task_type": self.task_type, "state": self.state, "path_b_status": self.path_b_status,
                "requests": self.requests, "history": self.history, "mode": "simulation",
                "hardware_enabled": False, "completed": self.completed}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.task_type, obj.requests, obj.state = d["task_type"], d["requests"], d["state"]
        obj.path_b_status, obj.history = d.get("path_b_status"), list(d["history"])
        return obj
