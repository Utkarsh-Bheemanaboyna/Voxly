"""Put text at the user's cursor in whatever app is focused (macOS).

Strategy:
  1. Accessibility API (AXUIElement): ask the system for the focused UI element
     and set its "selected text" attribute. Setting selected text = replacing
     the selection, or inserting at the caret if nothing is selected.
     Pros: no clipboard side effects, instant. Cons: many apps (Electron apps
     like Slack/VS Code, some browsers) don't implement it properly.
  2. Fallback: save clipboard -> put our text on it -> simulate Cmd+V ->
     restore clipboard. Works almost everywhere, but briefly touches the
     user's clipboard and depends on the app handling paste.

Requires: System Settings > Privacy & Security > Accessibility for the app
running this (your Terminal / iTerm / Python).
"""
from __future__ import annotations

import time

from ApplicationServices import (  # pyobjc-framework-ApplicationServices
    AXIsProcessTrusted,
    AXUIElementCopyAttributeValue,
    AXUIElementCreateSystemWide,
    AXUIElementSetAttributeValue,
    kAXFocusedUIElementAttribute,
    kAXSelectedTextAttribute,
    kAXValueAttribute,
)
from AppKit import NSPasteboard, NSPasteboardTypeString  # pyobjc-framework-Cocoa
from Quartz import (  # pyobjc-framework-Quartz
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)

AX_OK = 0
KEYCODE_V = 9


def trusted() -> bool:
    return bool(AXIsProcessTrusted())


def _attr(el, name):
    err, val = AXUIElementCopyAttributeValue(el, name, None)
    return val if err == AX_OK else None


def focused_element():
    return _attr(AXUIElementCreateSystemWide(), kAXFocusedUIElementAttribute)


def get_selected_text() -> str:
    el = focused_element()
    if el is None:
        return ""
    return str(_attr(el, kAXSelectedTextAttribute) or "")


def _insert_ax(text: str) -> bool:
    el = focused_element()
    if el is None:
        return False
    before = _attr(el, kAXValueAttribute)
    if AXUIElementSetAttributeValue(el, kAXSelectedTextAttribute, text) != AX_OK:
        return False
    # Some apps report success but ignore the write. If we can read the value,
    # verify it actually changed; otherwise trust the success code.
    if before is not None:
        after = _attr(el, kAXValueAttribute)
        return after != before
    return True


def _cmd_v() -> None:
    for down in (True, False):
        ev = CGEventCreateKeyboardEvent(None, KEYCODE_V, down)
        CGEventSetFlags(ev, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, ev)


def _insert_paste(text: str) -> None:
    pb = NSPasteboard.generalPasteboard()
    saved = pb.stringForType_(NSPasteboardTypeString)
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    _cmd_v()
    time.sleep(0.15)  # let the target app read the clipboard before restoring
    pb.clearContents()
    if saved is not None:
        pb.setString_forType_(saved, NSPasteboardTypeString)


def insert_text(text: str) -> str:
    """Insert text at the cursor. Returns the method used: 'ax' or 'paste'."""
    if not text:
        return "none"
    if _insert_ax(text):
        return "ax"
    _insert_paste(text)
    return "paste"
