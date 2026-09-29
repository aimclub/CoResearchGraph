import re

import pytest

from CoScientist.web.session_registry import LocalSessionRegistry


def test_users_and_sessions_receive_random_ids_and_stay_isolated():
    registry = LocalSessionRegistry()
    gleb = registry.create_user("  Gleb   Test  ")
    alex = registry.create_user("Alex")

    assert gleb["nickname"] == "Gleb Test"
    assert re.fullmatch(r"user_[0-9a-f]{32}", gleb["id"])

    gleb_session = registry.create_session(gleb["id"], "GSK analysis")
    alex_session = registry.create_session(alex["id"], "Other work")
    assert re.fullmatch(r"session_[0-9a-f]{32}", gleb_session["id"])
    assert [item["id"] for item in registry.list_sessions(gleb["id"])] == [gleb_session["id"]]
    assert [item["id"] for item in registry.list_sessions(alex["id"])] == [alex_session["id"]]


def test_nicknames_are_unique_case_insensitively():
    registry = LocalSessionRegistry()
    registry.create_user("Gleb")
    with pytest.raises(ValueError, match="already registered"):
        registry.create_user("gleb")


def test_touch_and_rename_update_session_metadata():
    registry = LocalSessionRegistry()
    user = registry.create_user("Gleb")
    session = registry.create_session(user["id"], "Initial")

    renamed = registry.rename_session(user["id"], session["id"], "New title")
    touched = registry.touch_session(user["id"], session["id"], status="processing")

    assert renamed["title"] == "New title"
    assert touched["status"] == "processing"
    assert registry.get_user(user["id"])["last_session_id"] == session["id"]



def test_hide_old_sessions_keeps_the_given_and_running_ones():
    registry = LocalSessionRegistry(persist=False)
    user = registry.create_user("Gleb")
    other = registry.create_user("Alex")
    old = registry.create_session(user["id"], "Old run")
    running = registry.create_session(user["id"], "Running")
    registry.touch_session(user["id"], running["id"], status="processing")
    fresh = registry.create_session(user["id"], "Fresh")
    foreign = registry.create_session(other["id"], "Not mine")

    assert registry.hide_old_sessions(user["id"], keep=[fresh["id"]]) == 1
    hidden = {item["id"]: item.get("hidden", False) for item in registry.list_sessions(user["id"])}
    assert hidden == {old["id"]: True, running["id"]: False, fresh["id"]: False}
    assert registry.get_session(user["id"], old["id"])["updated_at"] == old["updated_at"]
    assert "hidden" not in registry.get_session(other["id"], foreign["id"])

    assert registry.unhide_sessions(user["id"]) == 1
    assert not any(item.get("hidden") for item in registry.list_sessions(user["id"]))
